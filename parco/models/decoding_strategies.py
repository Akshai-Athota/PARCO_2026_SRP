import abc

from typing import Tuple

import torch
import torch.nn.functional as F

from einops import rearrange
from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import batchify, gather_by_index, unbatchify, unbatchify_and_gather
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

from .utils import parco_process_logits, tanh_clip

log = get_pylogger(__name__)


def parco_get_decoding_strategy(decoding_strategy, **config):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "sequential": SequentialSampling,
        "evaluate": Evaluate,
        "group_greedy": GroupGreedy,
        "group_sampling": GroupSampling,
    }

    if "multistart" in decoding_strategy:
        # raise ValueError("Multistart is not supported for multi-agent decoding")
        decoding_strategy = decoding_strategy.split("_")[-1]

    if decoding_strategy not in strategy_registry:
        log.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    return strategy_registry.get(decoding_strategy, Sampling)(**config)


class PARCODecodingStrategy(metaclass=abc.ABCMeta):
    name = "base"

    def __init__(
        self,
        num_agents: int,
        agent_handler=None,  # Agent handler
        use_init_logp: bool = True,  # Return initial logp for actions even with conflicts
        mask_handled: bool = False,  # Mask out handled actions (make logprobs 0)
        replacement_value_key: str = "current_node",  # When stopping arises (conflict or POS token), replace the value of this key
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        tanh_clipping: float = 10.0,
        tanh_clip_mode: str = "scaled",  # SRP Idea 2: "fixed" C*tanh(z) vs "scaled" C*tanh(z/C)
        multistart: bool = False,
        multisample: bool = False,
        num_samples: int = 1,
        select_best: bool = False,
        store_all_logp: bool = False,
        store_handling_mask: bool = False,  # TODO: check - memory issue?
    ) -> None:
        # PARCO-related
        if mask_handled and agent_handler is None:
            raise ValueError(
                "mask_handled is only supported when agent_handler is not None for now"
            )

        if store_all_logp and mask_handled:
            raise ValueError("store_all_logp is not supported when mask_handled is True")

        if mask_handled and use_init_logp:
            raise ValueError(
                "We should not mask out the initial action logp, rather the final action logp"
            )

        self.use_init_logp = use_init_logp
        self.mask_handled = mask_handled
        self.store_all_logp = store_all_logp
        self.store_handling_mask = store_handling_mask
        self.num_agents = num_agents
        self.agent_handler = agent_handler
        self.replacement_value_key = replacement_value_key

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.tanh_clipping = tanh_clipping
        self.tanh_clip_mode = tanh_clip_mode
        if multistart:
            raise ValueError("Multistart is not supported for multi-agent decoding")
        self.multistart = multistart
        self.multisample = multisample
        self.num_samples = num_samples
        if self.num_samples > 1:
            self.multisample = True
        self.select_best = select_best

        # initialize buffers
        self.actions = []
        self.logprobs = []
        self.handling_masks = []
        self.halting_ratios = []
        self.iter_count = 0

    @abc.abstractmethod
    def _step(
        self,
        logprobs: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict,
        action: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        raise NotImplementedError("Must be implemented by subclass")

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, action: torch.Tensor = None
    ):
        """Pre decoding hook. This method is called before the main decoding operation."""

        if self.num_samples >= 1:
            # Expand td to batch_size * num_samples
            td = batchify(td, self.num_samples)

        return td, env, self.num_samples  # TODO: check

    def post_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, RL4COEnvBase]:
        """ "
        Size depends on whether we store all log p or not. By default, we don't
        Returns:
            logprobs: [B, m, L]
            actions: [B, m, L]
        """
        assert (
            len(self.logprobs) > 0
        ), "No logprobs were collected because all environments were done. Check your initial state"
        # [B, m, L] (or is it?)
        logprobs = torch.stack(self.logprobs, -1)
        actions = torch.stack(self.actions, -1)

        if len(self.handling_masks) > 0:
            if self.handling_masks[0] is not None:
                self.handling_masks = torch.stack(self.handling_masks, 2)
            else:
                pass
        else:
            pass

        halting_ratios = (
            torch.stack(self.halting_ratios, 0) if len(self.halting_ratios) > 0 else 0
        )

        if self.num_samples > 0 and self.select_best:
            logprobs, actions, td, env = self._select_best(logprobs, actions, td, env)

        return logprobs, actions, td, env, halting_ratios

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict = None,
        agent_handler_kwargs: dict = {},
        **kwargs,
    ) -> TensorDict:
        self.iter_count += 1

        logprobs = parco_process_logits(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tanh_clipping=self.tanh_clipping,
            clip_mode=self.tanh_clip_mode,
        )

        logprobs, actions, td = self._step(logprobs, mask, td, **kwargs)
        actions_init = actions.clone()

        # Solve conflicts via agent handler
        replacement_value = td[self.replacement_value_key]  # replace with previous node

        actions, handling_mask, halting_ratio = self.agent_handler(
            actions, replacement_value, td, probs=logprobs.clone(), **agent_handler_kwargs
        )
        if self.store_handling_mask:
            # NOTE
            # this might be used for some PARCO improvements (i.e. during training)\
            # but will increase memory usage by a TON
            self.handling_masks.append(handling_mask)
        self.halting_ratios.append(halting_ratio)

        # for others
        if not self.store_all_logp:
            actions_gather = actions_init if self.use_init_logp else actions
            # logprobs: [B, m, N], actions_cur: [B, m]
            # transform logprobs to [B, m]

            logprobs = gather_by_index(logprobs, actions_gather, dim=-1)

            # We do this after gathering the logprobs
            if self.mask_handled:
                logprobs.masked_fill_(handling_mask, 0)

        td.set("action", actions)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        return td

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select the action with the highest probability."""
        selected = logprobs.argmax(dim=-1)  # [B, m, N] -> [B, m]
        if mask is not None:  # [B, m, N]
            assert (
                not (~mask).gather(-1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"
        return selected

    @staticmethod
    def sampling(logprobs, mask=None):
        """Sample an action with a multinomial distribution given by the log probabilities."""

        distribution = torch.distributions.Categorical(logits=logprobs)
        selected = distribution.sample()  # samples [B, m, N] -> [B, m]

        if mask is not None:
            # checking for bad values sampling; but is this needed?
            while (~mask).gather(-1, selected.unsqueeze(-1)).data.any():
                log.info("Sampled bad values, resampling!")
                # selected = probs.multinomial(1).squeeze(1)
                selected = distribution.sample()
            assert (
                not (~mask).gather(-1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"
        return selected

    def _select_best(self, logprobs, actions, td: TensorDict, env: RL4COEnvBase):
        # TODO: check
        rewards = env.get_reward(td, actions)
        _, max_idxs = unbatchify(rewards, self.num_samples).max(dim=-1)

        actions = unbatchify_and_gather(actions, max_idxs, self.num_samples)
        logprobs = unbatchify_and_gather(logprobs, max_idxs, self.num_samples)
        td = unbatchify_and_gather(td, max_idxs, self.num_samples)

        return logprobs, actions, td, env


class Greedy(PARCODecodingStrategy):
    name = "greedy"

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Select the action with the highest log probability"""
        selected = self.greedy(logprobs, mask)
        return logprobs, selected, td


class Sampling(PARCODecodingStrategy):
    name = "sampling"

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Sample an action with a multinomial distribution given by the log probabilities."""
        selected = self.sampling(logprobs, mask)
        return logprobs, selected, td


class SequentialSampling(PARCODecodingStrategy):
    name = "sequential"

    def __init__(self, num_agents: int, num_parallel_agents: int = 1, **kwargs):
        super().__init__(num_agents=num_agents, **kwargs)
        self.num_parallel_agents = num_parallel_agents

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict = None,
        agent_handler_kwargs: dict = {},
        **kwargs,
    ) -> TensorDict:
        self.iter_count += 1
        
        batch_size, num_agents, num_targets = logits.shape
        
        # Apply Tanh logit clipping (SRP Idea 2: fixed vs scaled form)
        if self.tanh_clipping > 0:
            logits = tanh_clip(logits, self.tanh_clipping, self.tanh_clip_mode)

        # Clone mask because we'll modify it dynamically during the loop
        step_mask = mask.clone()
        logits_masked = logits.masked_fill(~step_mask, -torch.inf)
        
        # Determine the fallback replacement value for agents that do NOT move
        replacement_value = td[self.replacement_value_key]
        if replacement_value.dim() == 1:
            replacement_value = replacement_value.unsqueeze(1).expand(-1, num_agents)
        elif replacement_value.size(1) != num_agents:
            replacement_value = replacement_value.expand(-1, num_agents)
            
        actions = replacement_value.clone()
        step_logprobs = torch.zeros(batch_size, num_agents, device=logits.device)
        b_idx = torch.arange(batch_size, device=logits.device)
        
        # Iteratively sample agents without replacing the decoder context
        for _ in range(min(self.num_parallel_agents, num_agents)):
            flat_logits = logits_masked.view(batch_size, -1)
            if self.temperature != 1.0:
                flat_logits = flat_logits / self.temperature
                
            flat_logprobs = F.log_softmax(flat_logits, dim=-1)
            probs = torch.exp(flat_logprobs)
            
            # Sample one joint (agent, target) pair
            selected_flat = torch.multinomial(probs, 1).squeeze(1)  # [B]
            
            selected_agent = selected_flat // num_targets  # [B]
            selected_target = selected_flat % num_targets  # [B]
            
            # Update the selected agent's action and logprob
            actions[b_idx, selected_agent] = selected_target
            step_logprobs[b_idx, selected_agent] = flat_logprobs[b_idx, selected_flat]
            
            # Mask out the chosen agent from being selected again in this timestep
            step_mask[b_idx, selected_agent, :] = False
            logits_masked[b_idx, selected_agent, :] = -torch.inf
            
        actions_init = actions.clone()

        # Solve any environmental conflicts via agent handler (e.g. 2 agents visiting same node)
        actions, handling_mask, halting_ratio = self.agent_handler(
            actions, replacement_value, td, probs=torch.exp(step_logprobs), **agent_handler_kwargs
        )
        
        if self.store_handling_mask:
            self.handling_masks.append(handling_mask)
        self.halting_ratios.append(halting_ratio)
        
        if not self.store_all_logp:
            if self.mask_handled:
                step_logprobs.masked_fill_(handling_mask, 0)

        td.set("action", actions)
        self.actions.append(actions)
        self.logprobs.append(step_logprobs)
        return td

    def _step(self, logprobs, mask, td, action=None, **kwargs):
        pass


class GroupDecoding(PARCODecodingStrategy):
    """Decode the M agents in sequential groups of `group_size`, using the
    logits from a single decoder forward pass (same trained checkpoint,
    zero retraining, zero model changes).

    Within a group, agents propose actions independently and in parallel
    (like Full PAR) and any conflicts are resolved by the usual
    `agent_handler`. Once a group commits, the nodes it consumed are masked
    out for every group decided afterwards, so there can be no conflict
    *across* groups.

    This interpolates between the two extremes studied in Idea 1:
        - group_size == num_agents -> identical to Full PAR (single group,
          conflicts possible, O(T/M) steps)
        - group_size == 1          -> Full AR (one agent at a time, no
          conflicts by construction, O(T) steps)
        - 1 < group_size < num_agents -> e.g. PAR-2, PAR-4
    """

    name = "group"

    def __init__(
        self, num_agents: int, group_size: int = 1, is_greedy: bool = False, **kwargs
    ) -> None:
        super().__init__(num_agents=num_agents, **kwargs)
        if self.store_all_logp:
            raise ValueError("store_all_logp is not supported by GroupDecoding")
        self.group_size = max(1, min(group_size, num_agents))
        self.is_greedy = is_greedy

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict = None,
        agent_handler_kwargs: dict = {},
        **kwargs,
    ) -> TensorDict:
        self.iter_count += 1

        # Same logit processing (temperature/top-p/top-k/tanh clipping) as
        # every other strategy, so comparisons only differ in decoding order
        logprobs = parco_process_logits(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tanh_clipping=self.tanh_clipping,
            clip_mode=self.tanh_clip_mode,
        )

        batch_size, num_agents, _num_targets = logprobs.shape
        device = logprobs.device

        replacement_value = td[self.replacement_value_key]
        if replacement_value.dim() == 1:
            replacement_value = replacement_value.unsqueeze(1).expand(-1, num_agents)
        elif replacement_value.size(1) != num_agents:
            replacement_value = replacement_value.expand(-1, num_agents)

        actions = replacement_value.clone()
        actions_init = replacement_value.clone()
        step_logprobs = torch.zeros(batch_size, num_agents, device=device)
        handling_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)

        # Node availability, progressively closed off as earlier groups commit
        remaining_mask = mask.clone()
        b_idx = torch.arange(batch_size, device=device)

        for start in range(0, num_agents, self.group_size):
            idx = list(range(start, min(start + self.group_size, num_agents)))

            group_mask = remaining_mask[:, idx, :].clone()
            group_logits_raw = logprobs[:, idx, :].clone()

            # An agent can end up with zero valid targets THIS round purely
            # because earlier groups in the same step already claimed every
            # node it could still reach (it hasn't necessarily finished, so
            # e.g. its own depot column may still be closed too). Since this
            # is not a real dead end (env.step() hasn't happened yet), give
            # it its usual conflict-loser fallback: stay in place this round.
            # (The fallback column is normally masked out of the true action
            # space, so we also need to give it a finite value to select.)
            has_valid = group_mask.any(dim=-1)
            if not has_valid.all():
                fallback_col = replacement_value[:, idx]
                deg_b, deg_a = (~has_valid).nonzero(as_tuple=True)
                group_mask[deg_b, deg_a, fallback_col[deg_b, deg_a]] = True
                group_logits_raw[deg_b, deg_a, fallback_col[deg_b, deg_a]] = 0.0

            group_logprobs = group_logits_raw.masked_fill(~group_mask, -torch.inf)

            group_actions = (
                self.greedy(group_logprobs, group_mask)
                if self.is_greedy
                else self.sampling(group_logprobs, group_mask)
            )
            actions_init[:, idx] = group_actions

            if len(idx) == 1:
                # A single agent cannot conflict with itself: skip the
                # agent_handler entirely (it assumes an agent dim >= 2 and
                # would otherwise squeeze away this size-1 dimension).
                group_resolved = group_actions
                group_handling_mask = torch.zeros_like(group_actions, dtype=torch.bool)
            else:
                group_replacement = replacement_value[:, idx]
                group_resolved, group_handling_mask, _ = self.agent_handler(
                    group_actions.clone(),
                    group_replacement,
                    td,
                    probs=group_logprobs.clone(),
                    **agent_handler_kwargs,
                )
                if group_handling_mask is None:
                    group_handling_mask = torch.zeros_like(group_actions, dtype=torch.bool)

            actions[:, idx] = group_resolved
            gather_from = group_actions if self.use_init_logp else group_resolved
            step_logprobs[:, idx] = gather_by_index(group_logprobs, gather_from, dim=-1)
            handling_mask[:, idx] = group_handling_mask

            # Close off the nodes this group actually committed to (i.e. not
            # replaced due to a lost conflict) so later groups cannot pick
            # them - this is what guarantees zero cross-group conflicts.
            committed = ~group_handling_mask
            if committed.any():
                sel_b = b_idx.unsqueeze(1).expand(-1, len(idx))[committed]
                sel_n = group_resolved[committed]
                remaining_mask[sel_b, :, sel_n] = False

        if self.mask_handled:
            step_logprobs = step_logprobs.masked_fill(handling_mask, 0)

        if self.store_handling_mask:
            self.handling_masks.append(handling_mask)
        halting_ratio = handling_mask.float().mean()
        self.halting_ratios.append(halting_ratio)

        td.set("action", actions)
        self.actions.append(actions)
        self.logprobs.append(step_logprobs)
        return td

    def _step(self, logprobs, mask, td, action=None, **kwargs):
        pass


class GroupGreedy(GroupDecoding):
    name = "group_greedy"

    def __init__(self, num_agents: int, group_size: int = 1, **kwargs) -> None:
        super().__init__(num_agents=num_agents, group_size=group_size, is_greedy=True, **kwargs)


class GroupSampling(GroupDecoding):
    name = "group_sampling"

    def __init__(self, num_agents: int, group_size: int = 1, **kwargs) -> None:
        super().__init__(
            num_agents=num_agents, group_size=group_size, is_greedy=False, **kwargs
        )


class FFSPGroupDecoding(GroupDecoding):
    """`GroupDecoding` adapted to FFSP's job/machine/wait action space, for
    SRP Idea 1 (Full PAR <-> PAR-k <-> Full AR, same checkpoint, zero
    retraining).

    In FFSP the last logit column (index `num_job`) is a *shared* wait
    action: any number of idle machines may select it in the same round
    with no real conflict, unlike a real job (which only one machine can
    claim). `GroupDecoding` assumes every action index is an exclusive
    resource, so reusing it unmodified for FFSP would (a) flag simultaneous
    "wait" picks as spurious conflicts, and (b) once any machine in an
    earlier sub-group commits to "wait", incorrectly close the wait action
    off for machines decided in a later sub-group of the same round. Both
    are fixed below; grouping, greedy/sampling selection, and cross-group
    closing of real jobs are otherwise identical to `GroupDecoding`.
    """

    def __init__(
        self,
        num_agents: int,
        num_job: int,
        group_size: int = 1,
        is_greedy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            num_agents=num_agents, group_size=group_size, is_greedy=is_greedy, **kwargs
        )
        self.num_job = num_job

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict = None,
        agent_handler_kwargs: dict = {},
        **kwargs,
    ) -> TensorDict:
        self.iter_count += 1

        logprobs = parco_process_logits(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tanh_clipping=self.tanh_clipping,
            clip_mode=self.tanh_clip_mode,
        )

        batch_size, num_agents, _num_targets = logprobs.shape
        device = logprobs.device

        replacement_value = td[self.replacement_value_key]
        if replacement_value.dim() == 1:
            replacement_value = replacement_value.unsqueeze(1).expand(-1, num_agents)
        elif replacement_value.size(1) != num_agents:
            replacement_value = replacement_value.expand(-1, num_agents)

        actions = replacement_value.clone()
        step_logprobs = torch.zeros(batch_size, num_agents, device=device)
        handling_mask = torch.zeros(
            batch_size, num_agents, dtype=torch.bool, device=device
        )

        # Node availability, progressively closed off as earlier groups
        # commit to a *real* job (wait is never closed off, see below)
        remaining_mask = mask.clone()
        b_idx = torch.arange(batch_size, device=device)

        for start in range(0, num_agents, self.group_size):
            idx = list(range(start, min(start + self.group_size, num_agents)))

            group_mask = remaining_mask[:, idx, :].clone()
            group_logits_raw = logprobs[:, idx, :].clone()

            # Same degenerate-case handling as GroupDecoding: a machine can
            # end up with zero valid targets this round purely because
            # earlier groups already claimed every job it could still
            # reach. Give it its usual fallback: wait this round (the
            # fallback column is normally masked out of the true action
            # space here too, so give it a finite logit to select).
            has_valid = group_mask.any(dim=-1)
            if not has_valid.all():
                fallback_col = replacement_value[:, idx]
                deg_b, deg_a = (~has_valid).nonzero(as_tuple=True)
                group_mask[deg_b, deg_a, fallback_col[deg_b, deg_a]] = True
                group_logits_raw[deg_b, deg_a, fallback_col[deg_b, deg_a]] = 0.0

            group_logprobs = group_logits_raw.masked_fill(~group_mask, -torch.inf)

            group_actions = (
                self.greedy(group_logprobs, group_mask)
                if self.is_greedy
                else self.sampling(group_logprobs, group_mask)
            )

            if len(idx) == 1:
                # A single machine cannot conflict with itself: skip the
                # agent_handler entirely, same as GroupDecoding.
                group_resolved = group_actions
                group_handling_mask = torch.zeros_like(group_actions, dtype=torch.bool)
            else:
                group_replacement = replacement_value[:, idx]
                group_resolved, group_handling_mask, _ = self.agent_handler(
                    group_actions.clone(),
                    group_replacement,
                    td,
                    probs=group_logprobs.clone(),
                    **agent_handler_kwargs,
                )
                if group_handling_mask is None:
                    group_handling_mask = torch.zeros_like(
                        group_actions, dtype=torch.bool
                    )
                # Picking "wait" can never be a genuine conflict (it's a
                # shared action, not an exclusive resource): un-flag any
                # machine whose own proposal was wait, even if the handler
                # matched it against another machine that also proposed
                # wait (its resolved action is wait either way).
                group_handling_mask = group_handling_mask & (
                    group_actions != self.num_job
                )

            actions[:, idx] = group_resolved
            gather_from = group_actions if self.use_init_logp else group_resolved
            step_logprobs[:, idx] = gather_by_index(group_logprobs, gather_from, dim=-1)
            handling_mask[:, idx] = group_handling_mask

            # Close off the real jobs this group actually committed to, so
            # later groups cannot pick them. "wait" is excluded from this:
            # it must stay available to every machine regardless of how
            # many earlier machines this round already chose it.
            committed = ~group_handling_mask
            if committed.any():
                sel_b = b_idx.unsqueeze(1).expand(-1, len(idx))[committed]
                sel_n = group_resolved[committed]
                real_job = sel_n != self.num_job
                sel_b, sel_n = sel_b[real_job], sel_n[real_job]
                if sel_b.numel() > 0:
                    remaining_mask[sel_b, :, sel_n] = False

        if self.mask_handled:
            step_logprobs = step_logprobs.masked_fill(handling_mask, 0)

        if self.store_handling_mask:
            self.handling_masks.append(handling_mask)
        halting_ratio = handling_mask.float().mean()
        self.halting_ratios.append(halting_ratio)

        td.set("action", actions)
        self.actions.append(actions)
        self.logprobs.append(step_logprobs)
        return td

    def _step(self, logprobs, mask, td, action=None, **kwargs):
        pass


class FFSPGroupGreedy(FFSPGroupDecoding):
    name = "ffsp_group_greedy"

    def __init__(
        self, num_agents: int, num_job: int, group_size: int = 1, **kwargs
    ) -> None:
        super().__init__(
            num_agents=num_agents,
            num_job=num_job,
            group_size=group_size,
            is_greedy=True,
            **kwargs,
        )


class FFSPGroupSampling(FFSPGroupDecoding):
    name = "ffsp_group_sampling"

    def __init__(
        self, num_agents: int, num_job: int, group_size: int = 1, **kwargs
    ) -> None:
        super().__init__(
            num_agents=num_agents,
            num_job=num_job,
            group_size=group_size,
            is_greedy=False,
            **kwargs,
        )


class Evaluate(PARCODecodingStrategy):
    name = "evaluate"

    def _step(
        self,
        logprobs: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict,
        action: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """The action is provided externally, so we just return the action"""
        selected = action
        return logprobs, selected, td


class PARCO4FFSPDecoding(PARCODecodingStrategy):

    def __init__(
        self,
        num_ma,
        num_job,
        use_pos_token: bool = False,
        tanh_clipping: float = 10.0,
        num_stages: int = 3,
    ) -> None:
        super().__init__(num_agents=num_ma)
        self.num_ma = num_ma // num_stages
        self.num_job = num_job
        self.use_pos_token = use_pos_token
        self.tanh_clipping = tanh_clipping or 1
        self.num_stages = num_stages

    def step(
        self, logits: torch.Tensor, mask: torch.Tensor, td: TensorDict = None, **kwargs
    ) -> TensorDict:

        batch_size = td.batch_size
        device = td.device

        if self.tanh_clipping > 1:
            logits = self.tanh_clipping * torch.tanh(logits)
        step_mask = ~mask.clone()
        step_mask[..., -1] = ~step_mask[..., :-1].all((1, 2)).unsqueeze(1)
        idle_machines = torch.arange(0, self.num_ma, device=device)[None, :].expand(
            *batch_size, -1
        )

        step_actions = torch.full(
            (*batch_size, self.num_ma),
            fill_value=self.num_job,
            device=device,
            dtype=torch.long,
        )
        step_logp = torch.zeros_like(step_actions, dtype=torch.float32)
        while not step_mask[..., :-1].all():
            # get the probabilities of all actions given the current mask
            logits_masked = logits.masked_fill(step_mask, -torch.inf)
            logits_reshaped = rearrange(logits_masked, "b m j -> b (j m)")
            rollout_logprobs = F.log_softmax(logits_reshaped, dim=-1)
            # perform decoding
            # shape: (batch * pomo)
            selected_action = rollout_logprobs.exp().multinomial(1).squeeze(1)
            action_logprob = rollout_logprobs.gather(
                1, selected_action.unsqueeze(1)
            ).squeeze(1)

            # translate the action
            # shape: (batch * pomo)
            job_selected = selected_action // self.num_ma
            selected_machine = selected_action % self.num_ma
            # determine which machines still have to select an action
            idle_machines = idle_machines[
                idle_machines != selected_machine[:, None]
            ].view(*batch_size, -1)

            step_actions.scatter_(
                dim=1, index=selected_machine[:, None], src=job_selected[:, None]
            )
            step_logp.scatter_(
                dim=1, index=selected_machine[:, None], src=action_logprob[:, None]
            )
            # mask job that has been selected in the current step so it cannot be selected by other agents
            step_mask = step_mask.scatter(
                -1, job_selected.view(*batch_size, 1, 1).expand(-1, self.num_ma, 1), True
            )
            if self.use_pos_token:
                # allow machines that are still idle to wait (for jobs to become available for example)
                step_mask[..., -1] = step_mask[..., -1].scatter(
                    -1, idle_machines.view(*batch_size, -1), False
                )
            else:
                step_mask[..., -1] = step_mask[..., -1].scatter(
                    -1,
                    idle_machines.view(*batch_size, -1),
                    ~(step_mask[..., :-1].all(-1)),
                )
            # lastly, mask all actions for the selected agent
            step_mask = step_mask.scatter(
                -2,
                selected_machine.view(*batch_size, 1, 1).expand(-1, 1, self.num_job + 1),
                True,
            )

        self.actions.append(step_actions)
        self.logprobs.append(step_logp)

        td.set("action", step_actions)
        return td

    def _step(self, logprobs, mask, td, action=None, **kwargs):
        pass
