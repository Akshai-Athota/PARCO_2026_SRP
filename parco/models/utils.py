import torch
import torch.nn.functional as F

from rl4co.utils.decoding import (
    modify_logits_for_top_k_filtering,
    modify_logits_for_top_p_filtering,
)
from rl4co.utils.ops import gather_by_index


def tanh_clip(logits: torch.Tensor, clip_value: float, clip_mode: str = "scaled"):
    """SRP Idea 2: two tanh-clipping functional forms for a value C.

        fixed:  phi_fixed(z; C)  = C * tanh(z)        -- saturates for |z| >> 1
        scaled: phi_scaled(z; C) = C * tanh(z / C)     -- near-linear for |z| << C

    `clip_value <= 0` (or None) disables clipping and returns logits unchanged.
    """
    if not clip_value:
        return logits
    if clip_mode == "fixed":
        return clip_value * torch.tanh(logits)
    elif clip_mode == "scaled":
        return clip_value * torch.tanh(logits / clip_value)
    raise ValueError(f"Unknown tanh clip_mode {clip_mode!r}, expected 'fixed' or 'scaled'")


def parco_process_logits(
    logits: torch.Tensor,
    mask: torch.Tensor = None,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    tanh_clipping: float = 0.0,
    clip_mode: str = "scaled",
    mask_logits: bool = True,
):
    """Same as `rl4co.utils.decoding.process_logits`, except the tanh-clipping
    functional form is switchable (SRP Idea 2: fixed vs scaled). Kept local to
    `parco/` instead of monkey-patching the installed rl4co dependency."""
    if tanh_clipping > 0:
        logits = tanh_clip(logits, tanh_clipping, clip_mode)

    if mask_logits:
        assert mask is not None, "mask must be provided if mask_logits is True"
        logits[~mask] = float("-inf")

    logits = logits / temperature  # temperature scaling

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        logits = modify_logits_for_top_k_filtering(logits, top_k)

    if top_p > 0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
        logits = modify_logits_for_top_p_filtering(logits, top_p)

    return F.log_softmax(logits, dim=-1)


def replace_key_td(td, key, replacement):
    # TODO: check if best way in TensorDict?
    td.pop(key)
    td[key] = replacement
    return td


def resample_batch(td, num_agents, num_locs):
    # Remove depots until num_agents
    td.set_("num_agents", torch.full((*td.batch_size,), num_agents, device=td.device))
    if "depots" in td.keys():
        # note that if we have "depot" instead, this will automatically
        # be repeated inside the environment
        td = replace_key_td(td, "depots", td["depots"][..., :num_agents, :])

    if "pickup_et" in td.keys():
        # Ensure num_locs is even for omdcpdp
        num_locs = num_locs - 1 if num_locs % 2 == 0 else num_locs
        # also, set the "num_agents" key to the new number of agents
        td.set_("num_agents", torch.full((*td.batch_size,), num_agents, device=td.device))

    td = replace_key_td(td, "locs", td["locs"][..., :num_locs, :])

    # For early time windows
    if "pickup_et" in td.keys():
        td = replace_key_td(td, "pickup_et", td["pickup_et"][..., : num_locs // 2])
    if "delivery_et" in td.keys():
        td = replace_key_td(td, "delivery_et", td["delivery_et"][..., : num_locs // 2])

    # Capacities
    if "capacity" in td.keys():
        td = replace_key_td(td, "capacity", td["capacity"][..., :num_agents])

    if "speed" in td.keys():
        td = replace_key_td(td, "speed", td["speed"][..., :num_agents])

    if "demand" in td.keys():
        td = replace_key_td(td, "demand", td["demand"][..., :num_locs])

    return td


def get_log_likelihood(log_p, actions=None, mask=None, return_sum: bool = False):
    """Get log likelihood of selected actions

    Args:
        log_p: [batch, n_agents, (decode_len), n_nodes]
        actions: [batch, n_agents, (decode_len)]
        mask: [batch, n_agents, (decode_len)]
    """

    # NOTE: we do not use this since it is more inefficient, we do it in the decoder
    if actions is not None:
        if log_p.dim() > 3:
            log_p = gather_by_index(log_p, actions, dim=-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[mask] = 0

    assert (
        log_p > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    # TODO: check the return sum argument.
    # TODO: Also, should we sum over agents too?
    if return_sum:
        return log_p.sum(-1)  # [batch, num_agents]
    else:
        return log_p  # [batch, num_agents, (decode_len)]
