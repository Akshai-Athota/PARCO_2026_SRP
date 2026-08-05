"""SRP Idea 1: Sequential (AR) vs Parallel (PAR) Decoding.

Takes a single trained PARCO checkpoint (zero retraining) and evaluates it
with four decoding modes that only change the *order* in which the M agents'
(or, for FFSP, machines') actions are committed within each environment
step:

    Full PAR  - all agents propose simultaneously (existing "greedy" mode)
    PAR-2     - agents decoded in groups of 2
    PAR-4     - agents decoded in groups of 4
    Full AR   - agents decoded one at a time, strictly sequential, no conflicts

For each mode we record wall-clock inference time and solution cost, then
report the solution-quality gap (%) relative to Full AR (the zero-conflict
reference) versus the speedup, and plot the resulting Pareto curve.

Usage (HCVRP / OMDCPDP - reads saved .npz test sets):
    python scripts/idea1_decode_modes.py --problem hcvrp \
        --checkpoint checkpoints/hcvrp/parco.ckpt \
        --datasets data/hcvrp/n100_m7_seed24610.npz

Usage (FFSP - no saved test sets in this repo; instances are generated
on the fly with a fixed seed so all 4 modes see the identical batch):
    python scripts/idea1_decode_modes.py --problem ffsp \
        --checkpoint checkpoints/ffsp/parco.ckpt \
        --ffsp_num_job 20 --ffsp_num_machine 4 --ffsp_num_stage 3 \
        --ffsp_num_instances 1000 --seed 24610
"""

import argparse
import csv
import os
import time

import torch

from rl4co.data.utils import load_npz_to_tensordict

from parco.envs import FFSPEnv, HCVRPEnv, OMDCPDPEnv
from parco.envs.ffsp.generator import FFSPGenerator
from parco.models import PARCORLModule
from parco.tasks.eval import get_dataloader

# Per-problem (name, decode_type, group_size) tuples. group_size=None means
# "the whole group at once" (Full PAR), resolved internally per-policy:
#   - HCVRP/OMDCPDP ("greedy"): all M agents in a single one-shot proposal.
#   - FFSP ("group_greedy", None): all machines in a stage in one group.
# "Full AR" is expressed as group_size=1 everywhere, i.e. one agent/machine
# committed at a time with zero possible conflicts by construction.
DECODE_MODES = {
    "hcvrp": [
        ("Full PAR", "greedy", None),
        ("PAR-2", "group_greedy", 2),
        ("PAR-4", "group_greedy", 4),
        ("Full AR", "group_greedy", 1),
    ],
    "omdcpdp": [
        ("Full PAR", "greedy", None),
        ("PAR-2", "group_greedy", 2),
        ("PAR-4", "group_greedy", 4),
        ("Full AR", "group_greedy", 1),
    ],
    "ffsp": [
        ("Full PAR", "group_greedy", None),
        ("PAR-2", "group_greedy", 2),
        ("PAR-4", "group_greedy", 4),
        ("Full AR", "group_greedy", 1),
    ],
}


def run_mode(policy, env, dataloader, device, decode_type, group_size):
    kwargs = {} if group_size is None else {"group_size": group_size}
    costs, steps_list = [], []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for batch in dataloader:
            td = env.reset(batch.to(device))
            out = policy(
                td, env, decode_type=decode_type, return_actions=False, **kwargs
            )
            costs.extend((-out["reward"]).tolist())
            steps_list.append(out["steps"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - t0

    mean_cost = sum(costs) / len(costs)
    return mean_cost, total_time, steps_list[0]


def build_ffsp_dataset(num_instances, num_job, num_machine, num_stage, seed):
    """FFSP has no saved test-set files in this repo (train/val/test
    instances are normally generated on the fly by FFSPGenerator). Generate
    one fixed-seed batch here so every decode mode is evaluated on the
    identical set of instances."""
    generator = FFSPGenerator(
        num_stage=num_stage, num_machine=num_machine, num_job=num_job
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        td = generator(batch_size=[num_instances])
    return td


def get_datasets(args):
    """Returns a list of (name, TensorDict) pairs to evaluate against."""
    if args.problem == "ffsp":
        name = f"ffsp_n{args.ffsp_num_job}_m{args.ffsp_num_machine}_s{args.ffsp_num_stage}_seed{args.seed}"
        td = build_ffsp_dataset(
            args.ffsp_num_instances,
            args.ffsp_num_job,
            args.ffsp_num_machine,
            args.ffsp_num_stage,
            args.seed,
        )
        return [(name, td)]

    datasets = args.datasets or [
        f"./data/{args.problem}/{f}"
        for f in os.listdir(f"./data/{args.problem}")
        if "sol" not in f
    ]
    return [
        (os.path.basename(d), load_npz_to_tensordict(d)) for d in sorted(datasets)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="hcvrp")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="results/idea1")
    parser.add_argument("--seed", type=int, default=24610)
    parser.add_argument(
        "--decode_type",
        type=str,
        default=None,
        help="If set, run only this single decode mode (e.g. 'group_greedy' "
        "with --group_size 1 for a guaranteed Full-AR/sequential-only test) "
        "instead of the full Full PAR/PAR-2/PAR-4/Full AR sweep.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="Only used together with --decode_type.",
    )
    # FFSP-only (no saved test-set files exist for it in this repo)
    parser.add_argument("--ffsp_num_job", type=int, default=20)
    parser.add_argument("--ffsp_num_machine", type=int, default=4)
    parser.add_argument("--ffsp_num_stage", type=int, default=3)
    parser.add_argument("--ffsp_num_instances", type=int, default=1000)
    args = parser.parse_args()

    if args.problem not in DECODE_MODES:
        raise ValueError(
            f"Unknown --problem '{args.problem}'. Choose from {list(DECODE_MODES)}"
        )
    if args.decode_type is not None:
        decode_modes = [(args.decode_type, args.decode_type, args.group_size)]
    else:
        decode_modes = DECODE_MODES[args.problem]

    device = torch.device(
        "cuda:0" if "cuda" in args.device and torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = args.checkpoint or f"./checkpoints/{args.problem}/parco.ckpt"
    print(f"Loading checkpoint from {checkpoint_path}")
    model = PARCORLModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=False
    )

    if args.problem == "ffsp":
        env = FFSPEnv(
            generator_params=dict(
                num_stage=args.ffsp_num_stage,
                num_machine=args.ffsp_num_machine,
                num_job=args.ffsp_num_job,
            )
        )
    else:
        env = {"hcvrp": HCVRPEnv, "omdcpdp": OMDCPDPEnv}[args.problem]()
    policy = model.policy.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    for name, td_test in get_datasets(args):
        print(f"\n=== Dataset: {name} ===")
        dataloader = get_dataloader(td_test, batch_size=args.batch_size)

        results = {}
        for mode_name, decode_type, group_size in decode_modes:
            mean_cost, total_time, steps = run_mode(
                policy, env, dataloader, device, decode_type, group_size
            )
            results[mode_name] = (mean_cost, total_time, steps)
            print(
                f"{mode_name:10s} | steps={steps:4d} | time={total_time*1000:9.1f} ms "
                f"| mean_cost={mean_cost:.4f}"
            )

        # "Full AR" only exists as a reference point in the 4-mode sweep;
        # in single-mode override there's nothing to compare against, so
        # gap_pct is trivially 0 for that one mode.
        ar_cost = results.get("Full AR", next(iter(results.values())))[0]
        for mode_name, (mean_cost, total_time, steps) in results.items():
            gap_pct = (mean_cost - ar_cost) / ar_cost * 100
            all_rows.append(
                {
                    "dataset": name,
                    "mode": mode_name,
                    "mean_cost": mean_cost,
                    "time_ms": total_time * 1000,
                    "steps": steps,
                    "gap_pct_vs_full_ar": gap_pct,
                }
            )

    csv_path = os.path.join(args.out_dir, f"idea1_{args.problem}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved raw results to {csv_path}")

    plot_pareto(
        all_rows,
        [name for name, _, _ in decode_modes],
        os.path.join(args.out_dir, f"idea1_{args.problem}_pareto.png"),
    )


def plot_pareto(rows, modes, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        xs = [r["time_ms"] for r in mode_rows]
        ys = [r["gap_pct_vs_full_ar"] for r in mode_rows]
        ax.scatter(xs, ys, label=mode, s=60)

    ax.set_xlabel("Inference time (ms)")
    ax.set_ylabel("Solution gap % vs Full AR")
    ax.set_title("Idea 1: Speed vs Solution Quality Tradeoff (same checkpoint)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved Pareto plot to {out_path}")


if __name__ == "__main__":
    main()
