"""SRP Idea 1: Sequential (AR) vs Parallel (PAR) Decoding.

Takes a single trained PARCO checkpoint (zero retraining) and evaluates it
with four decoding modes that only change the *order* in which the M agents'
actions are committed within each environment step:

    Full PAR  - all M agents propose simultaneously (existing "greedy" mode)
    PAR-2     - agents decoded in groups of 2
    PAR-4     - agents decoded in groups of 4
    Full AR   - agents decoded one at a time, strictly sequential, no conflicts

For each mode we record wall-clock inference time and solution cost, then
report the solution-quality gap (%) relative to Full AR (the zero-conflict
reference) versus the speedup, and plot the resulting Pareto curve.

Usage:
    python scripts/idea1_decode_modes.py --problem hcvrp \
        --checkpoint checkpoints/hcvrp/parco.ckpt \
        --datasets data/hcvrp/n100_m7_seed24610.npz
"""

import argparse
import csv
import os
import time

import torch

from rl4co.data.utils import load_npz_to_tensordict

from parco.envs import FFSPEnv, HCVRPEnv, OMDCPDPEnv
from parco.models import PARCORLModule
from parco.tasks.eval import get_dataloader

DECODE_MODES = [
    ("Full PAR", "greedy", None),
    ("PAR-2", "group_greedy", 2),
    ("PAR-4", "group_greedy", 4),
    ("Full AR", "group_greedy", 1),
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="hcvrp")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="results/idea1")
    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if "cuda" in args.device and torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = args.checkpoint or f"./checkpoints/{args.problem}/parco.ckpt"
    print(f"Loading checkpoint from {checkpoint_path}")
    model = PARCORLModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=False
    )

    env = {"hcvrp": HCVRPEnv, "omdcpdp": OMDCPDPEnv, "ffsp": FFSPEnv}[args.problem]()
    policy = model.policy.to(device).eval()

    datasets = args.datasets or [
        f"./data/{args.problem}/{f}"
        for f in os.listdir(f"./data/{args.problem}")
        if "sol" not in f
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    for dataset in sorted(datasets):
        print(f"\n=== Dataset: {dataset} ===")
        td_test = load_npz_to_tensordict(dataset)
        dataloader = get_dataloader(td_test, batch_size=args.batch_size)

        results = {}
        for name, decode_type, group_size in DECODE_MODES:
            mean_cost, total_time, steps = run_mode(
                policy, env, dataloader, device, decode_type, group_size
            )
            results[name] = (mean_cost, total_time, steps)
            print(
                f"{name:10s} | steps={steps:4d} | time={total_time*1000:9.1f} ms "
                f"| mean_cost={mean_cost:.4f}"
            )

        ar_cost = results["Full AR"][0]
        for name, (mean_cost, total_time, steps) in results.items():
            gap_pct = (mean_cost - ar_cost) / ar_cost * 100
            all_rows.append(
                {
                    "dataset": os.path.basename(dataset),
                    "mode": name,
                    "mean_cost": mean_cost,
                    "time_ms": total_time * 1000,
                    "steps": steps,
                    "gap_pct_vs_full_ar": gap_pct,
                }
            )

    csv_path = os.path.join(args.out_dir, "idea1_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved raw results to {csv_path}")

    plot_pareto(all_rows, os.path.join(args.out_dir, "idea1_pareto.png"))


def plot_pareto(rows, out_path):
    import matplotlib.pyplot as plt

    modes = [name for name, _, _ in DECODE_MODES]
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
