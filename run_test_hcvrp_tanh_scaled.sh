#!/bin/bash
#SBATCH --job-name=parco_hcvrp_tanh_scaled_test
#SBATCH --output=hcvrp_tanh_scaled_parco_test.log
#SBATCH --error=hcvrp_tanh_scaled_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:6

# Always run from the directory `sbatch` was submitted from -- SLURM copies
# the script into a spool dir before running it, so relative paths (.venv,
# data/, --checkpoint) would otherwise resolve against the wrong location.

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full PAR / greedy decode -- clip settings already baked into the checkpoint.
# TODO: fill in the timestamped run dir printed by run_hcvrp_tanh_scaled.sh
srun python test.py \
  --problem hcvrp \
  --checkpoint logs/train/runs/hcvrp_n100_m7/parco_tanh_scaled_c10/2026-08-13_20-14-23/checkpoints/last.ckpt \
  --decode_type greedy \
  --batch_size 128
