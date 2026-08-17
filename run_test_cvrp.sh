#!/bin/bash
#SBATCH --job-name=parco_cvrp_test
#SBATCH --output=cvrp_parco_test.log
#SBATCH --error=cvrp_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# Always run from the directory `sbatch` was submitted from -- SLURM copies
# the script into a spool dir before running it, so relative paths (.venv,
# data/, --checkpoint) would otherwise resolve against the wrong location.
cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full PAR / greedy decode. Requires data/cvrp/*.npz -- generate with
# `python scripts/generate_data.py` (run from repo root) if missing.
# TODO: fill in the timestamped run dir printed by run_cvrp.sh
srun python test.py \
  --problem cvrp \
  --checkpoint logs/train/runs/<FILL_IN_TIMESTAMP>/checkpoints/last.ckpt \
  --decode_type greedy \
  --batch_size 128
