#!/bin/bash
#SBATCH --job-name=parco_hcvrp_wait_action_test
#SBATCH --output=hcvrp_wait_action_parco_test.log
#SBATCH --error=hcvrp_wait_action_parco_test.err
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

# Full PAR / greedy decode -- use_pos_token is already baked into the
# checkpoint from training (it changes model architecture, not just decode
# behavior, so it can't be toggled at test time independently).
# TODO: fill in the timestamped run dir printed by run_hcvrp_wait_action.sh
srun python test.py \
  --problem hcvrp \
  --checkpoint logs/train/runs/<FILL_IN_TIMESTAMP>/checkpoints/last.ckpt \
  --decode_type greedy \
  --batch_size 128
