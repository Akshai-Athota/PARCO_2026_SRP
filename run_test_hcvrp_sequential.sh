#!/bin/bash
#SBATCH --job-name=parco_hcvrp_seq_test
#SBATCH --output=hcvrp_seq_parco_test.log
#SBATCH --error=hcvrp_seq_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:6

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --group_size 1 makes this explicitly Full AR (sequential) regardless of
# what's baked into the checkpoint, matching the training config above.
# TODO: fill in the timestamped run dir printed by run_hcvrp_sequential.sh
srun python test.py \
  --problem hcvrp \
  --checkpoint logs/train/runs/hcvrp_n100_m7/parco_sequential/2026-08-13_09-11-30/checkpoints/last.ckpt \
  --decode_type group_greedy \
  --group_size 1 \
  --batch_size 128
