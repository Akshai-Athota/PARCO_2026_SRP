#!/bin/bash
#SBATCH --job-name=parco_hcvrp_seq_test
#SBATCH --output=hcvrp_seq_parco_test.log
#SBATCH --error=hcvrp_seq_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TODO: fill in the timestamped run dir printed by run_hcvrp_sequential.sh,
# e.g. logs/train/runs/hcvrp_n100_m7/parco/<timestamp>/checkpoints/last.ckpt
srun python test.py \
  --problem hcvrp \
  --checkpoint logs/train/runs/<FILL_IN_TIMESTAMP>/checkpoints/last.ckpt \
  --decode_type group_greedy \
  --group_size 1 \
  --batch_size 128
