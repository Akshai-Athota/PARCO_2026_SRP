#!/bin/bash
#SBATCH --job-name=parco_hcvrp_test
#SBATCH --output=hcvrp_parco_test.log
#SBATCH --error=hcvrp_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python test.py \
  --problem hcvrp \
  --checkpoint logs/train/runs/hcvrp_n100_m7/parco/2026-05-23_04-14-55/checkpoints/last.ckpt \
  --decode_type greedy \
  --batch_size 128
