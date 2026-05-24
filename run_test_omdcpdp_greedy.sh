#!/bin/bash
#SBATCH --job-name=parco_omdcpdp
#SBATCH --output=omdcpdp_parco_test.log
#SBATCH --error=omdcpdp_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2
#SBATCH --mem=32G

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python test.py \
  --problem omdcpdp \
  --checkpoint logs/train/runs/omdcpdp_n100_m50/parco/2026-05-18_14-40-06/checkpoints/epoch_095.ckpt \
  --decode_type greedy \
  --batch_size 2       
