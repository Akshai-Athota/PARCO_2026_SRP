#!/bin/bash
#SBATCH --job-name=parco_hcvrp_test_seq
#SBATCH --output=hcvrp_parco_test_seq.log
#SBATCH --error=hcvrp_parco_test_seq.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:5

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


srun python test.py --problem hcvrp \
  --checkpoint logs/train/runs/hcvrp_n100_m7/parco_sequential/2026-08-06_06-34-46/checkpoints/last.ckpt \
  --decode_type group_greedy 
  --group_size 1 
  --batch_size 128
