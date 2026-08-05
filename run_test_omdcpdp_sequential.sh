#!/bin/bash
#SBATCH --job-name=parco_omdcpdp_seq_test
#SBATCH --output=omdcpdp_seq_parco_test.log
#SBATCH --error=omdcpdp_seq_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TODO: fill in the timestamped run dir printed by run_omdcpdp_sequential.sh
srun python test.py \
  --problem omdcpdp \
  --checkpoint logs/train/runs/<FILL_IN_TIMESTAMP>/checkpoints/last.ckpt \
  --decode_type group_greedy \
  --group_size 1 \
  --batch_size 2
