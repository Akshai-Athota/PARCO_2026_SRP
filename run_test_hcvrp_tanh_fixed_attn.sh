#!/bin/bash
#SBATCH --job-name=parco_hcvrp_tanh_fixed_attn_test
#SBATCH --output=hcvrp_tanh_fixed_attn_parco_test.log
#SBATCH --error=hcvrp_tanh_fixed_attn_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full PAR / greedy decode -- all clip settings already baked into the
# checkpoint from training.
# TODO: fill in the timestamped run dir printed by run_hcvrp_tanh_fixed_attn.sh
srun python test.py \
  --problem hcvrp \
  --checkpoint logs/train/runs/<FILL_IN_TIMESTAMP>/checkpoints/last.ckpt \
  --decode_type greedy \
  --batch_size 128
