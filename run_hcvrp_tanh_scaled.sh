#!/bin/bash
#SBATCH --job-name=parco_hcvrp_tanh_scaled
#SBATCH --output=hcvrp_tanh_scaled_parco.log
#SBATCH --error=hcvrp_tanh_scaled_parco.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:4

source .venv/bin/activate
export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# SRP Idea 2: phi_scaled(z;C)=C*tanh(z/C) decoder-logit clipping, C=10.
# Behaviorally identical to plain "experiment=hcvrp" (this is already the
# default) -- exists only for an explicitly labeled run/checkpoint/wandb
# name to compare against hcvrp_tanh_fixed / hcvrp_tanh_attn /
# hcvrp_tanh_fixed_attn. Full PAR train/val/test decode, no group_size.
srun python train.py experiment=hcvrp_tanh_scaled \
    model.batch_size=16 \
    model.val_batch_size=16 \
    model.test_batch_size=16 \
    model.num_augment=4 \
    +trainer.accumulate_grad_batches=8
