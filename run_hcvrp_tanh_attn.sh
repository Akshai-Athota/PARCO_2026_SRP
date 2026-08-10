#!/bin/bash
#SBATCH --job-name=parco_hcvrp_tanh_attn
#SBATCH --output=hcvrp_tanh_attn_parco.log
#SBATCH --error=hcvrp_tanh_attn_parco.err
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

# SRP Idea 2: adds encoder attention-logit clipping (phi_scaled, C=10, set
# in configs/experiment/hcvrp_tanh_attn.yaml) on top of the default decoder
# clipping. Full PAR train/val/test decode (unchanged default, no
# group_size) -- no sequential decoding here.
srun python train.py experiment=hcvrp_tanh_attn \
    model.batch_size=16 \
    model.val_batch_size=16 \
    model.test_batch_size=16 \
    model.num_augment=4 \
    +trainer.accumulate_grad_batches=8
