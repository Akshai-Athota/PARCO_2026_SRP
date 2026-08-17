#!/bin/bash
#SBATCH --job-name=parco_hcvrp_wait_action
#SBATCH --output=hcvrp_wait_action_parco.log
#SBATCH --error=hcvrp_wait_action_parco.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:4

# Always run from the directory `sbatch` was submitted from -- SLURM copies
# the script into a spool dir before running it, so relative paths (.venv,
# data/, logs/) would otherwise resolve against the wrong location.
cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate
export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# SRP Idea 5: explicit learnable "wait" action (use_pos_token=true). Full
# PAR train/val/test decode (unchanged default sampling/greedy, no
# group_size) -- no sequential decoding here.
srun python train.py experiment=hcvrp_wait_action \
    model.batch_size=16 \
    model.val_batch_size=16 \
    model.test_batch_size=16 \
    model.num_augment=4 \
    +trainer.accumulate_grad_batches=8
