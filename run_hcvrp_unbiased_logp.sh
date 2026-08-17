#!/bin/bash
#SBATCH --job-name=parco_hcvrp_unbiased_logp
#SBATCH --output=hcvrp_unbiased_logp_parco.log
#SBATCH --error=hcvrp_unbiased_logp_parco.err
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

# SRP Idea 3: use_init_logp=false -- REINFORCE credits the log-prob of the
# actually-executed action (post-conflict-resolution) instead of the
# proposed one. Full PAR train/val/test decode (unchanged default
# sampling/greedy) -- no sequential decoding here.
srun python train.py experiment=hcvrp_unbiased_logp \
    model.batch_size=16 \
    model.val_batch_size=16 \
    model.test_batch_size=16 \
    model.num_augment=4 \
    +trainer.accumulate_grad_batches=8
