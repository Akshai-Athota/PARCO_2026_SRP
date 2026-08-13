#!/bin/bash
#SBATCH --job-name=parco_hcvrp_seq
#SBATCH --output=hcvrp_seq_parco.log
#SBATCH --error=hcvrp_seq_parco.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:6

# Always run from this script's own location, regardless of where `sbatch`
# was invoked from -- train.py doesn't need this (pyrootutils anchors to
# the repo root independent of CWD), but test.py and the .venv activation
# below both rely on relative paths, so pin CWD explicitly.
cd "$(dirname "${BASH_SOURCE[0]}")"

source .venv/bin/activate
export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# SRP Idea 1: full sequential (Full AR) training, HCVRP only.
srun python train.py experiment=hcvrp_sequential \
    model.batch_size=16 \
    model.val_batch_size=16 \
    model.test_batch_size=16 \
    model.num_augment=4 \
    +trainer.accumulate_grad_batches=8
