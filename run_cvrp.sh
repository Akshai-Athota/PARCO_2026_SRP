#!/bin/bash
#SBATCH --job-name=parco_cvrp
#SBATCH --output=cvrp_parco.log
#SBATCH --error=cvrp_parco.err
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

# SRP Idea 4: homogeneous-fleet CVRP. Same architecture, decoder, and
# conflict handler as HCVRP -- only the env/generator differ (identical
# capacity across all agents, unit speed). Requires data/cvrp/*.npz to
# exist for validation -- run `python scripts/generate_data.py` first if
# it doesn't (see run script comment / ask for details).
srun python train.py experiment=cvrp \
    model.batch_size=16 \
    model.val_batch_size=16 \
    model.test_batch_size=16 \
    model.num_augment=4 \
    +trainer.accumulate_grad_batches=8
