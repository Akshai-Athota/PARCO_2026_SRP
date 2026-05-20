#!/bin/bash
#SBATCH --job-name=parco_hcvrp
#SBATCH --output=hcrvp_parco.log
#SBATCH --error=hcrvp_parco.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

source .venv/bin/activate

# Force wandb offline — cluster has no outbound internet
export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb

srun python train.py experiment=hcvrp \
    model.batch_size=16 \
    model.val_batch_size=16 \
    model.test_batch_size=16 \
    +trainer.accumulate_grad_batches=8 \
    +trainer.devices=1 \
    ~trainer.strategy
