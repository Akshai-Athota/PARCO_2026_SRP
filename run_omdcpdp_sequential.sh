#!/bin/bash
#SBATCH --job-name=parco_omdcpdp_seq
#SBATCH --output=omdcpdp_seq_parco.log
#SBATCH --error=omdcpdp_seq_parco.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python train.py experiment=omdcpdp_sequential \
    model.batch_size=8 \
    model.val_batch_size=8 \
    model.test_batch_size=8 \
    model.num_augment=4 \
    +trainer.accumulate_grad_batches=16 \
    +trainer.devices=1 \
    ~trainer.strategy \
    callbacks.model_checkpoint.monitor=val/reward
