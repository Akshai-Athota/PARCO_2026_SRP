#!/bin/bash
#SBATCH --job-name=parco_ffsp_legacy
#SBATCH --output=ffsp_parco.log
#SBATCH --error=ffsp_parco.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00

source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Legacy FFSP entrypoint — uses its own Hydra config in configs/ffsp/
srun python parco/tasks/ffsp_old/FFSP_PARCO/main.py \
    env=ffsp20 \
    train.cuda_device_num=0 \
    train.train_batch_size=16 \
    train.accumulation_steps=4 \
    test.test_batch_size=16
