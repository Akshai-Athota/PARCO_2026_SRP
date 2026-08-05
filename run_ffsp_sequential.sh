#!/bin/bash
#SBATCH --job-name=parco_ffsp_seq
#SBATCH --output=ffsp_seq_parco.log
#SBATCH --error=ffsp_seq_parco.err
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

# Uses the current train.py / PARCOMultiStagePolicy pipeline (not the
# legacy parco/tasks/ffsp_old entrypoint used by run_ffsp.sh)
srun python train.py experiment=ffsp_sequential
