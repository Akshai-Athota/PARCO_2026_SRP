#!/bin/bash
#SBATCH --job-name=parco_hcvrp
#SBATCH --output=hcrvp_parco.log
#SBATCH --error=hcrvp_parco.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:2

# Load modules if required by your cluster (e.g., module load python/3.10)
# module load python

# Activate the virtual environment. 
# If you created it via Anaconda/Miniconda:
# conda activate Parco_2026
# If you created it via standard python venv:
source  .venv/bin/activate
export WANDB_API_KEY="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"
# Run the PARCO training script.
# You can change 'experiment=hcvrp' to 'experiment=ffsp' or 'experiment=omadap'
srun python train.py experiment=hcvrp
srun python test.py --problem hcvrp --decode_type sampling --batch_size 1 --sample_size 1280
