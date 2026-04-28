#!/bin/bash
#SBATCH --job-name=parco_train
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# Load modules if required by your cluster (e.g., module load python/3.10)
# module load python

# Activate the virtual environment. 
# If you created it via Anaconda/Miniconda:
# conda activate Parco_2026
# If you created it via standard python venv:
conda activate parco2026

# Run the PARCO training script.
# You can change 'experiment=hcvrp' to 'experiment=ffsp' or 'experiment=omadap'
srun python train.py experiment=hcvrp
