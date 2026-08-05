#!/bin/bash
#SBATCH --job-name=parco_ffsp_seq_test
#SBATCH --output=ffsp_seq_parco_test.log
#SBATCH --error=ffsp_seq_parco_test.err
#SBATCH --mail-user=athota@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:6

source .venv/bin/activate

export WANDB_MODE=offline
export WANDB_DIR=$SLURM_SUBMIT_DIR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# test.py cannot be used for FFSP: it loads test sets from data/<problem>/*.npz,
# but FFSP has no saved test-set files in this repo (train/val/test instances
# are normally generated on the fly). scripts/idea1_decode_modes.py already
# handles this (seeded on-the-fly generation). --decode_type/--group_size
# pin it to a single guaranteed sequential (Full AR) run instead of the
# full Full PAR/PAR-2/PAR-4/Full AR sweep.
# TODO: fill in the timestamped run dir printed by run_ffsp_sequential.sh
srun python scripts/idea1_decode_modes.py \
  --problem ffsp \
  --checkpoint logs/train/runs/<FILL_IN_TIMESTAMP>/checkpoints/last.ckpt \
  --decode_type group_greedy \
  --group_size 1 \
  --ffsp_num_job 20 \
  --ffsp_num_machine 4 \
  --ffsp_num_stage 3
