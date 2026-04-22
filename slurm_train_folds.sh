#!/bin/bash --login
#SBATCH --job-name=dbm-flow
#SBATCH --output=logs/fold%a_%j.out
#SBATCH --error=logs/fold%a_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-4          # one job per fold; submit subset with --array=0,2 etc.

mkdir -p logs

module purge
module load Miniforge3
conda activate embed

echo "========================================"
echo "Node:      $(hostname)"
echo "Fold:      ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "========================================"

SCRIPT=/mnt/scratch/rubabfiz/repos/DBM/my_train_flow_only_long_eval_opt_protocol.py

python "$SCRIPT" --fold "${SLURM_ARRAY_TASK_ID}"

echo "Fold ${SLURM_ARRAY_TASK_ID} done."
