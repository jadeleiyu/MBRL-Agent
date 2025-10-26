#!/bin/bash
#SBATCH -o ./logs/%x-%j.out
#SBATCH -e ./logs/%x-%j.err
#SBATCH --qos=embodiment_shared
#SBATCH --gres=gpu:8
#SBATCH -c 16
#SBATCH --mem 500G
#SBATCH -t 24:00:00
#SBATCH --array=0-7
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mbrl_agent

python wm_cot_gen.py --job_id $SLURM_ARRAY_TASK_ID 
