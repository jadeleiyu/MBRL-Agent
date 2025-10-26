#!/bin/bash
#SBATCH -o ./logs/traj_score/%x-%j.out
#SBATCH -e ./logs/traj_score/%x-%j.err
#SBATCH --qos=multimodal-reasoning_high
#SBATCH -c 4
#SBATCH --mem 256G
#SBATCH -t 24:00:00
#SBATCH --array=0-15

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mbrl_agent

export HF_HUB_CACHE=/checkpoint/multimodal-reasoning/jadeleiyu/huggingface
python label_traj_values.py --job_id $SLURM_ARRAY_TASK_ID 


