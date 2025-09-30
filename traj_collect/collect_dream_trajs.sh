#!/bin/bash
#SBATCH -o ./logs/traj_collect/%x-%j.out
#SBATCH -e ./logs/traj_collect/%x-%j.err
#SBATCH --qos=multimodal-reasoning_high
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem 256G
#SBATCH -t 24:00:00
#SBATCH --array=0-15

set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mbrl_agent

python collect_dream_trajs.py --job_id $SLURM_ARRAY_TASK_ID 


