import json
import os
import argparse
import yaml
from tqdm import tqdm
from types import SimpleNamespace

import sys
from pathlib import Path

# Add project root (one level up from this file) to sys.path dynamically
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent  # projects/mbrl_agent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from world_model.critic import Critic


"""
python label_traj_values.py --config label_traj_values.yaml
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    config_args = parser.parse_args()

    with open(config_args.config, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = SimpleNamespace(**args)

    with open(args.dreamed_traj_path, 'r', encoding="utf-8") as f:
        dreamed_trajs = json.load(f)
    
    critic = Critic(args)
    
    for i in tqdm(range(len(dreamed_trajs))):
        dreamed_traj, objective = dreamed_trajs[i]['dreamed_trajectory'], dreamed_trajs[i]['objective'] 
        critic_values = critic.score(dreamed_traj, objective)
        # print(f"critic values for dreamed trajectory {i}: {critic_values}\n")
        for j in range(len(dreamed_traj)):
            dreamed_trajs[i]['dreamed_trajectory'][j]['critic_value'] = critic_values[j]
    
    wm_name_short = args.wm_model_name.split('/')[-1]
    agent_name_short = args.agent_model_name.split('/')[-1]
    critic_name_short = args.critic_model_name.split('/')[-1]

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"dream_trajs-{wm_name_short}-{agent_name_short}-{critic_name_short}.json",
    )

    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(dreamed_trajs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
