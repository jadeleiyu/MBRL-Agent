import json
import os
import argparse
import yaml
from tqdm import tqdm
from types import SimpleNamespace

import sys
sys.path.append('/home/jadeleiyu/projects/mbrl_agent')
from world_model.critic import Critic


"""
python label_traj_values.py --config label_traj_values.yaml
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        args = yaml.safe_load(f)
    args = SimpleNamespace(**args)

    with open(args.dreamed_traj_path, 'r') as f:
        dreamed_trajs = json.load(f)
    
    critic = Critic(args)
    
    for i in tqdm(range(len(dreamed_trajs))):
        dreamed_traj, objective = dreamed_trajs[i]['dreamed_trajectory'], dreamed_trajs[i]['objective'] 
        critic_values = critic.score(dreamed_traj, objective)
        # print(f"critic values for dreamed trajectory {i}: {critic_values}\n")
        for j in range(len(dreamed_traj)):
            dreamed_trajs[i]['dreamed_trajectory'][j]['critic_value'] = critic_values[j]
    
    wm_name_short, agent_name_short = args.wm_model_name.split('/')[-1], args.agent_model_name.split('/')[-1]
    critic_name_short = args.critic_model_name.split('/')[-1]
    save_path = os.path.join(args.output_dir, f"dream_trajs-{wm_name_short}-{agent_name_short}-{critic_name_short}.json")

    with open(save_path, 'w') as f:
        json.dump(dreamed_trajs, f)


if __name__ == "__main__":
    main()
