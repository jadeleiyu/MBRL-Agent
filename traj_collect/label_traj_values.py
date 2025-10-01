import json
import os
import argparse
import yaml
from tqdm import tqdm
from types import SimpleNamespace

from mbrl_agent.world_model.value import WebValueModel
from serve_vllm_models import get_vllm_servers

"""
export HF_HUB_CACHE=/checkpoint/multimodal-reasoning/jadeleiyu/huggingface;
python label_traj_values.py --job_id 0
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/label_traj_values.yaml")
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--job_id", type=int, default=0)
    config_args = parser.parse_args()

    with open(config_args.config, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = SimpleNamespace(**args)

    args.vllm_urls = get_vllm_servers(args.value_model_name.split('/')[-1])

    wm_name_short = args.wm_model_name.split('/')[-1]
    agent_name_short = args.agent_model_name.split('/')[-1]
    value_name_short = args.value_model_name.split('/')[-1]

    dreamed_traj_path = os.path.join(args.output_dir, f"dream_trajs_{args.dream_horizon}-{wm_name_short}-{agent_name_short}-{args.env}-{config_args.job_id}.json")
    with open(dreamed_traj_path, 'r', encoding="utf-8") as f:
        dreamed_trajs = json.load(f)
    
    value_model = WebValueModel(args)
    
    for i in tqdm(range(len(dreamed_trajs))):
    # for i in tqdm(range(4)):
        dreamed_traj, objective = dreamed_trajs[i]['dreamed_trajectory'], dreamed_trajs[i]['objective'] 
        values = value_model.score(dreamed_traj, objective)
        # print(f"critic values for dreamed trajectory {i}: {critic_values}\n")
        for j in range(len(dreamed_traj)):
            dreamed_trajs[i]['dreamed_trajectory'][j]['critic_value'] = values[j]

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"dream_trajs_{args.dream_horizon}-{wm_name_short}-{agent_name_short}-{args.env}-{config_args.job_id}-{value_name_short}-scored.json",
    )

    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(dreamed_trajs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
