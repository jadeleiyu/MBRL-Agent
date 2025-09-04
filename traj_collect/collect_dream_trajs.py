"""
python collect_dream_trajs.py --config collect_dream_trajs.yaml
"""
import sys
import os
import yaml
import json
import argparse
from types import SimpleNamespace
from tqdm import tqdm

from datasets import load_dataset

sys.path.append('/home/jadeleiyu/projects/mbrl_agent')
from world_model.web_world_model import WebWorldModel
from agents.vanilla_policy import VanillaPolicy

def dream_rollout(batch_tasks, agent, world_model, dream_horizon=5):
    """Offline dreamed web browsing trajectory collection."""

    batch_dreamed_trajs = []

    for task in batch_tasks:
        objective = task['messages'][1]['content'].split('\nOBJECTIVE:')[1].split('\nPREVIOUS ACTIONS')[0].strip()
        task['objective'] = objective
        obs_0 = task['messages'][1]['content'].split('OBSERVATION:\n')[1].split('\nURL:')[0].strip()
        batch_dreamed_trajs.append([{
            'step': 0,
            'action': 'None',
            "is_action_valid": True, 
            "next_observation": obs_0,
            "wm_cot": 'None'
        }])

    for t in range(dream_horizon):
        batch_actions, batch_is_valid_act = agent.act(batch_dreamed_trajs, batch_tasks)
        # if terminated:
        #     break
        batch_obs_next, batch_cot = world_model.step(batch_actions, batch_dreamed_trajs, batch_tasks)
        for i in range(len(batch_actions)):
            batch_dreamed_trajs[i].append(
                {
                    "step": t,
                    "action": batch_actions[i],
                    "is_action_valid": batch_is_valid_act[i], 
                    "next_observation": batch_obs_next[i],
                    "wm_cot": batch_cot[i]
                }
            )

    return batch_dreamed_trajs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    config_args = parser.parse_args()

    with open(config_args.config, "r") as f:
        args = yaml.safe_load(f)
    # wrap in SimpleNamespace so you can do args.base_model instead of args["base_model"]
    args = SimpleNamespace(**args)

    ds = load_dataset(args.dataset, split='train')
    world_model = WebWorldModel(args)
    agent = VanillaPolicy(args)

    dreamed_trajs = []
    if args.n_tasks > 0:
        n_tasks = args.n_tasks
    else:
        n_tasks = len(ds)
    print(f"number of tasks to dream: {n_tasks}\n")
    n_batch = int(n_tasks / args.task_batch_size)
    for i in tqdm(range(n_batch)):
        start, end = args.task_batch_size * i, args.task_batch_size * (i+1)
        batch_tasks = [ds[j] for j in range(start, end)]
        batch_objectives = [
            task['messages'][1]['content'].split('\nOBJECTIVE:')[1].split('\nPREVIOUS ACTIONS')[0].strip() for task in batch_tasks
        ]
        try:
            batch_dreamed_trajs = dream_rollout(batch_tasks, agent, world_model, dream_horizon=args.dream_horizon)
            for k in range(len(batch_dreamed_trajs)):
                dreamed_trajs.append({
                    'example_id': start+k,
                    'objective': batch_objectives[k],
                    'dreamed_trajectory': batch_dreamed_trajs[k]
                })
        except Exception as e:
            # print(f"error when dreaming task {i}: {e}\n")
            pass
    
    wm_name_short, agent_name_short = args.wm_model_name.split('/')[-1], args.agent_model_name.split('/')[-1]
    save_path = os.path.join(args.output_dir, f"dream_trajs-{wm_name_short}-{agent_name_short}.json")
    print(f"{len(dreamed_trajs)} out of {n_tasks} tasks have successfully been dreamed")
    with open(save_path, 'w') as f:
        json.dump(dreamed_trajs, f)


if __name__ == "__main__":
    main()
