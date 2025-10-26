"""
python collect_dream_trajs.py --job_id 0
"""
import sys
import os
import yaml
import json
import argparse
from types import SimpleNamespace
from tqdm import tqdm
import subprocess

from datasets import load_dataset

from mbrl_agent.world_model.web_world_model import WebWorldModel
from mbrl_agent.agents.vanilla_policy import VanillaPolicy
from serve_vllm_models import get_vllm_servers

os.environ["HF_HUB_CACHE"] = "/checkpoint/multimodal-reasoning/jadeleiyu/huggingface"

def get_vllm_servers(model_name_pattern):
    # Run squeue and capture output
    result = subprocess.run(['squeue', '--me', '-o', '"%j, %N, %T, %i"'], capture_output=True, text=True)
    lines = sorted(result.stdout.strip().split('\n'))

    # initialize server dict
    # server_dict = {}
    server_urls = []
    
    # Iterate over each line, skipping the header
    for line in lines[1:]:
        line = line.strip('\"')
        # Get job name, nodelist, and status
        job_name, nodelist, status, job_id = line.split(', ')
        if model_name_pattern in job_name:

            assert "[" not in nodelist, "Multi-node servers not currently supported."

            # keep only running jobs
            if status == "RUNNING" and job_name != "bash":

                try: 
                    if len(job_name.split(":")) < 2:
                        model_name = job_name
                        port = "8000"
                    else:
                        model_name = job_name.split(":")[0]
                        port = job_name.split(":")[1]  # Extract the port number from the job name

                    # model_path = model_paths[model_name]
                    server_address = f"http://{nodelist}:{port}/v1"
                    server_urls.append(server_address)

                except KeyError:
                    continue

    return sorted(server_urls)


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
            "wm_cot": 'None',
            "agent_response": 'None',
        }])

    for t in range(dream_horizon):
        batch_agent_responses, batch_actions, batch_is_valid_act = agent.act(batch_dreamed_trajs, batch_tasks)
        batch_wm_cot, batch_obs_next = world_model.step(batch_actions, batch_dreamed_trajs, batch_tasks)
        for i in range(len(batch_actions)):
            batch_dreamed_trajs[i].append(
                {
                    "step": t+1,
                    "action": batch_actions[i],
                    "is_action_valid": batch_is_valid_act[i], 
                    "next_observation": batch_obs_next[i],
                    "wm_cot": batch_wm_cot[i],
                    "agent_response": batch_agent_responses[i],
                }
            )

    return batch_dreamed_trajs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, 
                        default="configs/collect_dream_trajs.yaml")
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--job_id", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = SimpleNamespace(**config)

    config.vllm_urls = get_vllm_servers(config.wm_model_name.split('/')[-1])

    ds = load_dataset(f"stanfordnlp/nnetnav-{config.env}", split='train')
    world_model = WebWorldModel(config)
    agent = VanillaPolicy(config)

    dreamed_trajs = []
    task_chunk_size = int(len(ds) / args.n_jobs)
    task_start = task_chunk_size * args.job_id
    task_end = min(len(ds), task_start + task_chunk_size)
    n_tasks = task_end - task_start

    print('*'*20)
    print(f"web environment: {config.env}")
    print(f"job_id: {args.job_id}")
    print(f"dreaming from task {task_start} to task {task_end}\n")
    print('*'*20)

    n_batch = int(n_tasks / config.task_batch_size) + 1
    for i in tqdm(range(n_batch)):
    # for i in tqdm(range(4)):
        try:
            start = task_start + config.task_batch_size * i
            end = min(start + config.task_batch_size, len(ds))
            
            batch_tasks = [ds[j] for j in range(start, end)]
            batch_objectives = [
                task['messages'][1]['content'].split('\nOBJECTIVE:')[1].split('\nPREVIOUS ACTIONS')[0].strip() for task in batch_tasks
            ]

            # dream rollout
            batch_dreamed_trajs = dream_rollout(batch_tasks, agent, world_model, dream_horizon=config.dream_horizon)

            for k in range(len(batch_dreamed_trajs)):
                dreamed_trajs.append({
                    'example_id': start+k,
                    'objective': batch_objectives[k],
                    'dreamed_trajectory': batch_dreamed_trajs[k]
                })

        except Exception as e:
            # print(f"error when dreaming task {i}: {e}\n")
            pass
    
    wm_name_short, agent_name_short = config.wm_model_name.split('/')[-1], config.agent_model_name.split('/')[-1]
    os.makedirs(config.output_dir, exist_ok=True)
    save_path = os.path.join(config.output_dir, f"dream_trajs_{config.dream_horizon}-{wm_name_short}-{agent_name_short}-{config.env}-{args.job_id}.json")
    print(f"{len(dreamed_trajs)} out of {n_tasks} tasks have successfully been dreamed")
    with open(save_path, 'w') as f:
        json.dump(dreamed_trajs, f)


if __name__ == "__main__":
    main()

