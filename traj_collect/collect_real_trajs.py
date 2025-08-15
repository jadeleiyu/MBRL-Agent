# ==============================================
# File: collect_real_trajs.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Iterable
import json

from mbrl_agent.agents.vanilla_policy import VanillaPolicy
from mbrl_agent.agents.planner_policy import PlannerPolicy
from mbrl_agent.world_model.critic import Critic

# NOTE: We assume a WebArena adapter exists at envs/WebArena with API:
#   env = WebArenaEnv()
#   obs = env.reset(task: Dict[str,Any]) -> Dict[str,Any]  # {url, acc_tree, title, candidates?, done?}
#   obs_next, done, info = env.step(action: str)


def default_task_builder(i: int) -> Dict[str, Any]:
    return {"instruction": f"Episode {i}: complete the task.", "url": "https://example.com"}


@dataclass
class CollectConfig:
    episodes: int = 100
    horizon: int = 10
    save_path: str = "./rollouts/real_T1.jsonl"


class Collector:
    """Environment interaction loop for collecting **real** trajectories.

    Compatible with online PPO: pass a callback to consume transitions as they arrive.
    """

    def __init__(self, env, policy, critic: Optional[Critic] = None, cfg: CollectConfig = CollectConfig()):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.cfg = cfg

    def run(self, task_builder: Callable[[int], Dict[str, Any]] = default_task_builder, on_step: Optional[Callable[[Dict[str, Any]], None]] = None):
        osave = open(self.cfg.save_path, "w", encoding="utf-8")
        for epi in range(self.cfg.episodes):
            task = task_builder(epi)
            instruction = task.get("instruction", "")
            obs = self.env.reset(task)
            history: List[Dict[str, Any]] = []
            steps: List[Dict[str, Any]] = []
            for t in range(self.cfg.horizon):
                action = self.policy.act(instruction, obs, history)
                obs_next, done, info = self.env.step(action)

                step = {"observation": obs, "action": action, "next_observation": obs_next, "done": bool(done)}
                steps.append(step)

                if self.critic is not None and on_step is not None:
                    # Provide an online label if needed by PPO/GRPO
                    frag = [(obs, action, obs_next)]
                    r = self.critic.score(instruction, frag)
                    on_step({"instruction": instruction, "step": step, "reward": r})

                history.append({"action": action, "observation": obs_next})
                obs = obs_next
                if done:
                    break
            episode = {"type": getattr(self.policy, "__class__").__name__, "instruction": instruction, "steps": steps}
            osave.write(json.dumps(episode) + "\n"); osave.flush()
        osave.close()
        return self.cfg.save_path
