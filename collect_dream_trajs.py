from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json

from vanilla_policy import VanillaPolicy, build_user_prompt
from web_world_model import WebWorldModel
from critic import Critic


@dataclass
class DreamConfig:
    """Config for dreamed (world-model) rollout collection.

    episodes: number of dreamed episodes to generate when using the offline collector
    horizon:  maximum dreamed steps per episode
    save_path: JSONL file to write dreamed episodes (used by DreamCollector.run)
    """
    episodes: int = 1000
    horizon: int = 3
    save_path: str = "./rollouts/dream_T3.jsonl"


class DreamCollector:
    """Collect **dreamed** trajectories by rolling the vanilla policy inside the world model.

    Produces a JSONL with one episode per line:
      {"type":"T3", "instruction": str, "steps": [
          {"observation": obs_t, "action": str, "next_observation": obs_tp1, "delta": {...}, "done": bool}, ...
      ]}
    """

    def __init__(self, policy: VanillaPolicy, world: WebWorldModel, cfg: DreamConfig = DreamConfig()):
        self.policy = policy
        self.world = world
        self.cfg = cfg

    def run(self) -> str:
        """Offline dreamed collection: saves to cfg.save_path and returns the path."""
        with open(self.cfg.save_path, "w", encoding="utf-8") as f:
            for epi in range(self.cfg.episodes):
                instruction = f"Episode {epi}: complete the task."
                obs = {"url": "https://example.com", "title": "", "acc_tree": "", "candidates": []}
                history: List[Dict[str, Any]] = []
                steps: List[Dict[str, Any]] = []
                for _t in range(self.cfg.horizon):
                    action = self.policy.act(instruction, obs, history)
                    obs_next, delta = self.world.predict_next(instruction, obs, history, action)
                    steps.append(
                        {
                            "observation": obs,
                            "action": action,
                            "next_observation": obs_next,
                            "delta": delta.raw_json,
                            "done": bool(obs_next.get("done", False)),
                        }
                    )
                    history.append({"action": action, "observation": obs_next})
                    obs = obs_next
                    if obs.get("done"):
                        break
                episode = {"type": "T3", "instruction": instruction, "steps": steps}
                f.write(json.dumps(episode) + "\n")
                f.flush()
        return self.cfg.save_path


def _extract_action(text: str) -> str:
    import re
    pat = re.compile(r"(do\([^\)]*\)|exit\([^\)]*\)|go_backward\(\)|go_forward\(\))", re.I)
    m = pat.search(text)
    return m.group(1) if m else "do(action=\"Wait\")"


def dream_rollout_for_ppo(
    policy_tokenizer,
    policy_model,
    world: WebWorldModel,
    instruction: str,
    horizon: int,
    max_new_tokens: int = 64,
    critic: Optional[Critic] = None,
) -> List[Dict[str, Any]]:
    """Generate a single dreamed episode using the **current PPO policy model**.

    Returns a list of PPO-ready steps with keys: {prompt, action, reward, next_obs}.
    - `prompt` is the exact user prompt shown to the policy for this step
    - `action` is the one-line web action extracted from the model output
    - `reward` is optional (0.0 if no critic provided)
    - `next_obs` is the dreamed next observation from the world model
    """
    obs = {"url": "https://example.com", "title": "", "acc_tree": "", "candidates": []}
    history: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []

    for _t in range(horizon):
        prompt = build_user_prompt(instruction, obs, history)
        inputs = policy_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a web agent."},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(policy_model.pretrained_model.device)
        out = policy_model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=policy_tokenizer.eos_token_id,
        )
        raw = policy_tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        action = _extract_action(raw)

        obs_next, _delta = world.predict_next(instruction, obs, history, action)
        reward = 0.0
        if critic is not None:
            reward = float(critic.score(instruction, [(obs, action, obs_next)]))

        steps.append({"prompt": prompt, "action": action, "reward": reward, "next_obs": obs_next})
        history.append({"action": action, "observation": obs_next})
        obs = obs_next
        if bool(obs.get("done", False)):
            break

    return steps
