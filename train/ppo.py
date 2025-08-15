from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from mbrl_agent.world_model.critic import Critic
from mbrl_agent.agents.vanilla_policy import build_user_prompt
from mbrl_agent.world_model.web_world_model import WebWorldModel
from mbrl_agent.traj_collect.collect_dream_trajs import dream_rollout_for_ppo


@dataclass
class PPORunConfig:
    policy_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    learning_rate: float = 1e-6
    batch_size: int = 8
    mini_batch_size: int = 4
    ppo_epochs: int = 2
    target_kl: float = 0.1
    max_new_tokens: int = 64


def _extract_action(text: str) -> str:
    """Return the first valid one-line action from model output."""
    import re
    pat = re.compile(r"(do\([^\)]*\)|exit\([^\)]*\)|go_backward\(\)|go_forward\(\))", re.I)
    m = pat.search(text)
    return m.group(1) if m else "do(action=\"Wait\")"


def rollout_from_env(
    env,
    instruction: str,
    horizon: int,
    policy_tokenizer,
    policy_model,
    critic: Optional[Critic] = None,
    max_new_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """Collect a single real-environment episode with the current policy model."""
    obs = env.reset({"instruction": instruction})
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

        obs_next, done, _info = env.step(action)
        reward = 0.0
        if critic is not None:
            reward = float(critic.score(instruction, [(obs, action, obs_next)]))

        steps.append({"prompt": prompt, "action": action, "next_obs": obs_next, "reward": reward})
        history.append({"action": action, "observation": obs_next})
        obs = obs_next
        if done:
            break

    return steps


def rollout_from_replay(processed_path: str) -> List[Dict[str, Any]]:
    """Load PPO-ready steps from a processed replay JSONL."""
    items: List[Dict[str, Any]] = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            instr = ep.get("instruction", "")
            for st in ep.get("steps", []):
                obs = st.get("observation", {})
                prompt = (
                    f"Instruction: {instr}\n"
                    f"URL: {obs.get('url','')}\n\n"
                    f"# Current Accessibility Tree\n{(obs.get('acc_tree','') or '')[:3000]}\n\n"
                    f"Return next one-line action now."
                )
                items.append(
                    {
                        "prompt": prompt,
                        "action": st.get("action", "do(action=\"Wait\")"),
                        "reward": float(st.get("advantage", 0.0) + st.get("return", 0.0)),
                    }
                )
    return items


def rollout_from_world_model(
    world: WebWorldModel,
    instruction: str,
    horizon: int,
    policy_tokenizer,
    policy_model,
    critic: Optional[Critic] = None,
    max_new_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """Dreamed rollout driven by the **current** policy model."""
    return dream_rollout_for_ppo(
        policy_tokenizer=policy_tokenizer,
        policy_model=policy_model,
        world=world,
        instruction=instruction,
        horizon=horizon,
        max_new_tokens=max_new_tokens,
        critic=critic,
    )


def run_ppo(
    env: Optional[object],
    instructions: List[str],
    cfg: PPORunConfig = PPORunConfig(),
    processed_replay_path: Optional[str] = None,
    critic: Optional[Critic] = None,
    horizon: int = 10,
    world: Optional[WebWorldModel] = None,
) -> str:
    """Run a single PPO update step using data from (in priority):
       1) processed replay, 2) real env, or 3) world-model dreams.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_name, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_model_name, device_map="auto", torch_dtype=torch.bfloat16
    )

    ppo_cfg = PPOConfig(
        model_name=cfg.policy_model_name,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        target_kl=cfg.target_kl,
    )
    trainer = PPOTrainer(config=ppo_cfg, model=model, ref_model=ref_model, tokenizer=tokenizer)

    # Build training batch
    if processed_replay_path:
        batch = rollout_from_replay(processed_replay_path)
    elif env is not None:
        batch = []
        for instr in instructions:
            batch.extend(
                rollout_from_env(
                    env,
                    instr,
                    horizon,
                    tokenizer,
                    model,
                    critic=critic,
                    max_new_tokens=cfg.max_new_tokens,
                )
            )
    elif world is not None:
        batch = []
        for instr in instructions:
            batch.extend(
                rollout_from_world_model(
                    world,
                    instr,
                    horizon,
                    tokenizer,
                    model,
                    critic=critic,
                    max_new_tokens=cfg.max_new_tokens,
                )
            )
    else:
        raise ValueError("Provide either processed_replay_path, env, or world for rollouts.")

    queries = [b["prompt"] for b in batch]
    responses = [b["action"] for b in batch]
    rewards = [float(b.get("reward", 0.0)) for b in batch]

    trainer.step(queries, responses, rewards)

    out_dir = "./checkpoints/ppo-policy"
    trainer.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
