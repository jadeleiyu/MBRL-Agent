"""Off-policy, model-free PPO using pre-collected trajectories.

We load trajectories collected in a WebArena sandbox, label per-step rewards with the
ORM reward model, convert each (state -> action) pair into a TRL-compatible (query, response)
example, and run PPO updates without generating new rollouts during training.

Replay file schema (JSONL or JSON list): one episode per line/object:
{
  "instruction": str,
  "steps": [
    {"url": str, "dom": str, "action": str, "next_url": str, "next_dom": str, "done": bool}
  ]
}
"""
from __future__ import annotations

import json
from typing import Iterable, List, Optional, Tuple
from trl import PPOConfig, PPOTrainer

from ..agents.policy import PolicyWithValue


# ---------- Utilities ----------

def _read_json_like(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        try:
            obj = json.load(f)
            if isinstance(obj, dict) and "data" in obj:
                return obj["data"]
        except Exception:
            pass
        f.seek(0)
        return [json.loads(line) for line in f if line.strip()]


def load_replay(paths: Iterable[str], max_episodes: Optional[int] = None) -> List[dict]:
    episodes: List[dict] = []
    for p in paths:
        items = _read_json_like(p)
        if isinstance(items, dict):
            items = [items]
        episodes.extend(items)
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes[:max_episodes] if max_episodes else episodes


def build_query(instruction: str, url: str, dom: str) -> str:
    return (
        "You are a web agent. Read the current page DOM and output ONE atomic action (CLICK/TYPE/SELECT/NAVIGATE) in a strict schema."
        f"URL: {url}\n DOM: {dom}\n Instruction: {instruction}\n Action:"
    )


def to_ppo_batches(
    episodes: List[dict],
    tokenizer,
    max_steps_per_ep: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Tuple[str, str, str, str]]]:
    """Return (queries, responses, meta) flattened per step.

    meta[i] = (instruction, dom, action, next_dom) for reward labeling and audit.
    """
    queries: List[str] = []
    responses: List[str] = []
    meta: List[Tuple[str, str, str, str]] = []

    for ep in episodes:
        instr = ep.get("instruction", "Use the website to complete the task.")
        steps = ep.get("steps") or []
        if max_steps_per_ep:
            steps = steps[: max_steps_per_ep]
        for st in steps:
            url = st.get("url") or st.get("curr_url") or ""
            dom = st.get("dom") or st.get("curr_dom") or st.get("html") or ""
            action = st.get("action") or st.get("agent_action") or "CLICK(NODE())"
            next_dom = st.get("next_dom") or st.get("next_html") or ""
            q = build_query(instr, url, dom)
            queries.append(q)
            responses.append(action)
            meta.append((instr, dom, action, next_dom))

    return queries, responses, meta


# ---------- Main trainer ----------

def run_model_free_ppo(
    policy_model_name: str,
    ppo_config: PPOConfig,
    replay_paths: Iterable[str],
    orm_reward_model,
    max_episodes: Optional[int] = None,
    max_steps_per_ep: Optional[int] = None,
):
    """Off-policy PPO from pre-collected trajectories.

    Args:
      policy_model_name: path/HF id of the policy (SFT checkpoint recommended)
      ppo_config: TRL PPOConfig
      replay_paths: list/iterable of JSON/JSONL files with trajectories
      orm_reward_model: instance of ORMRewardModel (computes step rewards)
      max_episodes: optional cap on episodes loaded
      max_steps_per_ep: optional cap on steps per episode
    """
    # Load policy + value head and reference model for KL
    policy = PolicyWithValue(policy_model_name)
    ref_policy = PolicyWithValue(policy_model_name)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy.model,
        ref_model=ref_policy.model,
        tokenizer=policy.tokenizer,
    )

    # Build training batches from replay buffer
    episodes = load_replay(replay_paths, max_episodes=max_episodes)
    queries, responses, meta = to_ppo_batches(episodes, policy.tokenizer, max_steps_per_ep=max_steps_per_ep)

    # Label rewards with ORM (step-wise)
    rewards: List[float] = []
    for (instr, dom, action, next_dom) in meta:
        r = orm_reward_model.step_reward(instr, dom, action, next_dom)
        rewards.append(r)

    # TRL PPO expects per-sample scalar reward; we provide step-wise rewards directly.
    # Run a number of PPO epochs over the loaded buffer.
    for _ in range(ppo_config.ppo_epochs):
        ppo_trainer.step(queries, responses, rewards)

    return policy, ppo_trainer
