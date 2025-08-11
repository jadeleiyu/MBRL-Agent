# ==============================================
# File: mbrl/training/ppo_model_free.py
# ==============================================
from __future__ import annotations
import json
from typing import Iterable, List, Optional, Tuple, Dict, Any

from trl import PPOConfig, PPOTrainer

from ..agents.policy import PolicyAgent, _build_user_prompt
from ..envs.encoder import WebStateEncoder
from ..envs.wm_env import ORMRewardModel


# ---------- I/O ----------

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


# ---------- Builder ----------

def to_ppo_batches(
    episodes: List[dict],
    reward_model: ORMRewardModel,
    encoder: Optional[WebStateEncoder] = None,
    max_steps_per_ep: Optional[int] = None,
) -> Tuple[List[str], List[str], List[float]]:
    """Flatten replay into (queries, responses, rewards) using encoded states.

    Supports step formats:
      - Encoded:  step["state"|"encoded"], step["next_state"|"next_encoded"], step["action"]
      - Raw DOMs: step has {url, dom, next_url, next_dom, action} → encode with `encoder` if provided.
    """
    queries: List[str] = []
    responses: List[str] = []
    rewards: List[float] = []

    for ep in episodes:
        instr = ep.get("instruction", "Use the website to complete the task.")
        steps = ep.get("steps") or []
        if max_steps_per_ep:
            steps = steps[: max_steps_per_ep]
        history: List[str] = []
        prev_encoded: Optional[Dict[str, Any]] = None
        for st in steps:
            if "state" in st or "encoded" in st:
                s_enc = st.get("state") or st.get("encoded")
                ns_enc = st.get("next_state") or st.get("next_encoded") or {}
            else:
                if encoder is None:
                    # Cannot encode; skip
                    continue
                url = st.get("url", "")
                dom = st.get("dom", "")
                nurl = st.get("next_url", url)
                ndom = st.get("next_dom", "")
                s_enc = encoder.encode(dom, instr, url, prev_state=prev_encoded).to_dict()
                ns_enc = encoder.encode(ndom, instr, nurl, prev_state=s_enc).to_dict()

            action = st.get("action") or st.get("agent_action") or "do(action=\"Wait\")"
            query = _build_user_prompt(instr, history, s_enc)
            r = reward_model.step_reward(instr, s_enc, action, ns_enc)

            queries.append(query)
            responses.append(action)
            rewards.append(r)

            history.append(action)
            prev_encoded = ns_enc

    return queries, responses, rewards


# ---------- Main trainer ----------

def run_model_free_ppo(
    policy_model_name: str,
    ppo_config: PPOConfig,
    replay_paths: Iterable[str],
    orm_reward_model: ORMRewardModel,
    max_episodes: Optional[int] = None,
    max_steps_per_ep: Optional[int] = None,
    encode_missing: bool = True,
):
    """Off-policy PPO from pre-collected trajectories (encoded-state aware)."""
    agent = PolicyAgent(policy_model_name)
    ref_agent = PolicyAgent(policy_model_name)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=agent.model,
        ref_model=ref_agent.model,
        tokenizer=agent.tokenizer,
    )

    episodes = load_replay(replay_paths, max_episodes=max_episodes)
    encoder = WebStateEncoder() if encode_missing else None

    queries, responses, rewards = to_ppo_batches(
        episodes, reward_model=orm_reward_model, encoder=encoder, max_steps_per_ep=max_steps_per_ep
    )

    for _ in range(ppo_config.ppo_epochs):
        ppo_trainer.step(queries, responses, rewards)

    return agent, ppo_trainer
