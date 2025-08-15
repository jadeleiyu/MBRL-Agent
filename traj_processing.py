# ==============================================
# File: traj_processing.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os

from critic import Critic


@dataclass
class ProcConfig:
    gamma: float = 0.99
    save_path: str = "./rollouts/processed.jsonl"


def label_returns_and_adv(episodes: List[Dict[str, Any]], critic: Critic, cfg: ProcConfig = ProcConfig()) -> List[Dict[str, Any]]:
    """Compute per-step discounted returns and advantages using the critic as a value/reward model.

    For each episode: compute trajectory reward using critic on the full sequence; then per-step G_t; A_t = G_t - V_t
    Here we approximate V_t by critic on prefix fragment.
    """
    processed = []
    for ep in episodes:
        instr = ep.get("instruction", "")
        steps = ep.get("steps", [])
        # Full-trajectory reward (critic acts as reward on complete traj)
        triples = [(st["observation"], st["action"], st["next_observation"]) for st in steps]
        R_full = critic.score(instr, triples)
        # Per-step returns (sparse final reward)
        G = []
        g = R_full
        for _ in reversed(steps):
            G.append(g); g = cfg.gamma * g
        G = list(reversed(G))
        # Advantages by subtracting prefix value estimates
        Adv = []
        for t in range(len(steps)):
            pref = triples[: t + 1]
            Vt = critic.score(instr, pref)
            Adv.append(G[t] - Vt)
        proc_ep = {"instruction": instr, "steps": []}
        for t, st in enumerate(steps):
            st_out = dict(st)
            st_out["return"] = G[t]
            st_out["advantage"] = Adv[t]
            proc_ep["steps"].append(st_out)
        processed.append(proc_ep)
    return processed


def load_replay(paths: List[str]) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            head = f.read(1); f.seek(0)
            if head == "[":
                episodes.extend(json.load(f))
            else:
                for line in f:
                    if line.strip():
                        episodes.append(json.loads(line))
    return episodes


def save_processed(processed: List[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ep in processed:
            f.write(json.dumps(ep) + "\n")
    return path
