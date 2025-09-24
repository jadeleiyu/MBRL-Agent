# ==============================================
# File: planner_policy.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np

from mbrl.agents.vanilla_policy import VanillaPolicy
from mbrl.world_model.web_world_model import WebWorldModel
from mbrl.world_model.critic import Critic
from mbrl.envs.webvoyager_env import enforce_webarena


@dataclass
class PlannerConfig:
    num_candidates: int = 6
    horizon: int = 2
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 64
    gamma: float = 0.99


class PlannerPolicy:
    """One-step planning wrapper around a VanillaPolicy using a short-horizon world model + critic.

    At each step:
      1) Sample K candidate actions from the vanilla policy (with diversity).
      2) For each candidate, roll out d steps in the world model using the vanilla policy to continue.
      3) Score each imagined fragment with the Critic (discounted sum), choose the best first action.
    """

    def __init__(self, base: VanillaPolicy, wm: WebWorldModel, critic: Critic, cfg: PlannerConfig = PlannerConfig()):
        self.base = base
        self.wm = wm
        self.critic = critic
        self.cfg = cfg

    def act(self, instruction: str, observation: Dict[str, Any], history: List[Dict[str, Any]], T: int = 5) -> str:
        # 1) candidate actions
        candidates = []
        for k in range(self.cfg.num_candidates):
            a = self.base.act(
                instruction, observation, history, T=T,
                temperature=self.cfg.temperature + 0.1 * np.random.randn(),
                top_p=self.cfg.top_p, max_new_tokens=self.cfg.max_new_tokens
            )
            a = enforce_webarena(a) or "noop"
            candidates.append(a)

        # 2) score via short imagination
        best_score = -1e18
        best_a = candidates[0]
        for a0 in candidates:
            disc = 1.0
            total = 0.0
            obs_t = observation
            hist_t = list(history)
            # first transition
            next_obs, _delta = self.wm.predict_next(instruction, obs_t, hist_t, a0)
            frag = [(obs_t, a0, next_obs)]
            score = self.critic.score(instruction, frag)
            total += disc * score
            disc *= self.cfg.gamma
            obs_t = next_obs
            hist_t = hist_t + [{"action": a0, "observation": next_obs}]
            broke = False
            for _ in range(self.cfg.horizon - 1):
                if broke:
                    break
                a_t = self.base.act(instruction, obs_t, hist_t, T=T,
                                    temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                                    max_new_tokens=48)
                a_t = enforce_webarena(a_t) or "noop"
                next_obs, _ = self.wm.predict_next(instruction, obs_t, hist_t, a_t)
                frag.append((obs_t, a_t, next_obs))
                total += disc * self.critic.score(instruction, frag)
                disc *= self.cfg.gamma
                obs_t = next_obs
                hist_t = hist_t + [{"action": a_t, "observation": next_obs}]
            if total > best_score:
                best_score = total
                best_a = a0
        return enforce_webarena(best_a) or "noop"
