# ==============================================
# File: mbrl/agents/agent.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any

from ..envs.wm_env import WebWorldModel, ORMRewardModel


@dataclass
class StepOutput:
    next_state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class WebAgentLoop:
    """Encoded-state agent loop using the **encoded** WebWorldModel and ORM rewards.

    The policy is expected to be a callable with signature:
        act(instruction: str, action_history: List[str], encoded_state: Dict[str,Any]) -> str
    """

    def __init__(self, world: WebWorldModel, reward: ORMRewardModel, horizon: int = 10):
        self.world = world
        self.reward = reward
        self.horizon = horizon

    def reset(self, instruction: str, url: str = "https://example.com") -> Dict[str, Any]:
        # Minimal bootstrap encoded state
        return {
            "url": url,
            "title": "",
            "instruction": instruction,
            "summary": "Empty page.",
            "candidates": [],
            "diff": {"added": [], "removed": [], "changed": []},
            "done": False,
        }

    def step(self, instruction: str, state: Dict[str, Any], action: str) -> StepOutput:
        next_state = self.world.predict_next(state, instruction, action)
        r = self.reward.step_reward(instruction, state, action, next_state)
        done = bool(next_state.get("done")) or r >= 0.999 or (state.get("step", 0) + 1 >= self.horizon)
        return StepOutput(next_state=next_state, reward=r, done=done, info={})
