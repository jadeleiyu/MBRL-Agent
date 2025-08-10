from dataclasses import dataclass
from typing import Dict

from ..envs.wm_env import WMState, WMAWorldModelEnv, ORMRewardModel


@dataclass
class StepOutput:
    next_state: WMState
    reward: float
    done: bool
    info: Dict


class WebAgentLoop:
    def __init__(self, env: WMAWorldModelEnv, reward: ORMRewardModel, horizon: int = 10):
        self.env = env
        self.reward = reward
        self.horizon = horizon

    def reset(self, url: str, dom: str) -> WMState:
        return WMState(url=url, dom=dom, done=False, step=0)

    def step(self, state: WMState, instruction: str, action: str) -> StepOutput:
        pred = self.env.predict_next(state, instruction, action)
        next_state = WMState(url=pred.get("next_url", state.url), dom=pred.get("next_dom", state.dom), done=bool(pred.get("done", False)), step=state.step + 1)
        r = self.reward.step_reward(instruction, state.dom, action, next_state.dom)
        done = next_state.done or (state.step + 1 >= self.horizon) or (r >= 0.999)
        return StepOutput(next_state=next_state, reward=r, done=done, info={"reason": pred.get("reason", "")})
    
    