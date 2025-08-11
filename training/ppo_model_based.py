# ==============================================
# File: mbrl/training/ppo_model_based.py
# ==============================================
from __future__ import annotations
from typing import List

from trl import PPOConfig, PPOTrainer

from ..agents.policy import PolicyAgent, _build_user_prompt
from ..envs.wm_env import WebWorldModel, ORMRewardModel
from ..agents.agent import WebAgentLoop


def run_model_based_ppo(
    policy_ckpt: str,
    world_model: WebWorldModel,
    reward_model: ORMRewardModel,
    instructions: List[str],
    ppo_config: PPOConfig,
    horizon: int = 10,
    max_new_tokens: int = 64,
):
    """Model-based PPO over the **encoded-state** world model.

    Uses the PolicyAgent (WEBRL prompt) to generate actions given (instruction, history, encoded_state).
    """
    agent = PolicyAgent(policy_ckpt)
    ref_agent = PolicyAgent(policy_ckpt)  # reference for KL

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=agent.model,
        ref_model=ref_agent.model,
        tokenizer=agent.tokenizer,
    )

    loop = WebAgentLoop(world=world_model, reward=reward_model, horizon=horizon)

    for _ in range(ppo_config.ppo_epochs):
        queries, responses, rewards = [], [], []

        for instr in instructions:
            state = loop.reset(instruction=instr)
            history: List[str] = []
            for _ in range(horizon):
                query = _build_user_prompt(instr, history, state)
                action = agent.act(instr, history, state, max_new_tokens=max_new_tokens)
                step = loop.step(instr, state, action)

                queries.append(query)
                responses.append(action)
                rewards.append(step.reward)

                history.append(action)
                state = step.next_state
                if step.done:
                    break

        ppo_trainer.step(queries, responses, rewards)

    return agent, ppo_trainer
