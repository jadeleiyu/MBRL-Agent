from typing import List

import torch
from trl import PPOConfig, PPOTrainer

from ..agents.policy import PolicyWithValue
from ..agents.agent import WebAgentLoop


def build_observation(instruction: str, state_dom: str, url: str) -> str:
    return (
        "You are a web agent. Read the current page DOM and output ONE atomic action (CLICK/TYPE/NAVIGATE) in a strict schema.\n"
        f"URL: {url}\nDOM:\n{state_dom}\nInstruction: {instruction}\nAction:"
    )


def run_model_based_ppo(
    policy_model_name: str,
    agent_loop: WebAgentLoop,
    instructions: List[str],
    ppo_config: PPOConfig,
    max_new_tokens: int = 64,
):
    # Load model with value head
    policy = PolicyWithValue(policy_model_name)
    ref_policy = PolicyWithValue(policy_model_name)  # reference for KL in PPO

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy.model,
        ref_model=ref_policy.model,
        tokenizer=policy.tokenizer,
    )

    for epoch in range(ppo_config.ppo_epochs):
        queries, responses, rewards = [], [], []

        for instr in instructions:
            # start from an empty synthetic page
            state = agent_loop.reset(url="https://example.com", dom="<body>Home</body>")
            for t in range(agent_loop.horizon):
                obs = build_observation(instr, state.dom, state.url)
                action_text = policy.act([obs], max_new_tokens=max_new_tokens)[0]
                step = agent_loop.step(state, instruction=instr, action=action_text)

                queries.append(obs)
                responses.append(action_text)
                rewards.append(step.reward)
                state = step.next_state
                if step.done:
                    break

        # one PPO update over the collected step-wise data
        ppo_trainer.step(queries, responses, rewards)

    # return the fine-tuned policy in memory; caller can save
    return policy, ppo_trainer