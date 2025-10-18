"""
DevWeb - Virtual Web Environment for LLM Training

A reinforcement learning framework where:
- One LLM acts as a "virtual web environment" simulating web pages and interactions
- Another LLM acts as an agent interacting with the virtual web
- LLM-as-Judge provides rewards for successful interactions
- GRPO training optimizes agent behavior in the simulated web environment

Enables safe, scalable web interaction training without real internet access.
"""

from .core.llm_judge_reward import (
    LLMJudgeRewardFunction,
    create_llm_judge_reward
)

from .core.rollout_collector import (
    RolloutCollector,
    RolloutStep,
    RolloutTrajectory,
    create_rollout_collector
)

from .core.grpo_trainer import (
    GRPOTrainer,
    GRPOConfig,
    create_grpo_trainer
)

from .core.model_world_system import (
    ModelWorldSystem,
    ModelWorldSystemConfig,
    create_model_world_system
)

from .environments.model_world import ModelWorldEnvironment
from .agents.world_agent import WorldInteractionAgent

__all__ = [
    # LLM judge reward
    "LLMJudgeRewardFunction",
    "create_llm_judge_reward",

    # Rollout data collection
    "RolloutCollector",
    "RolloutStep",
    "RolloutTrajectory",
    "create_rollout_collector",

    # GRPO training
    "GRPOTrainer",
    "GRPOConfig",
    "create_grpo_trainer",

    # Model world system
    "ModelWorldSystem",
    "ModelWorldSystemConfig",
    "create_model_world_system",

    # Environments
    "ModelWorldEnvironment",

    # Agents
    "WorldInteractionAgent"
]

__version__ = "0.1.0"