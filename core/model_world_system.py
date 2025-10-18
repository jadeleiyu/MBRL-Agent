"""
Model World + GRPO Integrated System
Integrates model-driven virtual worlds with GRPO training
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from .llm_judge_reward import LLMJudgeRewardFunction, create_llm_judge_reward
from .rollout_collector import RolloutCollector, create_rollout_collector
from .grpo_trainer import GRPOTrainer, GRPOConfig, create_grpo_trainer
from ..environments.model_world import ModelWorldEnvironment
from ..agents.world_agent import WorldInteractionAgent


@dataclass
class ModelWorldSystemConfig:
    """Model World + GRPO system configuration"""
    world_model_path: str
    agent_model_path: str
    judge_model_path: str
    max_turns: int = 3
    storage_path: Optional[str] = None
    grpo_config: Optional[GRPOConfig] = None


class ModelWorldSystem:
    """Model-driven virtual world integrated system"""

    def __init__(self, config: ModelWorldSystemConfig):
        """
        Initialize model world system

        Args:
            config: System configuration
        """
        self.config = config

        # Initialize components
        self.world_environment = ModelWorldEnvironment(config.world_model_path)
        print(f"✓ Created Model World Environment with model: {config.world_model_path}")

        self.agent = WorldInteractionAgent(config.agent_model_path)
        print(f"✓ Created World Interaction Agent with model: {config.agent_model_path}")

        self.judge_reward = create_llm_judge_reward(
            model_path=config.judge_model_path
        )
        print(f"✓ Created LLM Judge with model: {config.judge_model_path}")

        self.rollout_collector = create_rollout_collector(
            storage_path=config.storage_path
        )
        print("✓ Created Rollout Collector")

        self.grpo_trainer = create_grpo_trainer(
            config=config.grpo_config
        )
        print("✓ Created GRPO Trainer")

        # Training state
        self.training_rounds = 0
        self.best_reward = -float('inf')

    def run_rollout(
        self,
        task_id: str,
        task_description: str,
        initial_task: str
    ) -> Dict[str, Any]:
        """
        Run one rollout in the model-driven virtual world

        Args:
            task_id: Task ID
            task_description: Task description
            initial_task: Initial task for the agent

        Returns:
            Rollout result
        """
        # Start trajectory collection
        self.rollout_collector.start_trajectory(
            task_id=task_id,
            task_description=task_description,
            agent_model=self.config.agent_model_path,
            env_model=self.config.world_model_path
        )

        # Initialize world state
        world_state = self.world_environment.get_world_state()
        total_reward = 0.0

        # Multi-turn interaction
        for turn in range(self.config.max_turns):
            # Agent generates action based on world state
            action = self.agent.act(world_state, initial_task)
            print(f"  Turn {turn}: Agent action: {action[:80]}...")

            # Build reward input
            task_info = {
                "instruction": task_description,
                "world_state": world_state,
                "agent_action": action
            }

            # Calculate reward using LLM judge
            from rllm.rewards.reward_types import RewardInput
            reward_input = RewardInput(
                task_info=task_info,
                action=action
            )
            reward_output = self.judge_reward(reward_input)

            reward = reward_output.reward
            total_reward += reward
            print(f"    Reward: {reward:.3f}")

            # Determine if episode ended
            done = (
                turn >= self.config.max_turns - 1 or
                reward < 0.3  # Early termination if reward too low
            )

            # Collect data
            self.rollout_collector.add_step(
                turn=turn,
                observation=world_state,
                action=action,
                reward=reward,
                done=done,
                info={
                    "world_summary": self.world_environment.get_state_summary(),
                    "judge_metadata": reward_output.metadata
                }
            )

            # Update world state based on agent action
            if not done:
                world_state = self.world_environment.get_world_state(action)
            else:
                world_state = {}

            if done:
                break

        # End trajectory collection
        trajectory = self.rollout_collector.end_trajectory()

        return {
            "trajectory": trajectory,
            "total_reward": total_reward,
            "num_steps": len(trajectory.steps),
            "world_summary": self.world_environment.get_state_summary()
        }

    def train_with_grpo(
        self,
        num_rollouts: int = 10,
        tasks: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Train using GRPO in model-driven virtual worlds

        Args:
            num_rollouts: Number of rollouts
            tasks: Task list

        Returns:
            Training result
        """
        if tasks is None:
            tasks = self._get_default_tasks()

        # Run multiple rollouts
        rollout_results = []
        for i in range(num_rollouts):
            task = tasks[i % len(tasks)]
            result = self.run_rollout(
                task_id=f"world_task_{i}",
                task_description=task["description"],
                initial_task=task["task"]
            )
            rollout_results.append(result)

        # Get all trajectories
        trajectories = self.rollout_collector.get_trajectories()

        if len(trajectories) == 0:
            return {
                "success": False,
                "message": "No trajectory data collected",
                "num_trajectories": 0
            }

        # Prepare training data
        training_data = self.grpo_trainer.prepare_training_data(trajectories)

        # Simulate training step
        current_logprobs = [0.0] * len(training_data["actions"])  # Simulate current policy logprobs

        # Execute training step
        training_stats = self.grpo_trainer.train_step(training_data, current_logprobs)

        # Update training state
        self.training_rounds += 1
        avg_reward = np.mean([r["total_reward"] for r in rollout_results])
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward

        return {
            "success": True,
            "training_round": self.training_rounds,
            "num_trajectories": len(trajectories),
            "average_reward": avg_reward,
            "best_reward": self.best_reward,
            "training_stats": training_stats,
            "rollout_results": rollout_results
        }

    def _get_default_tasks(self) -> List[Dict[str, str]]:
        """Get default task list for virtual world interactions"""
        return [
            {
                "description": "Explore and understand the virtual world",
                "task": "Explore this virtual world"
            },
            {
                "description": "Interact with elements in the virtual world",
                "task": "Interact with the world"
            },
            {
                "description": "Learn about the world's rules and dynamics",
                "task": "Learn world dynamics"
            },
            {
                "description": "Navigate through different locations in the world",
                "task": "Navigate the world"
            }
        ]

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        trajectories = self.rollout_collector.get_trajectories()
        training_history = self.grpo_trainer.get_training_history()

        return {
            "training_rounds": self.training_rounds,
            "best_reward": self.best_reward,
            "num_trajectories": len(trajectories),
            "num_training_steps": len(training_history),
            "world_model": self.config.world_model_path,
            "agent_model": self.config.agent_model_path,
            "judge_model": self.config.judge_model_path
        }

    def reset_system(self):
        """Reset system state"""
        self.rollout_collector.clear_trajectories()
        self.grpo_trainer.reset_training_history()
        self.world_environment.reset()
        self.training_rounds = 0
        self.best_reward = -float('inf')


# Convenience function
def create_model_world_system(
    world_model_path: str,
    agent_model_path: str,
    judge_model_path: str,
    max_turns: int = 3,
    storage_path: Optional[str] = None,
    grpo_config: Optional[GRPOConfig] = None
) -> ModelWorldSystem:
    """
    Convenience function to create model world system

    Args:
        world_model_path: World model path
        agent_model_path: Agent model path
        judge_model_path: Judge model path
        max_turns: Maximum turns
        storage_path: Storage path
        grpo_config: GRPO configuration

    Returns:
        ModelWorldSystem instance
    """
    config = ModelWorldSystemConfig(
        world_model_path=world_model_path,
        agent_model_path=agent_model_path,
        judge_model_path=judge_model_path,
        max_turns=max_turns,
        storage_path=storage_path,
        grpo_config=grpo_config
    )

    return ModelWorldSystem(config)