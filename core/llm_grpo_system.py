"""
LLM + GRPO Integrated System
Integrates LLM-as-Judge reward system with GRPO training
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from .llm_judge_reward import LLMJudgeRewardFunction, create_llm_judge_reward
from .rollout_collector import RolloutCollector, create_rollout_collector
from .grpo_trainer import GRPOTrainer, GRPOConfig, create_grpo_trainer


@dataclass
class LLMGRPOSystemConfig:
    """LLM + GRPO system configuration"""
    agent_model_path: str
    judge_model_path: str
    max_turns: int = 3
    storage_path: Optional[str] = None
    grpo_config: Optional[GRPOConfig] = None


class LLMGRPOSystem:
    """LLM + GRPO integrated system"""

    def __init__(self, config: LLMGRPOSystemConfig):
        """
        Initialize integrated system

        Args:
            config: System configuration
        """
        self.config = config

        # Initialize components with shared LLM instance
        from vllm import LLM

        # Create shared LLM instance to avoid reloading model
        self.shared_llm = LLM(model=config.judge_model_path, tensor_parallel_size=1)

        self.judge_reward = create_llm_judge_reward(
            model_path=config.judge_model_path,
            llm_instance=self.shared_llm
        )

        self.rollout_collector = create_rollout_collector(
            storage_path=config.storage_path
        )

        self.grpo_trainer = create_grpo_trainer(
            config=config.grpo_config
        )

        # Training state
        self.training_rounds = 0
        self.best_reward = -float('inf')

    def run_rollout(
        self,
        task_id: str,
        task_description: str,
        initial_question: str,
        agent
    ) -> Dict[str, Any]:
        """
        Run one rollout

        Args:
            task_id: Task ID
            task_description: Task description
            initial_question: Initial question
            agent: Agent instance

        Returns:
            Rollout result
        """
        # Start trajectory collection
        self.rollout_collector.start_trajectory(
            task_id=task_id,
            task_description=task_description,
            agent_model=self.config.agent_model_path,
            env_model=self.config.judge_model_path
        )

        # Initialize environment state
        observation = {"question": initial_question}
        total_reward = 0.0

        # Multi-turn interaction
        for turn in range(self.config.max_turns):
            # Agent generates action
            action = agent.act(observation)

            # Build reward input
            task_info = {
                "instruction": task_description,
                "question": observation.get("question", "")
            }

            # Calculate reward
            from rllm.rewards.reward_types import RewardInput
            reward_input = RewardInput(
                task_info=task_info,
                action=action
            )
            reward_output = self.judge_reward(reward_input)

            reward = reward_output.reward
            total_reward += reward

            # Determine if episode ended
            done = (
                turn >= self.config.max_turns - 1 or
                reward < 0.3  # Early termination if reward too low
            )

            # Collect data
            self.rollout_collector.add_step(
                turn=turn,
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                info=reward_output.metadata
            )

            # Generate next observation (simplified version)
            if not done:
                observation = {
                    "question": f"Continue answering: {observation.get('question', '')}"
                }
            else:
                observation = {}

            if done:
                break

        # End trajectory collection
        trajectory = self.rollout_collector.end_trajectory()

        return {
            "trajectory": trajectory,
            "total_reward": total_reward,
            "num_steps": len(trajectory.steps)
        }

    def train_with_grpo(
        self,
        num_rollouts: int = 10,
        agent=None,
        tasks: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Train using GRPO

        Args:
            num_rollouts: Number of rollouts
            agent: Agent instance
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
                task_id=f"task_{i}",
                task_description=task["description"],
                initial_question=task["question"],
                agent=agent
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

        # Simulate training step (in real implementation, this would call actual policy network)
        # Here we use simulated log probabilities
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

    def evaluate_agent(
        self,
        agent,
        eval_tasks: Optional[List[Dict[str, str]]] = None,
        num_eval_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate agent performance

        Args:
            agent: Agent instance
            eval_tasks: Evaluation task list
            num_eval_runs: Number of evaluation runs

        Returns:
            Evaluation result
        """
        if eval_tasks is None:
            eval_tasks = self._get_default_tasks()

        eval_results = []
        for i in range(num_eval_runs):
            task = eval_tasks[i % len(eval_tasks)]
            result = self.run_rollout(
                task_id=f"eval_{i}",
                task_description=task["description"],
                initial_question=task["question"],
                agent=agent
            )
            eval_results.append(result)

        rewards = [r["total_reward"] for r in eval_results]
        steps = [r["num_steps"] for r in eval_results]

        return {
            "num_evaluations": len(eval_results),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_steps": np.mean(steps),
            "std_steps": np.std(steps),
            "detailed_results": eval_results
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        trajectories = self.rollout_collector.get_trajectories()
        training_history = self.grpo_trainer.get_training_history()

        return {
            "training_rounds": self.training_rounds,
            "best_reward": self.best_reward,
            "num_trajectories": len(trajectories),
            "num_training_steps": len(training_history),
            "judge_model": self.config.judge_model_path,
            "agent_model": self.config.agent_model_path
        }

    def _get_default_tasks(self) -> List[Dict[str, str]]:
        """Get default task list"""
        return [
            {
                "description": "Evaluate factual question answering ability",
                "question": "What is artificial intelligence?"
            },
            {
                "description": "Evaluate logical reasoning ability",
                "question": "If all cats can climb trees, and Tom is a cat, can Tom climb trees?"
            },
            {
                "description": "Evaluate common sense understanding",
                "question": "Why is the sky blue?"
            },
            {
                "description": "Evaluate mathematical calculation ability",
                "question": "Calculate 15 times 23 equals how much?"
            }
        ]

    def reset_system(self):
        """Reset system state"""
        self.rollout_collector.clear_trajectories()
        self.grpo_trainer.reset_training_history()
        self.training_rounds = 0
        self.best_reward = -float('inf')


# Convenience function
def create_llm_grpo_system(
    agent_model_path: str,
    judge_model_path: str,
    max_turns: int = 3,
    storage_path: Optional[str] = None,
    grpo_config: Optional[GRPOConfig] = None
) -> LLMGRPOSystem:
    """
    Convenience function to create LLM + GRPO system

    Args:
        agent_model_path: Agent model path
        judge_model_path: Judge model path
        max_turns: Maximum turns
        storage_path: Storage path
        grpo_config: GRPO configuration

    Returns:
        LLMGRPOSystem instance
    """
    config = LLMGRPOSystemConfig(
        agent_model_path=agent_model_path,
        judge_model_path=judge_model_path,
        max_turns=max_turns,
        storage_path=storage_path,
        grpo_config=grpo_config
    )

    return LLMGRPOSystem(config)