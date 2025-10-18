"""
GRPO (Group Relative Policy Optimization) Trainer
Policy optimization based on rollout data
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class GRPOConfig:
    """GRPO training configuration"""
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    advantage_clip: float = 10.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    group_size: int = 4  # Group size


class GRPOTrainer:
    """GRPO trainer"""

    def __init__(self, config: Optional[GRPOConfig] = None):
        """
        Initialize GRPO trainer

        Args:
            config: Training configuration
        """
        self.config = config or GRPOConfig()
        self.training_history: List[Dict[str, Any]] = []

    def prepare_training_data(
        self,
        trajectories: List[Any],
        reference_policy_logprobs: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Prepare training data

        Args:
            trajectories: Rollout trajectory data
            reference_policy_logprobs: Reference policy log probabilities (optional)

        Returns:
            Training data dictionary
        """
        # Extract state, action, reward sequences
        states = []
        actions = []
        rewards = []
        dones = []

        for traj in trajectories:
            for step in traj.steps:
                states.append(step.observation)
                actions.append(step.action)
                rewards.append(step.reward)
                dones.append(step.done)

        # Calculate advantage functions
        advantages = self._compute_advantages(rewards, dones)

        # Use reference policy log probabilities if provided
        if reference_policy_logprobs is not None:
            assert len(reference_policy_logprobs) == len(actions), \
                "Reference policy logprobs count must match action count"

        training_data = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "advantages": advantages,
            "reference_logprobs": reference_policy_logprobs,
            "num_samples": len(states)
        }

        return training_data

    def train_step(
        self,
        training_data: Dict[str, Any],
        current_policy_logprobs: List[float],
        values: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Execute one training step

        Args:
            training_data: Training data
            current_policy_logprobs: Current policy log probabilities
            values: State value function values (optional)

        Returns:
            Training statistics
        """
        states = training_data["states"]
        actions = training_data["actions"]
        advantages = training_data["advantages"]
        reference_logprobs = training_data.get("reference_logprobs")

        # Ensure data lengths match
        assert len(states) == len(actions) == len(advantages) == len(current_policy_logprobs)

        # Calculate policy loss
        policy_loss = self._compute_policy_loss(
            current_policy_logprobs,
            advantages,
            reference_logprobs
        )

        # Calculate value loss (if value function provided)
        value_loss = 0.0
        if values is not None:
            value_loss = self._compute_value_loss(values, advantages)

        # Calculate entropy regularization
        entropy = self._compute_entropy(current_policy_logprobs)

        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_coef * value_loss -
            self.config.entropy_coef * entropy
        )

        # Record training statistics
        stats = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "mean_advantage": np.mean(advantages),
            "std_advantage": np.std(advantages)
        }

        self.training_history.append(stats)
        return stats

    def _compute_advantages(
        self,
        rewards: List[float],
        dones: List[bool],
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> List[float]:
        """
        Calculate Generalized Advantage Estimation (GAE)

        Args:
            rewards: Reward sequence
            dones: Termination flag sequence
            gamma: Discount factor
            lambda_: GAE parameter

        Returns:
            Advantage function value sequence
        """
        advantages = []
        advantage = 0.0

        # Calculate GAE backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0.0

            delta = rewards[t] - (0 if t == len(rewards) - 1 else 0)  # Simplified version
            advantage = delta + gamma * lambda_ * (0 if dones[t] else advantage)
            advantages.insert(0, advantage)

        # Normalize advantage function
        advantages = np.array(advantages)
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages.tolist()

    def _compute_policy_loss(
        self,
        current_logprobs: List[float],
        advantages: List[float],
        reference_logprobs: Optional[List[float]] = None
    ) -> float:
        """
        Calculate policy loss

        Args:
            current_logprobs: Current policy log probabilities
            advantages: Advantage function values
            reference_logprobs: Reference policy log probabilities

        Returns:
            Policy loss value
        """
        current_logprobs = np.array(current_logprobs)
        advantages = np.array(advantages)

        if reference_logprobs is not None:
            # Use reference policy for relative policy optimization
            reference_logprobs = np.array(reference_logprobs)
            log_ratios = current_logprobs - reference_logprobs
            ratios = np.exp(log_ratios)

            # Clip advantages
            advantages = np.clip(advantages, -self.config.advantage_clip, self.config.advantage_clip)

            # Policy loss
            policy_loss = -np.mean(ratios * advantages)
        else:
            # Standard policy gradient loss
            policy_loss = -np.mean(current_logprobs * advantages)

        return float(policy_loss)

    def _compute_value_loss(self, values: List[float], advantages: List[float]) -> float:
        """
        Calculate value function loss

        Args:
            values: Value function predictions
            advantages: Advantage function values

        Returns:
            Value loss value
        """
        values = np.array(values)
        advantages = np.array(advantages)

        # Simple value function loss
        value_loss = np.mean((values - advantages) ** 2)
        return float(value_loss)

    def _compute_entropy(self, logprobs: List[float]) -> float:
        """
        Calculate policy entropy

        Args:
            logprobs: Log probability sequence

        Returns:
            Entropy value
        """
        logprobs = np.array(logprobs)
        probs = np.exp(logprobs)
        entropy = -np.sum(probs * logprobs, axis=-1) if len(logprobs.shape) > 1 else -probs * logprobs
        return float(np.mean(entropy))

    def group_trajectories(
        self,
        trajectories: List[Any],
        group_size: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Group trajectories

        Args:
            trajectories: Trajectory list
            group_size: Group size

        Returns:
            Grouped trajectory list
        """
        group_size = group_size or self.config.group_size
        groups = []

        for i in range(0, len(trajectories), group_size):
            group = trajectories[i:i + group_size]
            groups.append(group)

        return groups

    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history"""
        return self.training_history.copy()

    def reset_training_history(self):
        """Reset training history"""
        self.training_history.clear()


# Convenience function
def create_grpo_trainer(config: Optional[GRPOConfig] = None) -> GRPOTrainer:
    """
    Convenience function to create GRPO trainer

    Args:
        config: Training configuration

    Returns:
        GRPOTrainer instance
    """
    return GRPOTrainer(config)