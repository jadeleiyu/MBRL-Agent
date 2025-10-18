"""
Rollout Data Collector
Used to collect interaction trajectory data of agents in environments
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class RolloutStep:
    """Single rollout step data"""
    turn: int
    observation: Dict[str, Any]
    action: str
    reward: float
    done: bool
    info: Dict[str, Any]
    timestamp: str


@dataclass
class RolloutTrajectory:
    """Complete rollout trajectory"""
    task_id: str
    task_description: str
    agent_model: str
    env_model: str
    steps: List[RolloutStep]
    total_reward: float
    trajectory_id: str
    created_at: str


class RolloutCollector:
    """Rollout data collector"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize collector

        Args:
            storage_path: Data storage path, if None then don't save to file
        """
        self.storage_path = storage_path
        self.trajectories: List[RolloutTrajectory] = []
        self.current_trajectory: Optional[RolloutTrajectory] = None
        self.current_steps: List[RolloutStep] = []

    def start_trajectory(
        self,
        task_id: str,
        task_description: str,
        agent_model: str,
        env_model: str
    ):
        """
        Start new trajectory collection

        Args:
            task_id: Task ID
            task_description: Task description
            agent_model: Agent model identifier
            env_model: Environment model identifier
        """
        self.current_steps = []
        self.current_trajectory = RolloutTrajectory(
            task_id=task_id,
            task_description=task_description,
            agent_model=agent_model,
            env_model=env_model,
            steps=[],
            total_reward=0.0,
            trajectory_id=self._generate_trajectory_id(),
            created_at=datetime.now().isoformat()
        )

    def add_step(
        self,
        turn: int,
        observation: Dict[str, Any],
        action: str,
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ):
        """
        Add a step of data

        Args:
            turn: Current turn
            observation: Observation state
            action: Executed action
            reward: Obtained reward
            done: Whether episode ended
            info: Additional information
        """
        if self.current_trajectory is None:
            raise ValueError("Must call start_trajectory() first")

        step = RolloutStep(
            turn=turn,
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            info=info,
            timestamp=datetime.now().isoformat()
        )

        self.current_steps.append(step)
        self.current_trajectory.total_reward += reward

    def end_trajectory(self) -> RolloutTrajectory:
        """
        End current trajectory collection

        Returns:
            Complete trajectory data
        """
        if self.current_trajectory is None:
            raise ValueError("No active trajectory")

        self.current_trajectory.steps = self.current_steps.copy()
        trajectory = self.current_trajectory

        # Save to memory
        self.trajectories.append(trajectory)

        # Save to file (if storage path configured)
        if self.storage_path:
            self._save_trajectory(trajectory)

        # Reset current trajectory
        self.current_trajectory = None
        self.current_steps = []

        return trajectory

    def get_trajectories(self) -> List[RolloutTrajectory]:
        """Get all collected trajectories"""
        return self.trajectories.copy()

    def get_trajectory_by_id(self, trajectory_id: str) -> Optional[RolloutTrajectory]:
        """Get trajectory by ID"""
        for traj in self.trajectories:
            if traj.trajectory_id == trajectory_id:
                return traj
        return None

    def clear_trajectories(self):
        """Clear all trajectory data"""
        self.trajectories.clear()

    def _generate_trajectory_id(self) -> str:
        """Generate trajectory ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"traj_{timestamp}"

    def _save_trajectory(self, trajectory: RolloutTrajectory):
        """Save trajectory to file"""
        if not self.storage_path:
            return

        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        # Convert trajectory to dictionary
        traj_dict = asdict(trajectory)

        # Append to file
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(traj_dict, ensure_ascii=False) + '\n')

    def load_trajectories_from_file(self) -> List[RolloutTrajectory]:
        """Load trajectory data from file"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return []

        trajectories = []
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    traj_dict = json.loads(line.strip())
                    # Convert dictionary back to RolloutTrajectory object
                    trajectory = RolloutTrajectory(**traj_dict)
                    trajectories.append(trajectory)

        self.trajectories.extend(trajectories)
        return trajectories


# Convenience function
def create_rollout_collector(storage_path: Optional[str] = None) -> RolloutCollector:
    """
    Convenience function to create rollout collector

    Args:
        storage_path: Storage path

    Returns:
        RolloutCollector instance
    """
    return RolloutCollector(storage_path=storage_path)