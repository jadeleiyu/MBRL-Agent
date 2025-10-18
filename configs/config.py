"""
Configuration management using dotenv
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: str
    judge_model_path: str
    agent_model_path: str
    environment_model_path: str


@dataclass
class TrainingConfig:
    """Training configuration"""
    max_turns: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    group_size: int


@dataclass
class LLMConfig:
    """LLM inference configuration"""
    temperature: float
    top_p: float
    max_tokens: int


@dataclass
class StorageConfig:
    """Storage configuration"""
    storage_path: str
    log_path: str
    checkpoint_path: str


@dataclass
class SystemConfig:
    """System configuration"""
    enable_logging: bool
    debug_mode: bool
    save_trajectories: bool


class DevWebConfig:
    """Main configuration class for DevWeb"""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration from environment file

        Args:
            env_file: Path to .env file, if None uses default .env
        """
        if env_file:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        else:
            from dotenv import load_dotenv
            load_dotenv()

        # Model configuration
        self.model = ModelConfig(
            model_path=os.getenv('MODEL_PATH', '/mnt/shared-storage-user/medeval-share/model/Qwen2.5-7B-Instruct'),
            judge_model_path=os.getenv('JUDGE_MODEL_PATH', os.getenv('MODEL_PATH')),
            agent_model_path=os.getenv('AGENT_MODEL_PATH', os.getenv('MODEL_PATH')),
            environment_model_path=os.getenv('ENVIRONMENT_MODEL_PATH', os.getenv('MODEL_PATH'))
        )

        # Training configuration
        self.training = TrainingConfig(
            max_turns=int(os.getenv('MAX_TURNS', '3')),
            batch_size=int(os.getenv('BATCH_SIZE', '32')),
            learning_rate=float(os.getenv('LEARNING_RATE', '1e-5')),
            num_epochs=int(os.getenv('NUM_EPOCHS', '3')),
            group_size=int(os.getenv('GROUP_SIZE', '4'))
        )

        # LLM configuration
        self.llm = LLMConfig(
            temperature=float(os.getenv('TEMPERATURE', '0.2')),
            top_p=float(os.getenv('TOP_P', '0.9')),
            max_tokens=int(os.getenv('MAX_TOKENS', '512'))
        )

        # Storage configuration
        self.storage = StorageConfig(
            storage_path=os.getenv('STORAGE_PATH', './data/trajectories.jsonl'),
            log_path=os.getenv('LOG_PATH', './logs/'),
            checkpoint_path=os.getenv('CHECKPOINT_PATH', './checkpoints/')
        )

        # System configuration
        self.system = SystemConfig(
            enable_logging=os.getenv('ENABLE_LOGGING', 'true').lower() == 'true',
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'false',
            save_trajectories=os.getenv('SAVE_TRAJECTORIES', 'true').lower() == 'true'
        )

    def validate(self):
        """Validate configuration"""
        # Check model paths
        if not os.path.exists(self.model.model_path):
            print(f"⚠ Warning: Model path {self.model.model_path} does not exist")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.storage.storage_path), exist_ok=True)
        os.makedirs(self.storage.log_path, exist_ok=True)
        os.makedirs(self.storage.checkpoint_path, exist_ok=True)

        return True


# Global configuration instance
_config: Optional[DevWebConfig] = None


def get_config(env_file: Optional[str] = None) -> DevWebConfig:
    """
    Get or create global configuration instance

    Args:
        env_file: Path to .env file

    Returns:
        DevWebConfig instance
    """
    global _config
    if _config is None:
        _config = DevWebConfig(env_file)
        _config.validate()
    return _config


def set_config(config: DevWebConfig):
    """
    Set global configuration

    Args:
        config: DevWebConfig instance
    """
    global _config
    _config = config