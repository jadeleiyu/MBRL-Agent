# ==============================================
# File: mbrl_agent/config.py
# ==============================================
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPaths:
    policy_base: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    world_model_base: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    world_model_adapter: str = (
        "LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-WM-webarena-16k-adapter"
    )
    reward_model: str = "zai-org/webrl-orm-llama-3.1-8b"


@dataclass
class TrainingArgs:
    # SFT
    sft_lr: float = 1e-5
    sft_batch_size: int = 4
    sft_epochs: int = 2
    max_seq_len: int = 4096
    gradient_accumulation_steps: int = 8

    # PPO
    ppo_batch_size: int = 64
    ppo_mini_batch_size: int = 8
    ppo_epochs: int = 2
    ppo_lr: float = 5e-6
    ppo_clip_range: float = 0.2
    target_kl: float = 0.1
    generation_max_new_tokens: int = 256
    rollout_horizon: int = 10

    # System
    bf16: bool = True
    use_8bit: bool = False
    fsdp: Optional[str] = "full_shard auto_wrap"