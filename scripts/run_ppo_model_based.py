# ==============================================
# File: scripts/run_ppo_model_based.py
# ==============================================
import os
from trl import PPOConfig

from ..config import TrainingArgs
from ..envs.wm_env import WebWorldModel, ORMRewardModel
from ..training.ppo_model_based import run_model_based_ppo
from ..data.datasets import mind2web_instructions, webrl_instructions


if __name__ == "__main__":
    targs = TrainingArgs()

    # World model (+ encoded-state I/O) and reward model
    world = WebWorldModel()
    orm = ORMRewardModel()

    # Build instruction pool
    mind2web_path = os.getenv("MIND2WEB_JSON", "/mnt/data/mind2web_sample.json")
    webrl_path = os.getenv("WEBRL_JSON", "/mnt/data/webrl_sample.json")

    instr = []
    instr += mind2web_instructions(local_path=mind2web_path if os.path.exists(mind2web_path) else None,
                                   split=os.getenv("MIND2WEB_SPLIT", "train"))
    if os.path.exists(webrl_path):
        instr += webrl_instructions(local_path=webrl_path)
    max_instr = int(os.getenv("PPO_MAX_INSTR", "200"))
    instructions = instr[:max_instr]

    # PPO config (policy starts from SFT ckpt)
    policy_ckpt = os.getenv("POLICY_CKPT", "./checkpoints/sft-llama3.1-8b")
    ppo_cfg = PPOConfig(
        model_name=policy_ckpt,
        learning_rate=targs.ppo_lr,
        batch_size=targs.ppo_batch_size,
        mini_batch_size=targs.ppo_mini_batch_size,
        ppo_epochs=targs.ppo_epochs,
        target_kl=targs.target_kl,
        gradient_accumulation_steps=targs.gradient_accumulation_steps,
    )

    agent, trainer = run_model_based_ppo(
        policy_ckpt=policy_ckpt,
        world_model=world,
        reward_model=orm,
        instructions=instructions,
        ppo_config=ppo_cfg,
        horizon=targs.rollout_horizon,
        max_new_tokens=targs.generation_max_new_tokens,
    )

    out_dir = os.getenv("PPO_OUT", "./checkpoints/ppo-llama3.1-8b-encoded")
    trainer.save_pretrained(out_dir)
    agent.tokenizer.save_pretrained(out_dir)
    print(f"PPO policy saved to {out_dir}")
