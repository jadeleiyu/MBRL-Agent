# ==============================================
# File: scripts/run_ppo_model_free.py
# ==============================================
"""
Off-policy, model-free PPO launcher.

Usage examples
--------------
# Train from a JSONL buffer collected earlier (one episode per line)
python -m scripts.run_ppo_model_free \
  --policy_ckpt ./checkpoints/sft-llama3.1-8b \
  --replay ./rollouts/webarena_buffer.jsonl \
  --orm_model zai-org/webrl-orm-llama-3.1-8b \
  --max_episodes 1000 --max_steps_per_ep 10

You can pass multiple --replay files; they will be concatenated.
"""
import os
import argparse

from trl import PPOConfig

from ..config import TrainingArgs
from ..envs.wm_env import ORMRewardModel
from ..training.ppo_model_free import run_model_free_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", default=os.getenv("POLICY_CKPT", "./checkpoints/sft-llama3.1-8b"))
    ap.add_argument("--orm_model", default=os.getenv("ORM_MODEL", "zai-org/webrl-orm-llama-3.1-8b"))
    ap.add_argument("--replay", action="append", required=True, help="Path(s) to JSON/JSONL replay files.")
    ap.add_argument("--max_episodes", type=int, default=int(os.getenv("MAX_EPISODES", "10000")))
    ap.add_argument("--max_steps_per_ep", type=int, default=int(os.getenv("MAX_STEPS_PER_EP", "10")))

    args = ap.parse_args()

    targs = TrainingArgs()
    ppo_cfg = PPOConfig(
        model_name=args.policy_ckpt,
        learning_rate=targs.ppo_lr,
        batch_size=targs.ppo_batch_size,
        mini_batch_size=targs.ppo_mini_batch_size,
        ppo_epochs=targs.ppo_epochs,
        target_kl=targs.target_kl,
        gradient_accumulation_steps=targs.gradient_accumulation_steps,
    )

    orm = ORMRewardModel(args.orm_model)

    policy, trainer = run_model_free_ppo(
        policy_model_name=args.policy_ckpt,
        ppo_config=ppo_cfg,
        replay_paths=args.replay,
        orm_reward_model=orm,
        max_episodes=args.max_episodes,
        max_steps_per_ep=args.max_steps_per_ep,
    )

    out_dir = os.getenv("PPO_OUT", "./checkpoints/ppo-llama3.1-8b-offpolicy")
    trainer.save_pretrained(out_dir)
    policy.tokenizer.save_pretrained(out_dir)
    print(f"Off-policy PPO policy saved to {out_dir}")


if __name__ == "__main__":
    main()
