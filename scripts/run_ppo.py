# ==============================================
# File: scripts/run_ppo.py
# ==============================================
import os
from datasets import load_dataset
from trl import PPOConfig

from ..config import ModelPaths, TrainingArgs
from ..envs.wm_env import WMAWorldModelEnv, ORMRewardModel
from ..agents.agent import WebAgentLoop
from ..training.ppo_train import run_model_based_ppo
from ..data.datasets import load_mind2web, load_agentinstruct

if __name__ == "__main__":
    mp = ModelPaths()
    targs = TrainingArgs()

    # 1) load world model simulator
    env = WMAWorldModelEnv(base_model=mp.world_model_base, adapter=mp.world_model_adapter)

    # 2) load reward model
    orm = ORMRewardModel(mp.reward_model)

    # 3) glue into an agent loop
    loop = WebAgentLoop(env, orm, horizon=targs.rollout_horizon)

    # 4) seed instructions from datasets
    ds1 = load_mind2web("train").select(range(200))
    ds2 = load_agentinstruct("train").select(range(200))
    instructions = [ex["instruction"] for ex in ds1] + [ex["instruction"] for ex in ds2]

    # 5) PPO config
    ppo_cfg = PPOConfig(
        model_name=os.getenv("POLICY_CKPT", "./checkpoints/sft-llama3.1-8b"),
        learning_rate=targs.ppo_lr,
        batch_size=targs.ppo_batch_size,
        mini_batch_size=targs.ppo_mini_batch_size,
        ppo_epochs=targs.ppo_epochs,
        target_kl=targs.target_kl,
        gradient_accumulation_steps=targs.gradient_accumulation_steps,
    )

    policy, trainer = run_model_based_ppo(
        policy_model_name=ppo_cfg.model_name,
        agent_loop=loop,
        instructions=instructions,
        ppo_config=ppo_cfg,
        max_new_tokens=targs.generation_max_new_tokens,
    )

    out_dir = os.getenv("PPO_OUT", "./checkpoints/ppo-llama3.1-8b")
    trainer.save_pretrained(out_dir)
    policy.tokenizer.save_pretrained(out_dir)
    print(f"PPO policy saved to {out_dir}")


