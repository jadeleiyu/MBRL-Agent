# Model-Based PPO for LLM Web Agents (Text-Only)

This is a **reference implementation** for training an LLM web agent with **model-based online RL (PPO)** using:

- **Policy**: Llama 3.1 8B Instruct (with a value head via TRL)
- **World model (simulator)**: WMA Adapter — `LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-WM-webarena-16k-adapter`
- **Reward model**: WebRL ORM — `zai-org/webrl-orm-llama-3.1-8b`
- **Pretraining corpora**: Mind2Web + AgentInstruct

> ⚠️ This code targets *research prototyping*. It uses prompt-based interfaces for both the world model and reward model, which mirrors the public artifacts. You can swap them with more structured heads if the upstream repos expose them.

## Setup

```bash
conda create -n mbrl-agent python=3.10 -y
conda activate mbrl-agent
pip install -r requirements.txt

# (optional) set HF tokens for gated Llama models
export HF_TOKEN=xxx
```

## Supervised Fine-Tuning (warm start)

```bash
# Edit env vars if needed
export POLICY_BASE=meta-llama/Meta-Llama-3.1-8B-Instruct
export SFT_OUT=./checkpoints/sft-llama3.1-8b
python scripts/run_sft.py
```

This produces a small SFT checkpoint used as the PPO initialization.

## Model-Based PPO

```bash
# Ensure your SFT checkpoint exists
export POLICY_CKPT=./checkpoints/sft-llama3.1-8b
python scripts/run_ppo.py
```

The PPO loop:
1. Samples instructions from Mind2Web + AgentInstruct.
2. For each instruction, rolls out *entirely* inside the **world model** (no live browser) for up to `rollout_horizon` steps.
3. After each action, computes a step reward via the **ORM** model.
4. Updates the policy with TRL's PPO using step-wise rewards.

## Multi-GPU / Cluster Notes

- This repo uses **Accelerate/DeepSpeed** via TRL under the hood. Launch with:

```bash
accelerate launch --config_file your_accelerate_config.yaml scripts/run_ppo.py
```

- For 8× H100/H200, prefer BF16 and FSDP. You can also quantize the policy to 8-bit during **inference** for rollouts by setting `use_8bit=True` in `PolicyWithValue` if memory is tight.

## Customize

- **Prompts**: adjust `mbrl/utils/prompting.py` for better JSON stability and action schema.
- **World model**: if you have the actual WMA code that returns structured diffs, replace `WMAWorldModelEnv.predict_next` accordingly.
- **Rewards**: If the ORM model exposes a scalar head in the future, replace the prompt parser in `ORMRewardModel`.

## Caveats

- Mind2Web/AgentInstruct are not 1:1 action-schema datasets; they seed high-level goals well but won’t directly supervise low-level DOM actions. The SFT stage here is a *light warm-start*.
- The world model and reward model are **LLM prompts**; malformed JSON can occur — we defensively parse and clip rewards.

## License
This template is provided for research use; respect the licenses of upstream models/datasets.
