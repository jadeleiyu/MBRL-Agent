# Model-Based + Off-Policy Web Agent (Refactored)

This repo provides a clean, modular scaffold to train LLM-based **web agents** with:

* **Vanilla policy** action selection
* **Planner policy** (short-horizon world model lookahead)
* **Web world model** (delta edits → next observation)
* **Critic / value model** for scoring partial or full trajectories
* **Real** and **dreamed** trajectory collection
* **Trajectory processing** (return/advantage labeling)
* **Off-policy AWR** training on a mixture of real + dreamed data
* **Flexible PPO** training from live env, world-model dreams, or replay

> Environments (WebArena, WebVoyager) are assumed to live under `envs/` and are not included. See WebArena’s **Quick Walkthrough** for an example adapter.

---

## File layout

```
agents/
    vanilla_policy.py       # Vanilla policy (LLM + value head) → one-line action
    planner_policy.py       # Planner-CoT-based policy (K candidates, d-step world model, critic scoring, TBD)
world_model/
    web_world_model.py      # LLM world model that predicts next web page accessibility tree
    critic.py               # Critic that predicts value/advantage of (partial) trajectory
traj_collect/
    collect_real_trajs.py   # Sandbox loop (T1/T2 collection); PPO-friendly callback hook (TBD)
    collect_dream_trajs.py  # Collecting agent - world model interaction trajectories
    traj_processing.py      # Compute per-step return/advantage with critic (TBD)
train/
    awr.py                  # Advantage-Weighted Regression (off-policy) training
    ppo.py                  # PPO training from env, world-model dreams, or replay (TRL)
```

---

## Setup

**Python:** 3.10+

**Install deps:**

```bash
pip install torch transformers trl peft accelerate scikit-learn
```

> Use a CUDA build of PyTorch and BF16-capable GPUs for best performance.

**Models:**

* Policy base: `meta-llama/Meta-Llama-3.1-8B-Instruct` (changeable).
* World model adapter: set inside `web_world_model.py` (`WMConfig.adapter`).
* Critic: initialized as a regression head over the same base; you can swap to a trained value/reward model.

---

## Environment adapters (expected API)

Create an adapter under `envs/WebArena` that minimally exposes:

```python
class WebArenaEnv:
    def reset(self, task: dict) -> dict:  # return initial observation
        return {"url": str, "title": str, "acc_tree": str, "candidates": list}

    def step(self, action: str) -> tuple[dict, bool, dict]:
        # returns (next_obs, done, info)
        ...
```

Observations are **accessibility-tree–centric**. Keep strings reasonably short (we truncate internally).

---

## Typical workflow

### 0) (Optional) SFT a base policy

Start from your favorite instruction-tuned LLM; optionally SFT on Mind2Web/WebRL demos.

### 1) Collect trajectories

* **T1** (vanilla policy in sandbox):

```python
from envs.WebArena.adapter import WebArenaEnv
from vanilla_policy import VanillaPolicy
from collect_real_trajs import Collector, CollectConfig

env = WebArenaEnv()
policy = VanillaPolicy()
collector = Collector(env, policy, critic=None,
                      cfg=CollectConfig(episodes=200, horizon=10, save_path="./rollouts/T1.jsonl"))
collector.run()
```

* **T2** (planner policy in sandbox):

```python
from planner_policy import PlannerPolicy, PlannerConfig
from web_world_model import WebWorldModel
from critic import Critic
from collect_real_trajs import Collector, CollectConfig

wm = WebWorldModel()
critic = Critic()
planner = PlannerPolicy(base=policy, wm=wm, critic=critic, cfg=PlannerConfig(num_candidates=6, horizon=2))
collector = Collector(env, planner, critic=None,
                      cfg=CollectConfig(episodes=200, horizon=10, save_path="./rollouts/T2.jsonl"))
collector.run()
```

* **T3** (dreamed with world model):

```python
from collect_dream_trajs import DreamCollector, DreamConfig

dreamer = DreamCollector(policy, wm,
                         cfg=DreamConfig(episodes=1000, horizon=3, save_path="./rollouts/T3.jsonl"))
dreamer.run()
```

### 2) Label returns & advantages with the critic

```python
from traj_processing import load_replay, label_returns_and_adv, save_processed
from critic import Critic

critic = Critic()
T1 = load_replay(["./rollouts/T1.jsonl"]) 
T2 = load_replay(["./rollouts/T2.jsonl"]) 
T3 = load_replay(["./rollouts/T3.jsonl"]) 
processed = label_returns_and_adv(T1 + T2 + T3, critic)
save_processed(processed, "./rollouts/processed.jsonl")
```

### 3) Train with AWR (off-policy)

```python
from awr import run_awr, AWRConfig

run_awr(
  policy_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
  processed_path="./rollouts/processed.jsonl",
  out_dir="./checkpoints/awr-policy",
  cfg=AWRConfig(beta=1.0, weight_clip=20.0, lr=5e-6)
)
```

### 4) PPO training (env, world-model, or replay)

* **From replay**:

```python
from ppo import run_ppo, PPORunConfig

run_ppo(env=None, instructions=[], processed_replay_path="./rollouts/processed.jsonl",
        cfg=PPORunConfig())
```

* **From live environment**:

```python
from envs.WebArena.adapter import WebArenaEnv
from ppo import run_ppo, PPORunConfig
from critic import Critic

env = WebArenaEnv()
critic = Critic()
run_ppo(env, instructions=["Find ...", "Book ..."], cfg=PPORunConfig(), critic=critic, horizon=10)
```

* **From world model (online dreams)**:

```python
from web_world_model import WebWorldModel
from ppo import run_ppo, PPORunConfig
from critic import Critic

wm = WebWorldModel()
critic = Critic()
run_ppo(env=None, world=wm, instructions=["Add Blue Tee to cart"],
        cfg=PPORunConfig(max_new_tokens=64), critic=critic, horizon=4)
```

> You can interleave env and dreamed batches by calling `run_ppo` multiple times or by stitching the step lists before `trainer.step`.

---

## Data formats

**Episode (T1/T2/T3)** — one JSON per line:

```json
{
  "type": "T1|T2|T3",
  "instruction": "...",
  "steps": [
    {
      "observation": {"url": "...", "title": "...", "acc_tree": "...", "candidates": [...]},
      "action": "do(action=\"Click\", element=\"7\")",
      "next_observation": {"url": "...", "acc_tree": "..."},
      "delta": {"edits": [...]},
      "done": false
    }
  ]
}
```

启用强化学习时，可通过 `scripts/agent/webarena/gspo.sh` 中的 `STANDARD_ANSWER_K`、`STANDARD_ANSWER_REWARD` 环境变量（对应配置 `rllm.standard_answer.inject_every_k`、`rllm.standard_answer.reward`）控制是否每隔若干训练 step 将 `standard_answer_messages` 注入到 rollout 批次中，超长 prompt/response 会被预先过滤。

**Processed (for AWR/PPO replay)**:

```json
{
  "instruction": "...",
  "steps": [
    {"observation": {...}, "action": "...", "return": 0.73, "advantage": 0.21}
  ]
}
```

### nnetnav → WebArena 标准答案数据

脚本 `world/build_webarena_trajs_from_nnetnav.py` 可将 `stanfordnlp/nnetnav-*` 数据转成 WebArena 训练样本，并为每条样本附加一条与前向/反向序列一致的 `standard_answer_messages`。该字段是一个完整的 message list（system + user + assistant），assistant 消息可由内置模板生成，也可通过 `--completion_model` 指定本地 HuggingFace Causal LM 来“补全”推理细节。

```bash
python world/build_webarena_trajs_from_nnetnav.py \
  --input_jsonl filtered.jsonl \
  --train_jsonl data/webarena_trajs.train.jsonl \
  --test_jsonl data/webarena_trajs.test.jsonl \
  --train_parquet data/webarena_trajs.train.parquet \
  --test_parquet data/webarena_trajs.test.parquet \
  --train_ratio 0.9 \
  --completion_model path/to/your-model 
```

输出中的主体字段与原始训练数据完全一致，仅新增 `standard_answer_messages`：

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "OBJECTIVE:\n...\nCURRENT OBSERVATION:\n..."}
  ],
  "standard_answer_messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "OBJECTIVE:\n..."},
    {"role": "assistant", "content": "INTERACTION HISTORY SUMMARY:\n..."}
  ],
  "extra_info": {
    "trajectory_id": "webarena_openended_3417",
    "window_length": 3,
    "window_start_step": 5,
    "window_end_step": 7
  }
}
```

其中 assistant 消息始终携带标准动作（ACTION 行），并与原样本的起点对齐。如果未提供 `--completion_model`，脚本会根据已有 action/observation 生成结构化模板文本；若提供模型，则会将 system/user prompt 送入模型补全文本，确保和训练时的前向/反向格式保持一致。窗口规则沿用“只截末尾 2–5 步且每个原始任务最多 2 条样本”的约束。

> ⚠️ 训练阶段会在 **构建 RL 数据集** 时再次检查 `standard_answer_messages` 的 token 长度：当 `data.filter_standard_answer_messages`（默认启用）为真时，`verl.utils.dataset.RLHFDataset` 会用当前 tokenizer 解析该字段，并按照 `data.standard_answer_prompt_max_length`（默认等于 `data.max_prompt_length`）与 `data.standard_answer_response_max_length`（默认等于 `data.max_response_length`）的限制丢弃超长样本。这样可确保进入 rollout 的标准答案在 prompt/response 两段都不会超过最大序列长度。

---

## Notes & tips

* **Accessibility tree** is the primary observation. Truncate long pages to control token cost.
* The **world model** predicts **delta edits** then applies them. This stabilizes imagination and keeps dreams local.
* The **planner** uses short-horizon lookahead (H=2–3). Longer lookahead can drift.
* For **AWR**, mix T1/T2/T3 but prefer T1/T2. Weight or filter low-confidence T3 if needed.
* PPO here is minimalist; for GRPO or more advanced reward shaping, extend the trainer accordingly.

---

## Troubleshooting

* CUDA OOM → reduce `max_new_tokens`, batch size, or truncate observations.
* Bad actions → tighten action extraction regex or add decoding constraints.
* World model drift → shorten `horizon`, add schema checks to delta edits, reseed from real states frequently.

---

## License

Apache-2.0 (adjust as needed).
