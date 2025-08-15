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
    planner_policy.py       # Planner wrapper (K candidates, d-step world model, critic scoring)
world_model/
    web_world_model.py      # LLM world model (delta edits → next observation) + SFT trainer
    critic.py               # Critic (sequence regression) + trainer predicting discounted returns
traj_collect/
    collect_real_trajs.py   # Sandbox loop (T1/T2 collection); PPO-friendly callback hook
    collect_dream_trajs.py  # Dreamed T3 generator + PPO dreamed rollout helper
    traj_processing.py      # Compute per-step return/advantage with critic
train/
    awr.py                  # Advantage-Weighted Regression (off-policy) training
    ppo.py                  # PPO training from env, world-model dreams, or replay (TRL)
envs/
  WebArena/             # your adapter to WebArena’s sandbox (not included)
  WebVoyager/           # optional live web env adapter (not included)
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

**Processed (for AWR/PPO replay)**:

```json
{
  "instruction": "...",
  "steps": [
    {"observation": {...}, "action": "...", "return": 0.73, "advantage": 0.21}
  ]
}
```

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
