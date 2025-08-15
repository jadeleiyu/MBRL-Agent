# ==============================================
# File: critic.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


@dataclass
class CriticConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    num_labels: int = 1
    device_map: str = "auto"
    torch_dtype = torch.bfloat16


class Critic:
    """Sequence-level critic: scores (partial) trajectories.

    - score(instruction, fragment) → scalar in [0,1] (sigmoid normalization)
    - train_critic(dataset) → learns to predict discounted returns G_t from sparse final rewards

    fragment format: list of tuples (obs_t, action_t, obs_{t+1}) for t=0..k-1
    """

    def __init__(self, cfg: CriticConfig = CriticConfig()):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, problem_type="regression", num_labels=cfg.num_labels,
            device_map=cfg.device_map, torch_dtype=cfg.torch_dtype
        )
        self.model.eval()

    def _render_traj(self, instruction: str, fragment: List):
        # fragment: [(obs, act, next_obs), ...]
        lines = [f"Instruction: {instruction}"]
        for t, (s, a, ns) in enumerate(fragment):
            lines.append(f"t={t} a={a}")
            lines.append(f"obs: {s.get('acc_tree','')[:1500]}")
            lines.append(f"next: {ns.get('acc_tree','')[:1500]}")
        return "\n".join(lines)

    @torch.inference_mode()
    def score(self, instruction: str, fragment: List) -> float:
        text = self._render_traj(instruction, fragment)
        toks = self.tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        out = self.model(**toks, return_dict=True)
        val = out.logits.squeeze().float().cpu().item()
        # map to [0,1]
        return float(torch.sigmoid(torch.tensor(val)).item())


# ---------------- Training ----------------
@dataclass
class CriticTrainConfig:
    lr: float = 5e-6
    batch_size: int = 4
    grad_accum: int = 2
    epochs: int = 1
    bf16: bool = True
    gamma: float = 0.99


def train_critic(dataset: List[Dict[str, Any]], cfg: CriticTrainConfig, critic: Critic):
    """Train critic to predict discounted returns G_t from sparse final rewards.

    dataset: list of episodes with keys:
      - instruction: str
      - steps: [{"observation": s_t, "action": a_t, "next_observation": s_{t+1}}...]
      - reward: float  # final reward for episode (sparse, at terminal)
    """
    texts, targets = [], []
    for ep in dataset:
        instr = ep.get("instruction", "")
        steps = ep.get("steps", [])
        R = float(ep.get("reward", 0.0))
        # compute discounted returns backwards with sparse final reward
        G = []
        g = R
        for _ in reversed(steps):
            G.append(g)
            g = cfg.gamma * g
        G = list(reversed(G))
        # build per-step samples
        for t in range(len(steps)):
            frag = steps[: t + 1]
            # map to (s,a,s') triples
            triples = [(frag[i]["observation"], frag[i]["action"], frag[i]["next_observation"]) for i in range(len(frag))]
            texts.append(Critic._render_traj(critic, instr, triples))
            targets.append(G[t])

    class _Ds(torch.utils.data.Dataset):
        def __init__(self, xs, ys): self.xs = xs; self.ys = ys
        def __len__(self): return len(self.xs)
        def __getitem__(self, i): return {"text": self.xs[i], "label": float(self.ys[i])}

    ds = _Ds(texts, targets)
    tok = critic.tokenizer

    def collate(batch):
        toks = tok([b["text"] for b in batch], return_tensors="pt", padding=True, truncation=True)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        toks["labels"] = labels.unsqueeze(-1)
        return toks

    args = TrainingArguments(
        output_dir="./critic-sft",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        bf16=cfg.bf16,
        logging_steps=50,
        save_total_limit=2,
    )
    trainer = Trainer(model=critic.model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    trainer.save_model("./critic-sft")
