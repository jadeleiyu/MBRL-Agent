# ==============================================
# File: awr.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


@dataclass
class AWRConfig:
    beta: float = 1.0
    weight_clip: float = 20.0
    lr: float = 5e-6
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 1
    bf16: bool = True


class AWRDataset(torch.utils.data.Dataset):
    def __init__(self, episodes: List[Dict[str, Any]], tokenizer: AutoTokenizer):
        self.samples = []
        self.tok = tokenizer
        for ep in episodes:
            instr = ep.get("instruction", "")
            for st in ep.get("steps", []):
                obs = st.get("observation", {})
                acc = (obs.get("acc_tree", "") or "")[:3000]
                url = obs.get("url", "")
                prompt = (
                    f"Instruction: {instr}\nURL: {url}\n\n"
                    f"# Current Accessibility Tree\n{acc}\n\n"
                    f"Return next one-line action now."
                )
                action = st.get("action", "do(action=\"Wait\")")
                adv = float(st.get("advantage", 0.0))
                w = float(np.exp(adv / 1.0))  # temperature handled in trainer
                self.samples.append({"text": prompt + action, "raw_weight": w})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class AWRTrainer(Trainer):
    def __init__(self, beta: float, weight_clip: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.weight_clip = weight_clip

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        weights = inputs.pop("weights")
        out = model(**inputs)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="none"
        ).view(labels.size(0), -1).mean(dim=1)
        w = torch.clamp(weights, max=self.weight_clip)
        loss = (w * ce).mean()
        return (loss, out) if return_outputs else loss


def collate_fn(tokenizer: AutoTokenizer, beta: float):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        raw_w = torch.tensor([b["raw_weight"] for b in batch], dtype=torch.float32)
        # anneal/temperature: raw_w already uses adv/1.0; scale here by 1/beta
        weights = torch.pow(raw_w, 1.0 / max(1e-6, beta))
        toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        toks["labels"] = toks["input_ids"].clone()
        toks["weights"] = weights
        return toks
    return _fn


def run_awr(policy_model_name: str, processed_path: str, out_dir: str, cfg: AWRConfig = AWRConfig()):
    # load processed episodes
    episodes = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(__import__('json').loads(line))

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_model_name, device_map="auto", torch_dtype=torch.bfloat16)

    ds = AWRDataset(episodes, tokenizer)
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        bf16=cfg.bf16,
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = AWRTrainer(beta=cfg.beta, weight_clip=cfg.weight_clip, model=model, args=args, data_collator=collate_fn(tokenizer, cfg.beta), train_dataset=ds, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir
