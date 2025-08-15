# ==============================================
# File: web_world_model.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# This world model works over **accessibility tree** observations.
# It predicts small **delta edits** first, then applies them to form the next observation.


# ---------------------- delta representation ----------------------
@dataclass
class DeltaPlan:
    """A structured set of edits to apply to the current observation.

    Supported ops: UPDATE_TEXT, UPDATE_ATTR, ADD_CANDIDATE, REMOVE_CANDIDATE, NAVIGATE, DONE
    """
    raw_json: Dict[str, Any]


# ---------------------- world model ----------------------
@dataclass
class WMConfig:
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    adapter: Optional[str] = "LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-WM-webarena-16k-adapter"
    device_map: str = "auto"
    torch_dtype = torch.bfloat16
    max_new_tokens: int = 512
    max_obs_chars: int = 10000


DELTA_SCHEMA = (
    "Return STRICT JSON with key 'edits' (list) and optional 'reason' and 'done'.\n"
    "Each edit is one of:\n"
    "- {\"op\":\"UPDATE_TEXT\", \"id\":str, \"text\":str}\n"
    "- {\"op\":\"UPDATE_ATTR\", \"id\":str, \"attrs\":{...}}\n"
    "- {\"op\":\"ADD_CANDIDATE\", \"candidate\":{id, role, text, selector, ...}}\n"
    "- {\"op\":\"REMOVE_CANDIDATE\", \"id\":str}\n"
    "- {\"op\":\"NAVIGATE\", \"url\":str, \"title\":str}\n"
    "- {\"op\":\"DONE\", \"message\":str}\n"
)


def _truncate_obs(obs: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    acc = (obs.get("acc_tree", "") or "")
    if len(acc) > max_chars:
        acc = acc[:max_chars]
    out = dict(obs)
    out["acc_tree"] = acc
    # cap candidates list length
    cands = out.get("candidates") or []
    out["candidates"] = cands[:60]
    return out


def _safe_json(text: str) -> Dict[str, Any]:
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


class WebWorldModel:
    """LLM-based world model that predicts **delta edits** then the **next observation**.

    Input: instruction, current observation (accessibility tree), and recent history.
    Output: next observation (accessibility tree dict) and the raw DeltaPlan.
    """

    def __init__(self, cfg: WMConfig = WMConfig()):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, device_map=cfg.device_map, torch_dtype=cfg.torch_dtype)
        self.model = PeftModel.from_pretrained(base, cfg.adapter) if cfg.adapter else base
        self.model.eval()

    @torch.inference_mode()
    def predict_next(self, instruction: str, observation: Dict[str, Any], history: List[Dict[str, Any]], action: str) -> Tuple[Dict[str, Any], DeltaPlan]:
        cur = _truncate_obs(observation, self.cfg.max_obs_chars)
        hist_txt = []
        for i, h in enumerate(history[-5:]):
            hist_txt.append(f"a[{i}]={h.get('action','')}")
            hist_txt.append(f"o[{i}]={(h.get('observation',{}).get('acc_tree','') or '')[:1000]}")
        system = (
            "You are a WEB WORLD MODEL.\n"
            "Given the current accessibility tree and a single user action, predict a compact set of **edits** that update the state.\n"
            + DELTA_SCHEMA +
            "Only output JSON."
        )
        user = (
            f"Instruction: {instruction}\n"
            f"URL: {cur.get('url','')}  Title: {cur.get('title','')}\n"
            f"Current Accessibility Tree:\n{cur.get('acc_tree','')}\n\n"
            f"Recent History:\n{chr(10).join(hist_txt)}\n\n"
            f"Action: {action}\n"
            "Return JSON now."
        )
        inputs = self.tokenizer.apply_chat_template(
            [{"role":"system","content":system},{"role":"user","content":user}],
            add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                                  temperature=0.0, eos_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        j = _safe_json(text)
        delta = DeltaPlan(raw_json=j)
        next_obs = self.apply_edits(cur, delta)
        return next_obs, delta

    # -------------------- editing engine --------------------
    def apply_edits(self, obs: Dict[str, Any], delta: DeltaPlan, max_add: int = 10, max_cands: int = 60) -> Dict[str, Any]:
        state = json.loads(json.dumps(obs))  # deep copy
        id2c = {c.get("id"): c for c in (state.get("candidates") or []) if c.get("id")}
        added, removed, changed = set(), set(), set()
        edits = delta.raw_json.get("edits", []) if isinstance(delta.raw_json, dict) else []
        adds = 0
        for e in edits:
            op = (e.get("op") or "").upper()
            if op == "UPDATE_TEXT":
                cid = e.get("id"); txt = e.get("text", "")
                if cid in id2c:
                    id2c[cid]["text"] = txt
                    changed.add(cid)
            elif op == "UPDATE_ATTR":
                cid = e.get("id"); attrs = e.get("attrs", {})
                if cid in id2c and isinstance(attrs, dict):
                    id2c[cid].setdefault("attrs", {}).update(attrs)
                    changed.add(cid)
            elif op == "ADD_CANDIDATE" and adds < max_add:
                cand = e.get("candidate", {})
                if isinstance(cand, dict) and cand.get("id") and cand.get("text") is not None:
                    state.setdefault("candidates", []).append(cand)
                    id2c[cand["id"]] = cand
                    added.add(cand["id"]); adds += 1
            elif op == "REMOVE_CANDIDATE":
                cid = e.get("id")
                if cid in id2c:
                    state["candidates"] = [c for c in state["candidates"] if c.get("id") != cid]
                    id2c.pop(cid, None)
                    removed.add(cid)
            elif op == "NAVIGATE":
                state["url"] = e.get("url", state.get("url", ""))
                state["title"] = e.get("title", state.get("title", ""))
            elif op == "DONE":
                state["done"] = True
        state["candidates"] = (state.get("candidates") or [])[:max_cands]
        state["diff"] = {"added": sorted(list(added)), "removed": sorted(list(removed)), "changed": sorted(list(changed))}
        return state


# ---------------------- training ----------------------
@dataclass
class WMTrainConfig:
    lr: float = 5e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    epochs: int = 1
    bf16: bool = True


def build_wm_training_example(instruction: str, obs_t: Dict[str, Any], action_t: str, obs_tp1: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "You are a WEB WORLD MODEL. "
        "Predict a compact set of edits to transform the CURRENT accessibility tree into the NEXT one.\n\n"
        + DELTA_SCHEMA + "\n" +
        f"Instruction: {instruction}\n\n"
        f"CURRENT:\nURL: {obs_t.get('url','')}\nTitle: {obs_t.get('title','')}\n{obs_t.get('acc_tree','')[:8000]}\n\n"
        f"Action: {action_t}\n\n"
        f"TARGET_NEXT:\nURL: {obs_tp1.get('url','')}\nTitle: {obs_tp1.get('title','')}\n{obs_tp1.get('acc_tree','')[:8000]}\n\n"
        "Return the JSON edits that would produce TARGET_NEXT when applied to CURRENT."
    )
    return {"prompt": prompt}


def train_world_model(dataset: List[Dict[str, Any]], cfg: WMTrainConfig, model: WebWorldModel):
    """SFT-style training: teacher-forcing on (obs_t, action_t → delta edits for obs_{t+1}).

    dataset: list of {instruction, obs_t, action_t, obs_tp1}
    """
    from transformers import Trainer, TrainingArguments

    texts = []
    for row in dataset:
        ex = build_wm_training_example(row["instruction"], row["obs_t"], row["action_t"], row["obs_tp1"]) 
        # We expect the target to be a JSON edits block; for SFT we concatenate prompt + gold JSON
        tgt_json = json.dumps(row.get("delta") or {"edits": []}, ensure_ascii=False)
        texts.append(ex["prompt"] + tgt_json)

    class _Ds(torch.utils.data.Dataset):
        def __init__(self, xs): self.xs = xs
        def __len__(self): return len(self.xs)
        def __getitem__(self, i): return {"text": self.xs[i]}

    ds = _Ds(texts)
    tok = model.tokenizer

    def collate(batch):
        toks = tok([b["text"] for b in batch], return_tensors="pt", padding=True, truncation=True)
        toks["labels"] = toks["input_ids"].clone()
        return toks

    args = TrainingArguments(
        output_dir="./wm-sft",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        bf16=cfg.bf16,
        save_total_limit=2,
        logging_steps=50,
    )

    base = model.model
    trainer = Trainer(model=base, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    trainer.save_model("./wm-sft")
