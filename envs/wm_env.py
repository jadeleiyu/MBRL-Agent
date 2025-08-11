# ==============================================
# File: mbrl/envs/wm_env.py
# ==============================================
from __future__ import annotations
import json
import re
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel


# ---------------------------------------------------------------------
# World model that maps (encoded_state, instruction, action) -> next_encoded_state
# ---------------------------------------------------------------------

_ENCODED_SCHEMA_SPEC = (
    "You predict NEXT ENCODED STATE as strict JSON with keys:\n"
    "{\n"
    "  \"url\": str,\n"
    "  \"title\": str,\n"
    "  \"instruction\": str,\n"
    "  \"summary\": str,\n"
    "  \"candidates\": [\n"
    "    {\"id\": str, \"tag\": str, \"role\": str, \"text\": str, \n"
    "     \"attrs\": {id?, name?, class?, placeholder?, aria-label?, data-testid?, role?},\n"
    "     \"selector\": str,\n"
    "     \"pos\": {\"depth\": int, \"order\": int, \"visible\": bool},\n"
    "     \"neighborhood\": str}\n"
    "  ],\n"
    "  \"diff\": {\"added\": [str], \"removed\": [str], \"changed\": [str]},\n"
    "  \"done\": bool\n"
    "}\n"
    "Rules: Keep IDs stable when elements persist. Update only changed items. Keep candidates concise (<=50)."
)


class WebWorldModel:
    """LM-driven world model that **inputs and outputs encoded JSON states**.

    Under the hood uses a causal LLM (e.g., WMA adapter) prompted to emit the next encoded state
    directly. No post-hoc HTML encoder is called here.
    """

    def __init__(
        self,
        base_model: str = "openai/gpt-oss-20b",
        # adapter: str = "LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-WM-webarena-16k-adapter",
        max_new_tokens: int = 768,
        device_map: str = "auto",
        torch_dtype="auto",
        # max_state_chars: int = 12000,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map=device_map
        )
        # self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # self.max_state_chars = max_state_chars

    @torch.inference_mode()
    def predict_next(self, encoded_state: Dict[str, Any], instruction: str, action: str) -> Dict[str, Any]:
        cur_url = encoded_state.get("url", "")
        cur_json = json.dumps(_truncate_state(encoded_state), ensure_ascii=False)

        system = (
            "You are a WEB WORLD MODEL for browser agents. Given the CURRENT ENCODED STATE (JSON),\n"
            "the user's instruction, and ONE next action, predict the NEXT ENCODED STATE.\n"
            + _ENCODED_SCHEMA_SPEC + "\n"
            "Maintain ID stability for unchanged elements; update URL/title/summary if navigation happens.\n"
            "Return ONLY valid JSON — no extra text."
        )
        user = (
            f"Instruction: {instruction}\n"
            f"Action: {action}\n"
            f"Current Encoded State (JSON):\n{cur_json}\n"
            f"Return the NEXT ENCODED STATE as JSON now."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        j = _safe_json(text)

        # Minimal normalization
        j.setdefault("instruction", instruction)
        j.setdefault("url", cur_url)
        j.setdefault("candidates", [])
        j.setdefault("summary", "")
        j.setdefault("title", "")
        j.setdefault("diff", {"added": [], "removed": [], "changed": []})
        j.setdefault("done", False)
        if isinstance(j.get("candidates"), list) and len(j["candidates"]) > 60:
            j["candidates"] = j["candidates"][:60]
        return j


# ---------------------------------------------------------------------
# ORM Reward model (encoded-state aware)
# ---------------------------------------------------------------------

def _render_state_for_reward(state: Dict[str, Any], max_cands: int = 30) -> str:
    lines = [
        f"URL: {state.get('url','')}",
        f"Title: {state.get('title','')}",
        f"Summary: {state.get('summary','')}",
        "Candidates:",
    ]
    for c in (state.get("candidates") or [])[:max_cands]:
        line = (
            f"- id={c.get('id')} tag={c.get('tag')} role={c.get('role','')} sel={c.get('selector','')}\n"
            f"  text={str(c.get('text',''))[:80]} | neigh={str(c.get('neighborhood',''))[:100]}"
        )
        lines.append(line)
    return "\n".join(lines)


class ORMRewardModel:
    """Wrapper around the WebRL ORM model that scores progress **from encoded states**.

    step_reward(instruction, curr_state_encoded, action, next_state_encoded) -> float in [0,1]
    """

    def __init__(
            self, 
            model_name: str = "openai/gpt-oss-20b", 
            device_map: str = "auto", 
            torch_dtype='auto'
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype
        )
        self.model.eval()

    @torch.inference_mode()
    def step_reward(
        self,
        instruction: str,
        curr_state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any],
        max_new_tokens: int = 96,
    ) -> float:
        system = (
            "You are a reward model for web agents. Given a user instruction, a summarized representation of the current web state, \n"
            "the action taken by the agent, and the summarized representation of the next web state, output strict JSON: {\"reward\": <float in [0,1]>, \"explanation\": <string>}\n"
            "Guidelines: 1.0 = task accomplished; ~0.5 = meaningful progress; 0.0 = regress/irrelevant."
        )
        user = (
            f"Instruction: {instruction}\n\n"
            f"CURRENT STATE:\n{_render_state_for_reward(curr_state)}\n\n"
            f"Action: {action}\n\n"
            f"NEXT STATE:\n{_render_state_for_reward(next_state)}\n\n"
            "Return JSON now."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        j = _safe_json(text)
        try:
            r = float(j.get("reward", 0.0))
        except Exception:
            r = _extract_float(text)
        r = max(0.0, min(1.0, float(r)))
        return r


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_json(text: str) -> Dict[str, Any]:
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def _extract_float(text: str) -> float:
    m = re.search(r"-?\d+\.?\d*", text)
    return float(m.group(0)) if m else 0.0


def _truncate_state(state: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """Trim overly large encoded states for prompting stability."""
    js = json.dumps(state, ensure_ascii=False)
    if len(js) <= max_chars:
        return state
    # Keep header and first 30 candidates
    trimmed = {k: state.get(k) for k in ("url", "title", "instruction", "summary", "diff", "done")}
    trimmed["candidates"] = (state.get("candidates", []) or [])[:30]
    return trimmed


