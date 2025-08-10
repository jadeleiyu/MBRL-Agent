# ==============================================
# File: mbrl_agent/agents/mpc_value.py
# ==============================================
from __future__ import annotations
import json
import re
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


VALUE_SYSTEM_PROMPT = (
    """You are a value model for web agents operating in text-only web environments.\n"
    "Given an instruction and a CANDIDATE TRAJECTORY (a sequence of steps with state/action/next_state),\n"
    "estimate how successful the trajectory is at completing the instruction.\n"
    "Return strict JSON with keys: {\"score\": <float in [0,1]>, \"explanation\": <string>}.\n"
    "Scoring guidelines: 1.0 = fully completes the task with minimal unnecessary steps; 0.0 = irrelevant or harmful steps.\n"
    "Consider correctness, progress toward the goal, and efficiency across the whole trajectory."""
)


def _trajectory_block(trajectory: List[Dict]) -> str:
    lines = []
    for i, st in enumerate(trajectory):
        url = st.get("url", "")
        dom = st.get("dom", "").strip()
        dom = dom[:2000]  # truncate to keep context small
        action = st.get("action", "")
        next_dom = st.get("next_dom", "").strip()[:1200]
        lines.append(
            f"Step {i+1}:\nURL: {url}\nDOM (trunc):\n{dom}\nACTION: {action}\nNEXT_DOM (trunc):\n{next_dom}\n---"
        )
    return "\n".join(lines)


class MPCValueModel:
    """Wrapper for the LangAGI-Lab value adapter on Meta-Llama-3.1-8B-Instruct.

    We prompt it to return a JSON object with {score, explanation}.
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter: str = (
            "LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-value-model-16k-qlora-adapter-v2"
        ),
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
        max_new_tokens: int = 128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, device_map=device_map, torch_dtype=torch_dtype
        )
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def score(self, instruction: str, trajectory: List[Dict]) -> float:
        messages = [
            {"role": "system", "content": VALUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Instruction: {instruction}\n\nCANDIDATE TRAJECTORY (states+actions+next_states):\n"
                    f"{_trajectory_block(trajectory)}\n\nReturn JSON now."
                ),
            },
        ]
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
        j = self._extract_json(text)
        try:
            s = float(j.get("score", 0.0))
        except Exception:
            s = self._extract_float(text)
        # clamp
        return max(0.0, min(1.0, float(s)))

    @staticmethod
    def _extract_json(text: str) -> Dict:
        try:
            m = re.search(r"\{[\s\S]*\}", text)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}

    @staticmethod
    def _extract_float(text: str) -> float:
        m = re.search(r"-?\d+\.?\d*", text)
        return float(m.group(0)) if m else 0.0
