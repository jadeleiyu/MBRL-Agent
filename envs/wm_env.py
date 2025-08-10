import json
import re
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .. import __init__  # noqa
from ..utils.prompting import (
    REWARD_SYSTEM_PROMPT,
    REWARD_USER_PROMPT,
    WORLD_MODEL_SYSTEM_PROMPT,
    format_world_model_prompt,
)


@dataclass
class WMState:
    url: str
    dom: str
    done: bool = False
    step: int = 0


####################################################
############# Web World Model #####################
####################################################

class WMAWorldModelEnv:
    """Lightweight text-only simulator using the WMA world model adapter.

    The world model is a causal LM + LoRA adapter generating JSON describing next state.
    """

    def __init__(
        self,
        base_model: str,
        adapter: str,
        max_new_tokens: int = 512,
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map=device_map
        )
        self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def predict_next(self, state: WMState, instruction: str, action: str) -> Dict:
        messages = [
            {"role": "system", "content": WORLD_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": format_world_model_prompt(state.url, state.dom, instruction, action)},
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
        gen = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        json_text = self._extract_json(gen)
        try:
            parsed = json.loads(json_text)
        except Exception:
            parsed = {"next_dom": state.dom, "next_url": state.url, "done": False, "reason": "parse_error"}
        return parsed

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"\{[\s\S]*\}", text)
        return m.group(0) if m else "{}"

####################################################
############# ORM Reward Model #####################
####################################################

class ORMRewardModel:
    """Wrapper for the WebRL ORM reward model (causal LM style).

    We prompt the model to emit JSON {"reward": <float>, "explanation": "..."} and parse the float.
    """

    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype
        )
        self.model.eval()

    @torch.inference_mode()
    def step_reward(self, instruction: str, dom: str, action: str, next_dom: str) -> float:
        messages = [
            {"role": "system", "content": REWARD_SYSTEM_PROMPT},
            {"role": "user", "content": REWARD_USER_PROMPT.format(instruction=instruction, dom=dom, action=action, next_dom=next_dom)},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(inputs, max_new_tokens=64, do_sample=False, temperature=0.0, eos_token_id=self.tokenizer.eos_token_id)
        gen = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        j = self._extract_json(gen)
        try:
            r = float(j.get("reward", 0.0))
        except Exception:
            r = self._extract_float(gen)
        # clip to [0,1]
        return max(0.0, min(1.0, float(r)))

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