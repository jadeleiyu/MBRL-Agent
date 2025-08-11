# ==============================================
# File: policy.py
# ==============================================
from __future__ import annotations
import re
from typing import List, Dict

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ..envs.encoder import render_state_for_policy
from ..utils.prompting import WEBRL_SYSTEM_PROMPT



def _build_user_prompt(instruction: str, action_history: List[str], encoded_state: Dict[str, any]) -> str:
    state_block = render_state_for_policy(encoded_state)
    hist = "\n".join(f"- {a}" for a in action_history[-10:]) if action_history else "(none)"
    return (
        f"Instruction: {instruction}\n\n"
        f"# History\n{hist}\n\n"
        f"# Current Page (encoded)\n{state_block}\n\n"
        f"Return exactly ONE line next action using the allowed API."
    )


def _first_action_line(text: str) -> str:
    # Extract the first valid action line among do(...), exit(...), go_backward(), go_forward()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    pattern = re.compile(r"^(do\([^\)]*\)|exit\([^\)]*\)|go_backward\(\)|go_forward\(\))", re.I)
    for l in lines:
        m = pattern.match(l)
        if m:
            return m.group(1)
    # fallback: try to find a do(...) anywhere
    m = re.search(r"do\([^\)]*\)|exit\([^\)]*\)|go_backward\(\)|go_forward\(\)", text, re.I)
    return m.group(0) if m else "do(action=\"Wait\")"


class PolicyAgent:
    """Policy agent that consumes (instruction, action_history, encoded_state) and emits one action line.

    Backed by a causal LLM with a value head (compatible with TRL). You can also use just the
    `.pretrained_model` for pure generation.
    """

    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype=torch.bfloat16, use_8bit: bool = False):
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if use_8bit:
            kwargs.update(dict(load_in_8bit=True))
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model.eval()

    @torch.inference_mode()
    def act(
        self,
        instruction: str,
        action_history: List[str],
        encoded_state: Dict[str, any],
        max_new_tokens: int = 64,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        messages = [
            {"role": "system", "content": WEBRL_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(instruction, action_history, encoded_state)},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.pretrained_model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        return _first_action_line(gen)

