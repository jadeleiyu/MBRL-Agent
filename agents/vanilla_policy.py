# ==============================================
# File: vanilla_policy.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

# ---------- Prompt templates ----------
SYSTEM_PROMPT = (
    "You are a skilled web-browsing agent. "
    "Given the user's high-level instruction, the current page (accessibility tree), "
    "and recent (action, observation) history, you will decide the NEXT single operation.\n\n"
    "Rules:\n"
    "- Output exactly ONE action line using the API: do(...), exit(...), go_backward(), or go_forward().\n"
    "- Keep to one line. No code fences. No extra commentary.\n"
    "- If you need to wait for page load, use do(action=\"Wait\").\n"
    "- Never use the browser address bar.\n"
)


def build_user_prompt(instruction: str, obs: Dict[str, Any], history: List[Dict[str, Any]], T: int = 5) -> str:
    # history items: {"action": str, "observation": {"acc_tree": str, ...}}
    h = history[-T:]
    hist_txt = []
    for i, item in enumerate(h):
        hist_txt.append(f"- a[{i}]: {item.get('action','')}")
        ob = item.get("observation", {})
        acc = ob.get("acc_tree", "")[:2000]
        hist_txt.append(f"  o[{i}]: {acc}")
    acc_tree = (obs.get("acc_tree", "") or "")[:4000]
    url = obs.get("url", "")
    return (
        f"Instruction: {instruction}\n"
        f"URL: {url}\n\n"
        f"# Recent History (most recent last)\n" + ("\n".join(hist_txt) if hist_txt else "(none)") + "\n\n"
        f"# Current Page (accessibility tree)\n{acc_tree}\n\n"
        "Return the next one-line action now."
    )


@dataclass
class VanillaConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device_map: str = "auto"
    torch_dtype = torch.bfloat16
    max_new_tokens: int = 64
    temperature: float = 0.3
    top_p: float = 0.9


class VanillaPolicy:
    """Vanilla web agent policy (decoder-only LLM with value head).

    .act(...) returns a **single** one-line action string using the allowed API.
    The value head is available via .value(...) for critic/advantage calculations.
    """

    def __init__(self, cfg: VanillaConfig = VanillaConfig()):
        self.cfg = cfg
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.model_name, device_map=cfg.device_map, torch_dtype=cfg.torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model.eval()

    @torch.inference_mode()
    def act(
        self,
        instruction: str,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
        T: int = 5,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(instruction, observation, history, T=T)},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.pretrained_model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self._pick(temperature, self.cfg.temperature),
            top_p=self._pick(top_p, self.cfg.top_p),
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        return self._extract_action(gen)

    @torch.inference_mode()
    def value(self, instruction: str, observation: Dict[str, Any], history: List[Dict[str, Any]], T: int = 5) -> float:
        prompt = build_user_prompt(instruction, observation, history, T=T)
        toks = self.tokenizer([prompt], return_tensors="pt").to(self.model.pretrained_model.device)
        out = self.model(**toks, return_dict=True)
        v = out.value[:, -1].squeeze().float().cpu().item()
        return float(v)

    @staticmethod
    def _pick(x, d):
        return d if x is None else x

    @staticmethod
    def _extract_action(text: str) -> str:
        # Find first do(...), exit(...), go_backward(), go_forward()
        import re
        pat = re.compile(r"(do\([^\)]*\)|exit\([^\)]*\)|go_backward\(\)|go_forward\(\))", re.I)
        m = pat.search(text)
        return m.group(1) if m else "do(action=\"Wait\")"
