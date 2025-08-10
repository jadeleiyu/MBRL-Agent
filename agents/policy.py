from typing import List

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


class PolicyWithValue:
    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype=torch.bfloat16, use_8bit: bool = False):
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if use_8bit:
            kwargs.update(dict(load_in_8bit=True))
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model.eval()

    def act(self, observations: List[str], max_new_tokens: int = 128) -> List[str]:
        with torch.inference_mode():
            inputs = self.tokenizer(observations, return_tensors="pt", padding=True, truncation=True).to(self.model.pretrained_model.device)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
            gens = self.tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return gens
