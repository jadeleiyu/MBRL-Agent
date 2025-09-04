# ==============================================
# File: web_world_model.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"   # set this *before* importing vllm
os.environ["VLLM_LOGGING_LEVEL"]    = "WARNING"  # or "ERROR"
from vllm import LLM, SamplingParams

# This world model works over **accessibility tree** observations.
# It predicts small **delta edits** first, then applies them to form the next observation.

WM_SYS_PROMPT = """You are an intelligent agent that predicts next state from given current action in a web environment, with your own logical reasoning. 

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The previous action: This is the action you just performed in the previous step. It may be helpful to track your progress. 
The current action: This is the current action that you performed to achieve the user's objective in the current web page's accessibility tree.
The format of previous actions can fall into several categories:
Page Operation Actions:

```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content]```: Use this to type the content into the field with id. By default, the 'Enter' key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.
```hover [id]```: Hover over an element with id.
```press [key_comb]```: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed)

Completion Action:
```stop [answer]```: Done when you believe the task is complete.

Follow the following rules for reasoning on next state prediction.
1. Please generate your answer starting with Let's think step by step, with your logical REASONING.
2. When you generate your logical reasoning, you must identify and mention only the changed parts of the [accessibility tree] for the next state based on the given current action. 
3. And then, you must generate a complete accessibility tree of the next web page based on the changed parts you identified.
4. Generate the next web page accessibility tree prediction in the correct format. Start with a "[Next State] The expected next web page accessibility tree is:" phrase.
"""

WM_USR_PROMPT_TEMPLATE = """User objective: {usr_obj}
Current web page accessibility tree: {curr_acc_tree}
Current web page URL: {curr_url}
Previous action: {prev_action}
Current action: {curr_action}
"""


class WebWorldModel:
    """LLM-based world model that predicts next web page observation.
    Input: instruction, current observation (accessibility tree), and recent history.
    Output: next observation (accessibility tree dict) and the raw DeltaPlan.
    """

    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.wm_model_name)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx_wm
        self.lm = LLM(
            model=args.wm_model_name, 
            trust_remote_code=True,
            tensor_parallel_size=args.wm_tensor_parallel_size,
            dtype=args.torch_dtype,
        )
        self.sampling_params = SamplingParams(
            max_tokens=args.wm_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        self.sys_prompt = WM_SYS_PROMPT
        self.user_prompt_template = WM_USR_PROMPT_TEMPLATE
        self.answer_parse_pattern_cot = "[Rationale]\n"
        self.answer_parse_pattern_obs = "The expected next web page accessibility tree is:"

    # wm_output = world_model.step(action, dreamed_traj, task)
    def step(self, batch_actions, batch_dreamed_trajs, tasks):
        batch_input_prompts = []
        for action, dreamed_traj, task in zip(batch_actions, batch_dreamed_trajs, tasks):
            prev_action = dreamed_traj[-1]['action']
            messages = [
                {'role':'system', 'content': self.sys_prompt},
                {'role': 'user', 'content': self.user_prompt_template.format(
                    usr_obj=task['objective'],
                    curr_acc_tree=dreamed_traj[-1]['next_observation'],
                    curr_url='None',
                    prev_action=prev_action,
                    curr_action=action
                )},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # adds the assistant turn to complete
            )
            batch_input_prompts.append(prompt)

        outputs = self.lm.generate(
            batch_input_prompts,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        batch_cot, batch_next_obs = [], []
        for out in outputs:
            response = out.outputs[0].text
            cot, next_obs = self.parse_output(response)
            batch_cot.append(cot)
            batch_next_obs.append(next_obs)

        return batch_cot, batch_next_obs
    
    def parse_output(self, response):
        try:
            cot_and_obs = response.split(self.answer_parse_pattern_cot)[-1]
            cot, obs = cot_and_obs.split(self.answer_parse_pattern_obs)
            return cot, obs
        except Exception as e:
            return "None", "None"

        