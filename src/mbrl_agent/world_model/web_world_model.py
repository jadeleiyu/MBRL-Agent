# ==============================================
# File: web_world_model.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
import os
import subprocess
import random

from openai import OpenAI
from transformers import AutoTokenizer

# os.environ["VLLM_CONFIGURE_LOGGING"] = "0"   # set this *before* importing vllm
# os.environ["VLLM_LOGGING_LEVEL"]    = "WARNING"  # or "ERROR"

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

Given the information above, you should first perform reasoning to predict expected changes on the current web page's accessibility tree,
and then generate the resulting next web page's accessibility tree based on your predicted web page changes.
Generate your answer in the following format: 
[Web state changes]
changes

[Next page accessibility tree]
next_acc_tree

where ``changes`` are the predicted web page changes, and ``next_acc_tree`` is your predicted next page accessibility tree.
For next_acc_tree, you MUST generate a valid accessibility tree based on your predicted changes, do NOT output summary descriptions of how the next_acc_tree will change.
If the full predicted next_acc_tree is too long, you should output a pruned tree by removing unimportant elements that the web agent will unlikely use in subsequent action steps.
Meanwhile, you should NOT omit key unchanged elements that the web agent might interact with in future time steps.
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
        self.wm_model_name = args.wm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.wm_model_name)
        self.vllm_urls = args.vllm_urls
        self.clients = []
        for vllm_url in self.vllm_urls:
            client = OpenAI(
                api_key="web_world_model", # just a place holder
                base_url=vllm_url,
            )
            self.clients.append(client)
        self.max_new_tokens = args.wm_max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.sys_prompt = WM_SYS_PROMPT
        self.user_prompt_template = WM_USR_PROMPT_TEMPLATE
        self.answer_parse_pattern_cot = "Web state changes\n"
        self.answer_parse_pattern_obs = "Next page accessibility tree"

    # wm_output = world_model.step(action, dreamed_traj, task)
    def step(self, batch_actions, batch_dreamed_trajs, tasks):
        batch_inputs = []
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
            batch_inputs.append(prompt)

        client = random.choice(self.clients)
        batch_outputs = client.completions.create(
            model=self.wm_model_name,
            prompt=batch_inputs,   
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        batch_delta, batch_next_obs = [], []
        for choice in batch_outputs.choices:
            response = choice.text
            delta, next_obs = self.parse_output(response)
            batch_delta.append(delta)
            batch_next_obs.append(next_obs)

        return batch_delta, batch_next_obs
    
    def parse_output(self, response):
        try:
            model_answer = response.split("assistantfinal")[-1]
            delta_and_obs = model_answer.split(self.answer_parse_pattern_cot)[-1]
            delta, obs = delta_and_obs.split(self.answer_parse_pattern_obs)
            return delta.strip(), obs.strip()
        except Exception as e:
            return "None", "None"

        