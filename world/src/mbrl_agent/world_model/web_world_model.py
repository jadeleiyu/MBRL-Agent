# ==============================================
# File: web_world_model.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
import os
import subprocess
import random
import time

from openai import OpenAI

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
Note: Even if the web page does not change, you must still output the complete next_acc_tree !
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
        # We do not rely on any tokenizer/HF assets; prompts are constructed from messages directly.
        self.tokenizer = None
        # Support OpenAI-compatible gateways (e.g., vLLM/OpenAI router) with Chat Completions
        self.vllm_urls = getattr(args, "vllm_urls", [])
        self.api_key = getattr(args, "api_key", os.environ.get("WM_API_KEY", "web_world_model"))
        self.use_chat_completions = getattr(args, "use_chat_completions", True)

        self.clients = []
        for vllm_url in self.vllm_urls:
            client = OpenAI(
                api_key=self.api_key,
                base_url=vllm_url,  # should end with /v1
            )
            self.clients.append(client)

        self.max_new_tokens = args.wm_max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.sys_prompt = WM_SYS_PROMPT
        self.user_prompt_template = WM_USR_PROMPT_TEMPLATE
        self.answer_parse_pattern_cot = "Web state changes\n"
        self.answer_parse_pattern_obs = "Next page accessibility tree"

        # Retry configs
        self.retry_max_attempts = int(getattr(args, "retry_max_attempts", 1_000_000))
        self.retry_sleep_s = float(getattr(args, "retry_sleep_s", 0.01))

    def _do_with_retries(self, fn):
        last_exc = None
        delay = self.retry_sleep_s
        for _ in range(self.retry_max_attempts):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                time.sleep(delay)
                delay = delay * 2  # exponential backoff
        # Exhausted retries
        if last_exc is not None:
            raise last_exc
        return None

    def _format_messages_as_prompt(self, messages: list[dict]) -> str:
        """Fallback prompt builder when tokenizer is unavailable.
        Format:
        <system> ... \n<user> ...
        """
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            parts.append(f"<{role}>\n{content}\n")
        return "\n".join(parts)

    # wm_output = world_model.step(action, dreamed_traj, task)
    def step(self, batch_actions, batch_dreamed_trajs, tasks):
        batch_inputs = []
        batch_messages = []
        for action, dreamed_traj, task in zip(batch_actions, batch_dreamed_trajs, tasks):
            prev_action = dreamed_traj[-1]['action']
            messages = [
                {'role': 'system', 'content': self.sys_prompt},
                {'role': 'user', 'content': self.user_prompt_template.format(
                    usr_obj=task['objective'],
                    curr_acc_tree=dreamed_traj[-1]['next_observation'],
                    curr_url='None',
                    prev_action=prev_action,
                    curr_action=action
                )},
            ]
            batch_messages.append(messages)

            prompt = self._format_messages_as_prompt(messages)
            batch_inputs.append(prompt)

        client = random.choice(self.clients)

        batch_delta, batch_next_obs = [], []
        if self.use_chat_completions:
            # Chat Completions currently do not support true batching; call per item
            for i, messages in enumerate(batch_messages):
                try:
                    out = self._do_with_retries(lambda: client.chat.completions.create(
                        model=self.wm_model_name,
                        messages=messages,
                        max_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    ))
                    response = out.choices[0].message.content if out.choices else ""
                    
                    delta, next_obs = self.parse_output(response or "")
                    
                    # 调试：显示解析结果
                    print(f"Parsed - delta: {delta[:50]}{'...' if len(delta) > 50 else ''}")
                    print(f"Parsed - next_obs: {next_obs[:50]}{'...' if len(next_obs) > 50 else ''}")
                    
                    batch_delta.append(delta)
                    batch_next_obs.append(next_obs)
                except Exception as e:
                    print(f"Debug: World model API call failed: {e}")
                    raise e
        else:
            batch_outputs = self._do_with_retries(lambda: client.completions.create(
                model=self.wm_model_name,
                prompt=batch_inputs,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ))
            for choice in batch_outputs.choices:
                response = choice.text
                delta, next_obs = self.parse_output(response)
                batch_delta.append(delta)
                batch_next_obs.append(next_obs)

        return batch_delta, batch_next_obs

    def judge_completion(self, objective: str, current_obs: str) -> bool:
        """Ask the model to judge if the task is completed based on objective and observation.

        The judge must output a structured decision with reasoning:
        [REASON]\n<why the current observation satisfies (or not) the objective>\n
        [DECISION]\nCOMPLETED: YES|NO
        """
        judge_system = (
            "You are a precise evaluator for web navigation tasks."
            " Decide strictly whether the user's objective has been accomplished in the CURRENT OBSERVATION."
            " Only answer YES if the objective is explicitly and verifiably satisfied; otherwise answer NO."
            " Provide a brief rationale first, then a structured decision in the specified format."
            "\n\nOutput format (must follow exactly):\n"
            "[REASON]\n<concise rationale>\n\n[DECISION]\nCOMPLETED: YES|NO"
        )
        user_text = (
            "TASK OBJECTIVE:\n"
            f"{objective}\n\n"
            "CURRENT OBSERVATION (flattened accessibility text):\n"
            f"{current_obs}\n\n"
            "Decide if the objective is accomplished now based ONLY on the current observation."
        )

        def _parse_judge_decision(text: str) -> Optional[bool]:
            try:
                if not text:
                    return None
                # Prefer structured block
                lower = text.lower()
                # Extract decision block
                if "[decision]" in lower:
                    # Get the substring after [DECISION]
                    idx = lower.rfind("[decision]")
                    segment = lower[idx:]
                    # Look for line like "completed: yes" or "completed: no"
                    import re as _re
                    m = _re.search(r"completed\s*:\s*(yes|no)", segment, flags=_re.IGNORECASE)
                    if m:
                        return m.group(1).strip().lower() == "yes"
                # Fallback: any decisive yes/no occurrence
                if "yes" in lower and "no" not in lower:
                    return True
                if "no" in lower and "yes" not in lower:
                    return False
                if any(k in lower for k in ("completed", "success")) and "not" not in lower:
                    # very weak signal, avoid using if both yes/no present or none
                    return True
                return None
            except Exception:
                return None

        client = random.choice(self.clients)
        if self.use_chat_completions:
            def _judge_decisive_chat():
                out = client.chat.completions.create(
                    model=self.wm_model_name,
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": user_text},
                    ],
                    max_tokens=1024,
                    temperature=0.0,
                    top_p=1.0,
                )
                if out and getattr(out, 'choices', None) and len(out.choices) > 0:
                    content = getattr(out.choices[0].message, 'content', None)
                    decision = _parse_judge_decision(content or "")
                else:
                    raise ValueError("Empty judge choices from world model")
                if decision is None:
                    raise ValueError("Judge result not decisive")
                return decision
            decision_bool = self._do_with_retries(_judge_decisive_chat)
        else:
            messages = [
                {"role": "system", "content": judge_system},
                {"role": "user", "content": user_text},
            ]
            prompt = self._format_messages_as_prompt(messages)
            def _judge_decisive_text():
                out = client.completions.create(
                    model=self.wm_model_name,
                    prompt=[prompt],
                    max_tokens=1024,
                    temperature=0.0,
                    top_p=1.0,
                )
                if out and getattr(out, 'choices', None) and len(out.choices) > 0:
                    raw = getattr(out.choices[0], 'text', None)
                    decision = _parse_judge_decision(raw or "")
                else:
                    raise ValueError("Empty judge choices from world model")
                if decision is None:
                    raise ValueError("Judge result not decisive")
                return decision
            decision_bool = self._do_with_retries(_judge_decisive_text)

        return bool(decision_bool)
    
    def parse_output(self, response):
        try:
            # Look for the expected format: [Web state changes] ... [Next page accessibility tree] ...
            if "[Web state changes]" in response and "[Next page accessibility tree]" in response:
                # Split by the patterns
                parts = response.split("[Web state changes]")
                if len(parts) > 1:
                    delta_and_obs = parts[1]
                    if "[Next page accessibility tree]" in delta_and_obs:
                        delta, obs = delta_and_obs.split("[Next page accessibility tree]", 1)
                        return delta.strip(), obs.strip()
            
            # Fallback: return the response as observation
            return "None", response.strip()
        except Exception as e:
            return "None", "None"

        