# ==============================================
# File: web_world_model.py
# ==============================================
from __future__ import annotations

import os
import random
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Sequence

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


@dataclass
class WMConfig:
    wm_model_name: str
    vllm_urls: Sequence[str]
    api_key: str
    use_chat_completions: bool
    wm_max_new_tokens: int
    temperature: float
    top_p: float


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
        self.answer_parse_pattern_obs = "[Next page accessibility tree]\n"

    @staticmethod
    def _format_messages_as_prompt(messages):
        out = []
        for item in messages:
            role = item["role"]
            content = item["content"]
            out.append(f"<|role|>{role}\n<|content|>\n{content}")
        return "\n".join(out)

    @staticmethod
    def _parse_response_delta_and_obs(response_text):
        """Parse the world model output into delta and next observation."""
        if not response_text:
            return "None", "None"

        if "[Web state changes]" in response_text and "[Next page accessibility tree]" in response_text:
            try:
                delta = response_text.split("[Web state changes]", 1)[1]
                if "[Next page accessibility tree]" in delta:
                    delta, obs = delta.split("[Next page accessibility tree]", 1)
                    return delta.strip(), obs.strip()
            except Exception:
                pass

        # Fallback: return the response as observation
        return "None", response_text.strip()

    def _do_with_retries(self, fn, max_retries=100, base_delay=1.0):
        retry = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                retry += 1
                if retry > max_retries:
                    raise exc
                sleep_time = base_delay * (2**(retry - 1))
                time.sleep(sleep_time + random.uniform(0, 0.5 * sleep_time))

    def _format_messages(self, messages):
        formatted = []
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], str):
                content = msg["content"]
            else:
                content = ""
            formatted.append({"role": msg.get("role", "user"), "content": content})
        return formatted

    def _query_client_chat(self, client, messages):
        formatted = self._format_messages(messages)

        def _call():
            resp = client.chat.completions.create(
                model=self.wm_model_name,
                messages=formatted,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if not resp or not getattr(resp, "choices", None):
                raise ValueError("Empty world model response")
            return getattr(resp.choices[0].message, "content", None)

        return self._do_with_retries(_call)

    def _query_client_text(self, client, messages):
        prompt = self._format_messages_as_prompt(messages)

        def _call():
            resp = client.completions.create(
                model=self.wm_model_name,
                prompt=[prompt],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if not resp or not getattr(resp, "choices", None):
                raise ValueError("Empty world model response")
            return getattr(resp.choices[0], "text", None)

        return self._do_with_retries(_call)

    def step(self, actions, dreamed_trajs, tasks):
        outputs = []
        next_obs = []

        for action, dream_traj, task in zip(actions, dreamed_trajs, tasks):
            prev_obs = dream_traj[-1]["next_observation"] if dream_traj else task.get("axtree_txt", "")
            prev_action = dream_traj[-1]["action"] if dream_traj else "None"
            curr_url = task.get("url", "")
            user_prompt = self.user_prompt_template.format(
                usr_obj=task.get("objective", ""),
                curr_acc_tree=prev_obs,
                curr_url=curr_url,
                prev_action=prev_action,
                curr_action=action or "None",
            )
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_prompt},
            ]

            client = random.choice(self.clients)
            if self.use_chat_completions:
                response = self._query_client_chat(client, messages)
            else:
                response = self._query_client_text(client, messages)

            delta, obs = self._parse_response_delta_and_obs(response or "")
            outputs.append(delta)
            next_obs.append(obs)

        return outputs, next_obs

    def judge_completion(self, objective: str, current_obs: str) -> bool:
        """Judge whether the task is completed based on the current observation."""
        if not self.clients:
            raise ValueError("World model clients not initialized")

        judge_system = (
            "You will be given a TASK OBJECTIVE and the CURRENT OBSERVATION."
            " Decide if the objective has been completed given ONLY the information in the CURRENT OBSERVATION."
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
                lower = text.lower()
                if "[decision]" in lower:
                    idx = lower.rfind("[decision]")
                    segment = lower[idx:]
                    import re as _re

                    match = _re.search(r"completed\s*:\s*(yes|no)", segment, flags=_re.IGNORECASE)
                    if match:
                        return match.group(1).strip().lower() == "yes"
                if "yes" in lower and "no" not in lower:
                    return True
                if "no" in lower and "yes" not in lower:
                    return False
                if any(k in lower for k in ("completed", "success")) and "not" not in lower:
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
                if out and getattr(out, "choices", None) and len(out.choices) > 0:
                    content = getattr(out.choices[0].message, "content", None)
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
                if out and getattr(out, "choices", None) and len(out.choices) > 0:
                    raw = getattr(out.choices[0], "text", None)
                    decision = _parse_judge_decision(raw or "")
                else:
                    raise ValueError("Empty judge choices from world model")
                if decision is None:
                    raise ValueError("Judge result not decisive")
                return decision

            decision_bool = self._do_with_retries(_judge_decisive_text)

        return bool(decision_bool)

    @staticmethod
    def parse_output(response):
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
        except Exception:
            return "None", "None"

    @staticmethod
    def merge_lora(base_model_path, lora_path, output_path, python_exec="python"):
        cmd = [
            python_exec,
            "-m",
            "swift.deploy.merge_lora",
            "--base_model",
            base_model_path,
            "--lora_model",
            lora_path,
            "--output_dir",
            output_path,
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"merge_lora failed: {stderr.decode('utf-8')}")
        return stdout.decode("utf-8").strip()


__all__ = ["WebWorldModel", "WMConfig"]
