# ==============================================
# File: critic.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import os
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams


SYS_PROMPT_CRITIC = """You are an expert in evaluating and guiding a web navigation agent. Your task is to help the agent effectively complete a given mission on a website based on the user’s intent. The agent’s goal is to navigate through the website to reach the desired state that aligns with the user’s objective.

You will analyze the agent's prediction about the next state of the webpage after each action and determine whether the agent is successfully progressing towards the task goal. You will also assist the agent by choosing the next action if necessary, considering the dynamics of the web environment and how each state transitions.

Key Points:
1. Understand the intent:
- Identify the user’s goal (e.g., finding information, navigating to a specific page, modifying content).
- Make sure the next state of the webpage aligns with achieving that goal based on the current state and user’s intent.

2. Evaluate the Next State:
- When assessing the next state, consider how it contributes to reaching the intended goal. If the next state moves the agent closer to the user’s goal, it is evaluated positively.
- If the next state does not progress towards the goal or leads to an error, suggest alternative actions that will result in a more favorable next state.

3. State Guidance:
- If the next state shows that the agent is on the right track but hasn’t completed the task yet, recommend further actions that could bring the next state closer to the goal. Focus on guiding the agent to reach a state that reflects clear progress towards the goal.

4. Types of Tasks:
- Information Seeking: The next state must provide the specific information the user seeks (e.g., product price, reviews). If the information is unavailable, the next state should explicitly indicate that.
- Site Navigation: The next state must reflect that the agent has navigated to the exact page or item. Check if the state includes content based on the user’s intent.
- Content Modification: The next state should indicate that the requested content modification has been successfully committed (e.g., form submission, comment posting).
- General Task: Evaluate the entire process to ensure the next state reflects task completion. Stop actions should only be issued when the objective is met.

5. Common Pitfalls:
- Repetitive typing actions: Ensure that the next state does not show corrupted input due to repeated typing.
- Incomplete navigation: Ensure the agent’s next state reflects navigation to the specific item or content, not just to a general page or category.

Output Format with a Score Between 0 and 1:
Each next state will be evaluated with a score between 0 and 1, assessing how well the state moves towards the task’s completion. This score provides nuanced feedback on the state’s effectiveness.
0: The next state is a failure or leads away from the task.
Values closer to 0 (e.g., 0.1, 0.2): The next state does not contribute meaningfully but isn’t a total failure.
0.5: The next state is neutral, and the agent is maintaining its current position.
Values closer to 1 (e.g., 0.7, 0.8): The next state is helpful and moves the agent closer to the task goal.
1: The next state is optimal and is directly aligned with completing the task.

Response Format:
1. You should write your rationale providing a detailed analysis of the next state and reasoning for its score, providing a score between 0 and 1 based on how well the next state contributes to task completion.

Output Format: [Rationale] <your thought> [Score] <a value between 0 and 1>"""


USER_PROMPT_CRITIC = """OBJECTIVE: {usr_obj}
PREVIOUS ACTION: {prev_action}
CURRENT OBSERVATION: {curr_acc_tree}
CURRENT ACTION: {curr_action}
NEXT STATE PREDICTION: {next_state_pred}
"""

ASSISTANT_PROMPT_WM_SFT = """[Rationale]\n{rationale}\n\n[Next State]\nThe expected next web page accessibility tree is:\n\n{next_acc_tree}"""


# Primary labeled patterns (same idea as before)
PRIMARY = [
    re.compile(r'(?is)\bscore\b[^0-9%]{0,10}\[?\s*(?P<val>[01](?:[.,]\d+)?|\.\d+)\s*(?:%|\s*/\s*1(?:\.0+)?)?\s*\]?', re.I),
    re.compile(r'(?is)"score"\s*:\s*(?P<val>[01](?:[.,]\d+)?|\.\d+)\b'),
    re.compile(r'(?is)\b(final|overall|aggregate)\b[^0-9%]{0,20}\bscore\b[^0-9%]{0,10}(?P<val>[01](?:[.,]\d+)?|\.\d+)\b'),
    re.compile(r'(?is)\bscore\b[^0-9%]{0,10}(?P<pct>\d{1,3}(?:[.,]\d+)?)\s*%'),
    re.compile(r'(?is)\bscore\b[^0-9%]{0,10}(?P<frac>[01](?:[.,]\d+)?|\.\d+)\s*/\s*1(?:\.0+)?'),
]

KEYWORDS = re.compile(r'\b(score|final|overall|aggregate|rating)\b', re.I)

def _to_float(s: str) -> float:
    s = s.strip().rstrip('.,)]}')
    s = s.replace(',', '.')
    if s.startswith('.'):
        s = '0' + s
    return float(s)

def _near_keyword(text: str, idx: int, window: int = 40) -> bool:
    a = max(0, idx - window)
    b = min(len(text), idx + window)
    return bool(KEYWORDS.search(text[a:b]))

def _fallback_single_float(text: str) -> Optional[float]:
    cands = []  # (value, start_idx)

    # Percentages like "82%" (normalize to 0.82)
    for m in re.finditer(r'(?is)\b(\d{1,3}(?:[.,]\d+)?)\s*%', text):
        v = _to_float(m.group(1)) / 100.0
        if 0.0 <= v <= 1.0:
            cands.append((v, m.start()))

    # Fractions like "82/100", "8/10", "0.82/1"
    for m in re.finditer(r'(?is)\b(\d{1,3}(?:[.,]\d+)?)\s*/\s*(100|10|1(?:\.0+)?)\b', text):
        num = _to_float(m.group(1))
        den = float(m.group(2).replace('.0', '')) if '.' in m.group(2) else float(m.group(2))
        if den != 0:
            v = num / den
            if 0.0 <= v <= 1.0:
                cands.append((v, m.start()))

    # Plain decimals like "0.82" or ".82" or "1.0"
    for m in re.finditer(r'(?<!\d)(?:0?\.\d+|1(?:\.0+)?)\b', text):
        # Heuristic: skip list items like "1." at line starts
        line_start = text.rfind('\n', 0, m.start()) + 1
        if m.group(0) in ('1.', '1') and re.match(r'^\s*\d+\.\s', text[line_start:m.start()+2]):
            continue
        v = _to_float(m.group(0))
        if 0.0 <= v <= 1.0:
            cands.append((v, m.start()))

    if not cands:
        # Last-ditch: any number near the word "score"
        m = re.search(r'(?is)\bscore\b.{0,40}?([01](?:[.,]\d+)?|\.\d+)\b', text)
        if m:
            return max(0.0, min(1.0, _to_float(m.group(1))))
        return None

    # If they all collapse to one numeric value, return it
    uniq = {}
    for v, i in cands:
        uniq.setdefault(round(v, 6), []).append(i)
    if len(uniq) == 1:
        return list(uniq.keys())[0]

    # Otherwise, prefer candidates near "score/final/overall"
    scored = []
    for v, i in cands:
        near = _near_keyword(text, i)
        scored.append((1 if near else 0, len(text) - i, v))  # near first, then latest
    scored.sort()
    chosen = scored[-1][2]
    return max(0.0, min(1.0, chosen))


def extract_judge_score(text: str) -> Optional[float]:
    # 1) Try primary labeled patterns
    for pat in PRIMARY:
        for m in pat.finditer(text):
            gd = m.groupdict()
            if 'pct' in gd and gd['pct']:
                return max(0.0, min(1.0, _to_float(gd['pct']) / 100.0))
            if 'frac' in gd and gd['frac']:
                return max(0.0, min(1.0, _to_float(gd['frac'])))
            if 'val' in gd and gd['val']:
                return max(0.0, min(1.0, _to_float(gd['val'])))
    # 2) Fallback tolerant scan
    return _fallback_single_float(text)

class Critic:

    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.critic_model_name)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx_critic
        self.lm = LLM(
            model=args.critic_model_name, 
            trust_remote_code=True,
            tensor_parallel_size=args.critic_tensor_parallel_size,
            dtype=args.torch_dtype
        )
        self.sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        self.sys_prompt = SYS_PROMPT_CRITIC
        self.user_prompt_template = USER_PROMPT_CRITIC
        self.answer_parse_pattern_cot = "[Rationale]"
        self.answer_parse_pattern_score = "[Score]"

    def parse_output(self, output_text):
        try:
            cot_and_score = output_text.split(self.answer_parse_pattern_cot)[1]
            cot, score = cot_and_score.split(self.answer_parse_pattern_score)
            score = float(score.strip())
            return score
        except Exception as e:
            try:
                score = extract_judge_score(output_text)
                return score
            except Exception as e:
                print('error in parsing critic score')
                print(f"critic output text: {output_text}")
                print('\n\n')
                # print(f'output parsing error: {e} \n')
                return -1.


    def score(self, dreamed_traj, objective):
        batch_inputs = []
        for t in range(1, len(dreamed_traj)):
            messages = [
                {'role':'system', 'content': self.sys_prompt},
                {'role': 'user', 'content': self.user_prompt_template.format(
                    usr_obj=objective,
                    curr_acc_tree=dreamed_traj[t-1]['next_observation'],
                    prev_action=dreamed_traj[t-1]['action'],
                    curr_action=dreamed_traj[t]['action'],
                    next_state_pred=ASSISTANT_PROMPT_WM_SFT.format(
                        rationale=dreamed_traj[t]['wm_cot'],
                        next_acc_tree=dreamed_traj[t]['next_observation']
                    )
                )},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # adds the assistant turn to complete
            )
            batch_inputs.append(prompt)

        batch_outputs = self.lm.generate(
            batch_inputs,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        batch_values = [-1.] + [self.parse_output(output.outputs[0].text) for output in batch_outputs]
        return batch_values

        


