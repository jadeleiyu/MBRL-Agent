# ==============================================
# File: critic.py
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import os

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


class Critic:

    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.critic_model_name)

        self.lm = LLM(
            model=args.critic_model_name, 
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
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
            print('error in parsing critic score')
            print(cot_and_score.split(self.answer_parse_pattern_score))
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

        


