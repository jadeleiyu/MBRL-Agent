import json
import re
from pathlib import Path
from typing import Any, TypedDict

from mbrl.envs.browser_env import ActionParsingError

# from envs.browser_env.env_config import URL_MAPPINGS
URL_MAPPINGS = {}

# from browser_env.utils import StateInfo
# from llms import lm_config
# from llms.tokenizers import Tokenizer
# from llms.utils import APIInput


class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def get_lm_api_input(
        self, intro: str, examples: list[tuple[str, str]], current: str
    ):
        """Return the require format for an API"""
        messages = [{"role": "system", "content": intro}]
        for (x, y) in examples:
            messages.append(
                {
                    "role": "system",
                    "name": "example_user",
                    "content": x,
                }
            )
            messages.append(
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": y,
                }
            )
        messages.append({"role": "user", "content": current})
        message = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # adds the assistant turn to complete
        )
        return message

    def construct(
        self,
        trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ):
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        tokenizer,
        max_obs_length=1920,
    ):
        super().__init__(instruction_path, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]
        self.max_obs_length = max_obs_length

    def construct(
        self,
        trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ):
        # dream_trajectory[i] = {
        #         "step": t,
        #         "action": action,
        #         "next_observation": obs_next,
        #         "wm_cot": cot,
        #         "done": terminated,
        # }
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info = trajectory[-1]  # type: ignore[assignment]

        if "observation" in state_info:
            obs = state_info["observation"][self.obs_modality]
        else:
            obs = state_info["next_observation"]
        max_obs_length = self.max_obs_length
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        if "info" in state_info:
            page = state_info["info"]["page"]
            url = page.url
            previous_action_str = meta_data["action_history"][-1]
            current = template.format(
                objective=intent,
                url=self.map_url_to_real(url),
                observation=obs,
                previous_action=previous_action_str,
            )
        else:
            if trajectory:
                previous_action_str = trajectory[-1]['action']
            else:
                previous_action_str = 'None'
            current = template.format(
                objective=intent,
                url='',
                observation=obs,
                previous_action=previous_action_str,
            )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )



