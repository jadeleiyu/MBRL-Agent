from __future__ import annotations

import importlib
import json
import logging
import re
from types import SimpleNamespace
from typing import Any, Optional

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)


class WorldModelWebEnv(BaseEnv):
    """Simulated WebArena environment backed by a learned world model.

    The environment mirrors the old MBRL-Agent implementation but is adapted to the
    new rllm 0.2 framework. A world model predicts the next accessibility-tree
    observation without requiring a live browser session, enabling offline training.

    Expected ``env_args`` (via :meth:`from_dict`):
        wm_class_path: Import path to the world model class (defaults to
            ``rllm.world_model.web_world_model.WebWorldModel``).
        wm_kwargs: Dict passed to the world model constructor. Alternatively, the
            individual fields below may be provided and will be folded into
            ``wm_kwargs``:
            wm_model_name, vllm_urls, api_key, use_chat_completions,
            wm_max_new_tokens, temperature, top_p.
        task: Dict describing the episode objective with fields like
            ``objective`` and ``axtree_txt``.
        max_steps: Optional[int] to terminate the episode after N steps.
    """

    def __init__(self, wm_class_path: str, wm_kwargs: Optional[dict] = None, task: Optional[dict] = None, max_steps: int = 3):
        super().__init__()
        self.wm_class_path = wm_class_path
        self.wm_kwargs = wm_kwargs or {}
        self.task = task or {}
        self.max_steps = max_steps

        self.world_model = self._instantiate_world_model()
        self._episode_steps: int = 0
        self._dreamed_traj: list[dict[str, Any]] = []

        self._initial_obs = self._build_initial_obs()

    def _build_initial_obs(self) -> dict[str, Any]:
        """Construct the base observation returned on reset."""
        return {
            "goal_object": [{"text": self.task.get("objective", "")}],
            "axtree_txt": self.task.get("axtree_txt", ""),
            "chat_messages": [],
            "last_action": None,
            "last_action_error": None,
            "open_pages_urls": [],
            "open_pages_titles": [],
            "active_page_index": 0,
        }

    def _instantiate_world_model(self):
        try:
            module_path, class_name = self.wm_class_path.rsplit(".", 1)
        except ValueError as exc:  # noqa: BLE001
            raise ValueError(f"Invalid wm_class_path: {self.wm_class_path}") from exc

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        args = self.wm_kwargs
        if isinstance(args, dict):
            args = SimpleNamespace(**args)
        return cls(args)

    def reset(self) -> tuple[dict, dict]:
        self._episode_steps = 0
        self._dreamed_traj = [
            {
                "step": 0,
                "action": "None",
                "is_action_valid": True,
                "next_observation": self._initial_obs.get("axtree_txt", ""),
                "wm_cot": "None",
                "agent_response": "None",
            }
        ]
        return self._initial_obs.copy(), {}

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        action_str = "" if action is None else str(action)
        has_stop_signal, stop_answer = self._extract_stop_answer(action_str)

        if has_stop_signal:
            logger.debug("STOP action detected; skipping world model rollout.")
            next_obs_text = self._dreamed_traj[-1]["next_observation"] if self._dreamed_traj else self._initial_obs.get("axtree_txt", "")
            wm_cot = ""
        else:
            actions = [action_str]
            try:
                logger.debug("World model step action=%s", action_str or "(empty)")
                wm_cot_list, next_obs_list = self.world_model.step(actions, [self._dreamed_traj], [self.task])
            except Exception as exc:  # noqa: BLE001
                logger.exception("World model step failed")
                raise exc

            next_obs_text = next_obs_list[0] if next_obs_list else ""
            wm_cot = wm_cot_list[0] if wm_cot_list else ""

        if not self._looks_like_structured_tree(next_obs_text) and self._dreamed_traj:
            # Fallback to last known observation to avoid drifting into free-form text
            next_obs_text = self._dreamed_traj[-1].get("next_observation", next_obs_text)

        self._episode_steps += 1
        self._dreamed_traj.append(
            {
                "step": self._episode_steps,
                "action": action_str,
                "is_action_valid": True,
                "next_observation": next_obs_text,
                "wm_cot": wm_cot,
                "agent_response": None,
            }
        )

        if has_stop_signal:
            is_task_completed = self._check_task_completion(next_obs_text)
            reward = 1.0 if is_task_completed else 0.0
            done = True
        else:
            is_task_completed = False
            reward = 0.0
            done = self._episode_steps >= self.max_steps

        obs = {
            "goal_object": [{"text": self.task.get("objective", "")}],
            "axtree_txt": next_obs_text,
            "chat_messages": [],
            "last_action": action_str,
            "last_action_error": None,
            "open_pages_urls": [],
            "open_pages_titles": [],
            "active_page_index": 0,
        }

        info = {
            "wm_cot": wm_cot,
            "has_stop_signal": has_stop_signal,
            "stop_answer": stop_answer if has_stop_signal else None,
            "is_task_completed": bool(is_task_completed),
            "reward_reason": (
                "stop+correct=1.0" if (has_stop_signal and is_task_completed)
                else "stop+incorrect=0.0" if has_stop_signal
                else ("timeout=0.0" if done else "in_progress=0.0")
            ),
        }
        return obs, reward, done, info

    @staticmethod
    def _looks_like_structured_tree(text: str) -> bool:
        if not text:
            return False
        return ("[" in text and "]" in text) or ("role" in text.lower())

    @staticmethod
    def from_dict(info: dict) -> "WorldModelWebEnv":
        wm_class_path = info.get("wm_class_path") or "rllm.world_model.web_world_model.WebWorldModel"
        wm_kwargs = info.get("wm_kwargs", {})
        if isinstance(wm_kwargs, str):
            try:
                wm_kwargs = json.loads(wm_kwargs)
            except Exception:
                logger.warning("Failed to parse wm_kwargs JSON string; using raw value.")
                wm_kwargs = {}

        if not wm_kwargs:
            wm_kwargs = {
                "wm_model_name": info.get("wm_model_name", "ms-w7gsz7wq"),
                "vllm_urls": info.get("vllm_urls", []),
                "api_key": info.get("api_key"),
                "use_chat_completions": info.get("use_chat_completions", True),
                "wm_max_new_tokens": info.get("wm_max_new_tokens", 8192),
                "temperature": info.get("temperature", 0.7),
                "top_p": info.get("top_p", 0.9),
            }

        task = info.get("task", {})
        if not task:
            objective = info.get("objective", "")
            axtree_txt = info.get("axtree_txt", "")
            if objective or axtree_txt:
                task = {"objective": objective, "axtree_txt": axtree_txt}

        max_steps = int(info.get("max_steps", 3))
        return WorldModelWebEnv(wm_class_path=wm_class_path, wm_kwargs=wm_kwargs, task=task, max_steps=max_steps)

    @staticmethod
    def _extract_stop_answer(action: str) -> tuple[bool, str]:
        match = re.search(r"stop\s*\[(.+?)\]", action, re.IGNORECASE | re.DOTALL)
        if match:
            return True, match.group(1).strip()
        return False, ""

    def _check_task_completion(self, current_obs: str) -> bool:
        objective = self.task.get("objective", "")
        try:
            return bool(self.world_model.judge_completion(objective=objective, current_obs=current_obs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("World model judgment failed: %s", exc)
            return False
