from __future__ import annotations

import re
from typing import Any

from rllm.agents.webarena_agent import WebArenaAgent


class WorldModelWebAgent(WebArenaAgent):
    """Variant of :class:`WebArenaAgent` tailored for the world-model environment.

    The simulated environment returns a flattened accessibility tree string
    (`axtree_txt`) instead of a structured object. This agent keeps the rest of
    the prompting logic identical but bypasses BrowserGym-specific processing and
    returns raw bracket-style actions (e.g., ``click [12]``) so that the world
    model receives the exact command tokens it was trained on.
    """

    def _preproc_obs(self, obs: dict) -> dict:
        if "goal_object" in obs and obs["goal_object"]:
            goal_entry = obs["goal_object"][0]
            if isinstance(goal_entry, dict):
                self.objective = goal_entry.get("text", self.objective)
        axtree_txt = obs.get("axtree_txt", "")
        self.id2node = {}
        return {
            "chat_messages": obs.get("chat_messages", []),
            "goal_object": obs.get("goal_object", []),
            "last_action": obs.get("last_action"),
            "last_action_error": obs.get("last_action_error"),
            "open_pages_urls": obs.get("open_pages_urls", []),
            "open_pages_titles": obs.get("open_pages_titles", []),
            "active_page_index": obs.get("active_page_index", 0),
            "axtree_txt": axtree_txt,
        }

    def _parse_model_response(self, response: str) -> tuple[dict[str, Any], str]:
        parts = [
            "INTERACTION HISTORY SUMMARY",
            "OBSERVATION DESCRIPTION",
            "REASON",
            "ACTION",
            "OBSERVATION HIGHLIGHT",
        ]
        result: dict[str, Any] = {}
        current_part = None

        for line in response.split("\n"):
            header = line.strip(": ")
            if header in parts:
                current_part = header
                result[current_part] = []
            elif current_part:
                result[current_part].append(line.strip())

        for key in result:
            result[key] = "\n".join(result[key]).strip()

        action_str = result.get("ACTION", "").strip()
        if not action_str:
            fallback_pattern = re.compile(
                r"(click\s*\[\d+\]"
                r"|type\s*\[\d+\]\s*\[.*?\]\s*\[(?:0|1)\]"
                r"|hover\s*\[\d+\]"
                r"|scroll\s*\[(?:up|down)\]"
                r"|press\s*\[[^\]]+\]"
                r"|goto\s*\[.*?\]"
                r"|go_back"
                r"|go_forward"
                r"|new_tab"
                r"|close_tab"
                r"|tab_focus\s*\[\d+\]"
                r"|stop\s*\[.*?\])",
                flags=re.IGNORECASE | re.DOTALL,
            )
            match = fallback_pattern.search(response)
            if match:
                action_str = match.group(1).strip()

        if not action_str:
            action_str = "stop [Unable to parse action]"

        return result, action_str
