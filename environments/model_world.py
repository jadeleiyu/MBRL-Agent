"""
Model-driven Virtual World Environment
Uses an LLM to simulate a dynamic virtual world
"""

from typing import Dict, Any, Optional
from vllm import LLM, SamplingParams


class ModelWorldEnvironment:
    """
    Virtual world environment powered by an LLM
    The model acts as the world simulator, generating environment states
    """

    def __init__(self, model_path: str):
        """
        Initialize model-driven world environment

        Args:
            model_path: Path to the world model
        """
        self.model_path = model_path
        self.llm = LLM(model=model_path, tensor_parallel_size=1)
        self.sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512
        )

        # World state
        self.current_state = {
            "description": "A virtual world waiting for interaction",
            "available_actions": ["explore", "interact", "observe"],
            "objects": [],
            "characters": [],
            "locations": ["starting_point"]
        }

    def get_world_state(self, agent_action: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current world state, optionally updated by agent action

        Args:
            agent_action: Agent's action that affects the world

        Returns:
            Current world state
        """
        if agent_action:
            # Use model to simulate world response to agent action
            world_prompt = f"""
You are simulating a virtual world. The current world state is:
{self.current_state}

An agent just performed this action: "{agent_action}"

Describe how the world responds and updates. Include:
1. Updated world description
2. Available actions for the agent
3. Any new objects, characters, or locations
4. Any changes to the environment

Respond in JSON format:
{{
    "description": "updated world description",
    "available_actions": ["list", "of", "actions"],
    "objects": ["list", "of", "objects"],
    "characters": ["list", "of", "characters"],
    "locations": ["list", "of", "locations"]
}}
"""

            try:
                outputs = self.llm.generate(world_prompt, self.sampling_params)
                world_response = outputs[0].outputs[0].text.strip()

                # Parse model response
                updated_state = self._parse_world_response(world_response)
                if updated_state:
                    self.current_state.update(updated_state)
            except Exception as e:
                print(f"⚠ World model error: {e}")
                # Fallback: simple state update
                self.current_state["description"] = f"World responded to: {agent_action}"

        return self.current_state.copy()

    def _parse_world_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse model's world response

        Args:
            response: Model's response text

        Returns:
            Parsed world state or None if parsing fails
        """
        import json
        import re

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except Exception as e:
            print(f"⚠ Failed to parse world response: {e}")

        return None

    def reset(self):
        """Reset world to initial state"""
        self.current_state = {
            "description": "A virtual world waiting for interaction",
            "available_actions": ["explore", "interact", "observe"],
            "objects": [],
            "characters": [],
            "locations": ["starting_point"]
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get world state summary"""
        return {
            "description": self.current_state.get("description", ""),
            "available_actions_count": len(self.current_state.get("available_actions", [])),
            "objects_count": len(self.current_state.get("objects", [])),
            "characters_count": len(self.current_state.get("characters", [])),
            "locations_count": len(self.current_state.get("locations", []))
        }