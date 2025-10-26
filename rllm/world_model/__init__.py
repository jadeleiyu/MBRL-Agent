"""
Utilities for learned world-model components used by RLLM agents.

Currently exposes the WebWorldModel class that simulates WebArena style
observations. The implementation is migrated from the legacy MBRL-Agent
repository and kept self-contained so that the new rllm package can import
it without relying on the old codebase.
"""

from rllm.world_model.web_world_model import WebWorldModel

__all__ = ["WebWorldModel"]
