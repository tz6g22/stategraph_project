"""LLM client stub.

No actual provider clients are initialized or called in this framework stage.
"""

from __future__ import annotations


class LLMClient:
    """Placeholder client interface for future model-backed components."""

    def complete(self, prompt: str) -> str:
        """Raise until an explicit LLM integration is implemented."""
        _ = prompt
        raise NotImplementedError("LLM calls are not implemented in this skeleton.")

