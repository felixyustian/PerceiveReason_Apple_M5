"""Anthropic Claude provider.

This is the "cloud" implementation of the :class:`LanguageModel` protocol — the
exact role Claude plays in Apple's WWDC 2026 model-abstraction layer, where an
app prototypes on the free on-device model and routes harder queries to Claude
through the same session API.

Auth: reads ``ANTHROPIC_API_KEY`` from the environment. Never hard-code keys.
The ``anthropic`` SDK is imported lazily so the rest of the package (and the
offline tests) work without it installed.
"""
from __future__ import annotations

import os

from ..language_model import LanguageModel, LanguageModelResponse


class ClaudeProvider(LanguageModel):
    """Reasoning backed by the Anthropic Messages API.

    Args:
        model: Model id. Defaults to ``claude-sonnet-4-6`` (fast/cheap, good for
            structured reasoning over vision output). Use ``claude-opus-4-8``
            for the hardest judgement calls.
        api_key: Overrides ``ANTHROPIC_API_KEY`` if provided.
    """

    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # created on first use

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or pass api_key=..., "
                "or use MockProvider for offline runs."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'anthropic' package is required for ClaudeProvider. "
                "Install it with: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024) -> LanguageModelResponse:
        self._ensure_client()
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in msg.content if block.type == "text")
        usage = {}
        if getattr(msg, "usage", None) is not None:
            usage = {
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            }
        return LanguageModelResponse(text=text, provider=self.name, usage=usage)
