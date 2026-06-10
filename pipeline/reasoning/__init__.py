"""The *Reason* stage: turn structured vision output into a calibrated assessment.

The design mirrors Apple's WWDC 2026 Foundation Models update, where a single
``LanguageModel`` protocol lets you swap the on-device Apple model for Claude or
Gemini without touching downstream code. Here:

* :class:`language_model.LanguageModel`        — the provider interface.
* :class:`language_model.LanguageModelSession` — the call surface (mirrors
  Apple's ``LanguageModelSession``).
* :mod:`providers.claude_provider`             — Anthropic Claude implementation.
* :mod:`providers.mock_provider`               — deterministic offline provider
  (so CI and demos run without network or API keys).
* :func:`perceive_reason.reason_over_detections` — the orchestration glue.
"""
from .language_model import (  # noqa: F401
    LanguageModel,
    LanguageModelResponse,
    LanguageModelSession,
)

__all__ = ["LanguageModel", "LanguageModelResponse", "LanguageModelSession"]
