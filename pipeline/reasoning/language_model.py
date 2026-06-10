"""Provider-agnostic language-model interface.

This is the Python mirror of the ``LanguageModel`` protocol Apple introduced at
WWDC 2026. The point of the abstraction is that the *perceive -> reason* glue
(:mod:`pipeline.reasoning.perceive_reason`) never names a vendor: you write the
session logic once and choose the provider at the edges. Swapping Apple's
on-device model for Claude (or a mock in tests) is a one-line change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LanguageModelResponse:
    """A normalised response, regardless of which provider produced it.

    Attributes:
        text: The model's textual output.
        provider: Identifier of the backing provider (e.g. ``"claude"``).
        usage: Optional token accounting, when the provider reports it.
    """

    text: str
    provider: str
    usage: dict = field(default_factory=dict)


class LanguageModel(ABC):
    """Interface every reasoning provider implements.

    Keep it intentionally tiny — one method — so a new backend (Gemini, a local
    GGUF model, Apple's on-device model via the Python SDK) is trivial to add.
    """

    name: str = "abstract"

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024) -> LanguageModelResponse:
        """Produce a response for ``prompt`` under an optional ``system`` prompt."""
        raise NotImplementedError


class LanguageModelSession:
    """Thin, stateful wrapper over a :class:`LanguageModel`.

    Named after Apple's ``LanguageModelSession`` so the Python and Swift sides
    read the same way. Holds the system prompt and (optionally) a running
    transcript, so callers just do ``session.respond(prompt)``.
    """

    def __init__(self, model: LanguageModel, *, system: str | None = None):
        self.model = model
        self.system = system
        self.transcript: list[tuple[str, str]] = []

    def respond(self, prompt: str, *, max_tokens: int = 1024) -> LanguageModelResponse:
        """Send ``prompt`` to the underlying model and record the exchange."""
        resp = self.model.generate(prompt, system=self.system, max_tokens=max_tokens)
        self.transcript.append((prompt, resp.text))
        return resp
