"""Orchestration: vision detections -> language-model assessment.

This is the seam between the two stages. It is deliberately provider-agnostic:
hand it any :class:`LanguageModel` (Claude in production, Mock in CI, an Apple
on-device model via the Python SDK if you have one) and it behaves the same.
"""
from __future__ import annotations

from ..config import ReasoningConfig
from .language_model import LanguageModel, LanguageModelResponse, LanguageModelSession
from .prompts import Detection, build_prompt


def reason_over_detections(
    detections: list[Detection],
    task: str,
    model: LanguageModel,
    *,
    config: ReasoningConfig | None = None,
) -> LanguageModelResponse:
    """Run the Reason stage over a list of vision detections.

    Args:
        detections: Structured output from the Perceive (Core ML) stage.
        task: A one-line description of what the assessment is for, e.g.
            "Assess PPE compliance for this worker."
        model: Any provider implementing :class:`LanguageModel`.
        config: Reasoning knobs (system prompt, max tokens).

    Returns:
        The provider's normalised :class:`LanguageModelResponse`.
    """
    config = config or ReasoningConfig()
    session = LanguageModelSession(model, system=config.system_prompt)
    prompt = build_prompt(detections, task)
    return session.respond(prompt, max_tokens=config.max_tokens)
