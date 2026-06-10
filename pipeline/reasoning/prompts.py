"""Prompt construction for the Reason stage.

Isolating prompt text here (rather than f-strings scattered through the
orchestrator) keeps the wording reviewable and lets you tune it without
touching control flow — the same reason you keep SQL out of business logic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """One structured output from the vision model.

    Attributes:
        label: Class name, e.g. ``"hard_hat"`` or ``"document"``.
        confidence: Model confidence in ``[0, 1]``.
        box: Optional ``(x, y, w, h)`` in normalised image coordinates.
    """

    label: str
    confidence: float
    box: tuple[float, float, float, float] | None = None


def build_prompt(detections: list[Detection], task: str) -> str:
    """Render detections + a task description into a single user prompt.

    The ``label (confidence)`` shape is intentional: it is both human-readable
    and trivially parseable, which is what the offline MockProvider relies on.
    """
    if not detections:
        observations = "(no detections returned by the vision model)"
    else:
        observations = "\n".join(
            f"- {d.label} ({d.confidence:.2f})"
            + (f" at box={tuple(round(v, 3) for v in d.box)}" if d.box else "")
            for d in detections
        )
    return (
        f"Task: {task}\n\n"
        f"Machine-vision detections:\n{observations}\n\n"
        "Using only the detections above, give a brief assessment, state your "
        "confidence, and recommend the next action. If the evidence is weak or "
        "ambiguous, say so explicitly rather than guessing."
    )
