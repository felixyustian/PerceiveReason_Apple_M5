"""Offline, deterministic provider.

Used by the test-suite and by ``--provider mock`` in the CLI so the full
perceive->reason flow can run in CI with no network and no API key. It applies a
tiny rule over the detections embedded in the prompt, which is enough to prove
the wiring works end to end.
"""
from __future__ import annotations

import re

from ..language_model import LanguageModel, LanguageModelResponse


class MockProvider(LanguageModel):
    """A rules-only stand-in for a real language model."""

    name = "mock"

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024) -> LanguageModelResponse:
        # Pull "label (0.87)" style pairs out of the prompt the orchestrator built.
        pairs = re.findall(r"([A-Za-z][A-Za-z _]*?)\s*\(([01]\.\d+)\)", prompt)
        if not pairs:
            text = "No detections were provided; insufficient evidence to assess."
        else:
            label, conf = max(pairs, key=lambda p: float(p[1]))
            conf_f = float(conf)
            band = ("high" if conf_f >= 0.8 else
                    "moderate" if conf_f >= 0.5 else "low")
            text = (
                f"Primary observation: {label.strip()} (confidence {conf_f:.2f}, "
                f"{band}). "
                + ("Evidence is strong enough to act on."
                   if band == "high" else
                   "Recommend a second look or human review before acting.")
            )
        return LanguageModelResponse(text=text, provider=self.name,
                                     usage={"input_tokens": len(prompt.split())})
