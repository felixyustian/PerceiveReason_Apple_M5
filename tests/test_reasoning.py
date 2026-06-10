"""Reasoning tests — verify the provider abstraction and orchestration.

No network or API key required: these use the offline MockProvider, which also
proves the LanguageModel interface is honoured.
"""
from pipeline.reasoning.language_model import LanguageModel, LanguageModelSession
from pipeline.reasoning.perceive_reason import reason_over_detections
from pipeline.reasoning.prompts import Detection, build_prompt
from pipeline.reasoning.providers import MockProvider


def test_mock_provider_is_a_language_model():
    assert isinstance(MockProvider(), LanguageModel)


def test_prompt_includes_detections():
    prompt = build_prompt([Detection("hard_hat", 0.94)], "Assess PPE.")
    assert "hard_hat" in prompt and "0.94" in prompt and "Assess PPE." in prompt


def test_empty_detections_are_handled():
    resp = reason_over_detections([], "Assess the scene.", MockProvider())
    assert "insufficient" in resp.text.lower()


def test_reason_over_detections_picks_highest_confidence():
    dets = [Detection("person", 0.99), Detection("hard_hat", 0.40)]
    resp = reason_over_detections(dets, "Assess PPE compliance.", MockProvider())
    assert resp.provider == "mock"
    assert "person" in resp.text  # highest-confidence detection surfaced


def test_session_records_transcript():
    session = LanguageModelSession(MockProvider(), system="sys")
    session.respond("hard_hat (0.94)")
    assert len(session.transcript) == 1
