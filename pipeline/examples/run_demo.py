"""End-to-end demo: convert -> optimize -> inspect -> reason.

Runs fully offline by default (random-weight model + MockProvider), so it works
in CI on Linux. Pass ``provider="claude"`` (with ANTHROPIC_API_KEY set) to route
the Reason stage to Claude instead.

    python -m pipeline demo               # offline
    python -m pipeline demo --provider claude
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def main(provider: str = "mock") -> int:
    from ..coreml_convert import convert, optimize, inspect_model
    from ..config import ConvertConfig, OptimizeConfig
    from ..reasoning.perceive_reason import reason_over_detections
    from ..reasoning.prompts import Detection
    from ..reasoning.providers import ClaudeProvider, MockProvider

    workdir = Path(tempfile.mkdtemp(prefix="perceive_reason_demo_"))
    fp16 = workdir / "model_fp16.mlpackage"
    int8 = workdir / "model_int8.mlpackage"

    print("== Perceive: convert MobileNetV3 -> Core ML ==")
    mlmodel = convert.convert_torchvision(
        "mobilenet_v3_small", fp16, config=ConvertConfig())
    print(inspect_model.summarize(fp16).as_dict())

    print("\n== Optimize: int8 weight quantisation ==")
    optimize.compress(mlmodel, int8, config=OptimizeConfig(dtype="int8"))
    print(f"compression ratio: {inspect_model.compression_ratio(fp16, int8)}x")

    print("\n== Reason: assess detections ==")
    detections = [
        Detection("hard_hat", 0.94),
        Detection("safety_vest", 0.88),
        Detection("person", 0.99),
    ]
    model = MockProvider() if provider == "mock" else ClaudeProvider()
    resp = reason_over_detections(
        detections, "Assess PPE compliance for this worker.", model)
    print(f"[{resp.provider}] {resp.text}")

    print(f"\nArtifacts in: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
