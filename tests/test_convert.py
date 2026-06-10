"""Conversion tests — verify the Perceive stage produces a valid package.

These run on Linux: they check *structure*, not on-device behaviour.
"""
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("coremltools")

from pipeline.coreml_convert import convert, inspect_model
from pipeline.config import ConvertConfig


def test_convert_torchvision_produces_valid_mlprogram():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.mlpackage"
        convert.convert_torchvision("mobilenet_v3_small", out,
                                    config=ConvertConfig(image_size=224))
        summary = inspect_model.summarize(out)
        assert summary.model_type == "mlProgram"
        assert "image" in summary.inputs
        assert "logits" in summary.outputs
        assert summary.size_mb > 0


def test_unknown_arch_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            convert.convert_torchvision("not_a_real_model", Path(d) / "x.mlpackage")
