"""Optimisation tests — verify compression actually shrinks the package."""
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("coremltools")

from pipeline.coreml_convert import convert, optimize, inspect_model
from pipeline.config import ConvertConfig, OptimizeConfig


def test_int8_quantisation_reduces_size():
    with tempfile.TemporaryDirectory() as d:
        fp16 = Path(d) / "fp16.mlpackage"
        int8 = Path(d) / "int8.mlpackage"
        mlmodel = convert.convert_torchvision(
            "mobilenet_v3_small", fp16, config=ConvertConfig())
        optimize.compress(mlmodel, int8, config=OptimizeConfig(dtype="int8"))
        ratio = inspect_model.compression_ratio(fp16, int8)
        # int8 weights should be meaningfully smaller than the fp16 baseline.
        assert ratio > 1.3
