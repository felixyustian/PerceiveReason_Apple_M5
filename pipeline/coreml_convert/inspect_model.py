"""Inspect a Core ML package without the on-device runtime.

Because Core ML predictions require macOS, CI on Linux verifies *structure*
instead of *behaviour*: that the package exists, has the expected I/O, is an
``mlprogram``, and that compression actually shrank it. That is enough to catch
the overwhelming majority of conversion regressions before they reach a device.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

import coremltools as ct


@dataclass
class ModelSummary:
    """A structured, JSON-serialisable description of an mlpackage."""

    path: str
    model_type: str
    inputs: list[str]
    outputs: list[str]
    size_mb: float

    def as_dict(self) -> dict:
        return asdict(self)


def _dir_size_mb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / 1e6, 3)


def summarize(mlpackage_path: str | Path) -> ModelSummary:
    """Return a :class:`ModelSummary` for a saved ``.mlpackage``.

    Works on any OS: it reads the model *spec* (a protobuf), never running it.
    """
    path = Path(mlpackage_path)
    spec = ct.utils.load_spec(str(path))
    return ModelSummary(
        path=str(path),
        model_type=spec.WhichOneof("Type") or "unknown",
        inputs=[i.name for i in spec.description.input],
        outputs=[o.name for o in spec.description.output],
        size_mb=_dir_size_mb(path),
    )


def compression_ratio(original: str | Path, compressed: str | Path) -> float:
    """Return ``original_size / compressed_size`` (>1 means it got smaller)."""
    o = _dir_size_mb(Path(original))
    c = _dir_size_mb(Path(compressed))
    return round(o / c, 3) if c else float("inf")
