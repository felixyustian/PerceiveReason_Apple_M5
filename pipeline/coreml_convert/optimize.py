"""Post-training weight compression for Core ML packages.

Smaller weights mean a smaller app download, less memory pressure and — on the
Neural Engine — often faster inference. This module wraps
``coremltools.optimize.coreml`` so the CLI and tests have one entry point.

Two strategies are supported:

* **Linear quantisation** (``int8``): each weight tensor is mapped to integers
  with a per-tensor scale. Simple, robust, ~2x smaller than fp16.
* **Palettisation** (``nbits``): weights are clustered (k-means) into a small
  lookup table. At 4 bits you get ~4x compression versus fp16, with more
  accuracy risk — always re-check accuracy on device before shipping.
"""
from __future__ import annotations

from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
    palettize_weights,
)

from ..config import OptimizeConfig


def compress(
    mlmodel: ct.models.MLModel,
    output_path: str | Path | None = None,
    *,
    config: OptimizeConfig | None = None,
) -> ct.models.MLModel:
    """Compress weights of a converted Core ML model.

    Args:
        mlmodel: An ``MLModel`` (e.g. returned by ``convert_*``).
        output_path: If given, save the compressed package here.
        config: Compression knobs; see :class:`OptimizeConfig`. When
            ``nbits_palette`` is set, palettisation is used; otherwise linear
            quantisation.

    Returns:
        The compressed ``MLModel``.
    """
    config = config or OptimizeConfig()

    if config.nbits_palette is not None:
        op_cfg = OpPalettizerConfig(mode="kmeans", nbits=config.nbits_palette)
        opt = OptimizationConfig(global_config=op_cfg)
        compressed = palettize_weights(mlmodel, config=opt)
    else:
        op_cfg = OpLinearQuantizerConfig(mode=config.mode, dtype=config.dtype)
        opt = OptimizationConfig(global_config=op_cfg)
        compressed = linear_quantize_weights(mlmodel, config=opt)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compressed.save(str(output_path))
    return compressed
