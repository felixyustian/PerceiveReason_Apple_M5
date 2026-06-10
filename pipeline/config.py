"""Shared configuration objects for the pipeline.

Keeping the tunable knobs in one place makes the CLI, the tests and the example
scripts agree on defaults, and documents — in code — every decision a reviewer
might ask about (deployment target, compute units, quantisation mode, model id).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --- Perception (Core ML) -------------------------------------------------- #

ComputeUnits = Literal["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"]
QuantMode = Literal["linear", "linear_symmetric"]


@dataclass(frozen=True)
class ConvertConfig:
    """Settings for the PyTorch -> Core ML conversion step.

    Attributes:
        image_size: Square input resolution the model expects (H == W).
        deployment_target: Minimum Apple OS. ``iOS18`` is a safe floor that all
            current Apple-silicon devices satisfy; bump to ``iOS26``/``iOS27``
            only if you depend on newer Core ML ops.
        compute_units: Which engines Core ML may schedule onto. ``ALL`` lets the
            runtime pick the Neural Engine when it is the fastest option.
        as_image_input: When True the model takes a Core ML ``ImageType`` (a
            pixel buffer the camera can hand over directly). When False it takes
            a float ``TensorType`` — useful when preprocessing happens upstream.
    """

    image_size: int = 224
    deployment_target: str = "iOS18"
    compute_units: ComputeUnits = "ALL"
    as_image_input: bool = True


@dataclass(frozen=True)
class OptimizeConfig:
    """Settings for post-training weight compression.

    Attributes:
        mode: ``linear_symmetric`` is the Neural-Engine-friendly default.
        dtype: Target weight dtype. ``int8`` ~= 2x smaller than fp16 weights.
        nbits_palette: If set (e.g. 4 or 6), apply k-means palettisation at this
            bit width instead of linear quantisation. Smaller, slightly lossier.
    """

    mode: QuantMode = "linear_symmetric"
    dtype: Literal["int8", "int4"] = "int8"
    nbits_palette: int | None = None


# --- Reasoning (LanguageModel) --------------------------------------------- #

@dataclass(frozen=True)
class ReasoningConfig:
    """Settings for the Reason stage.

    ``model`` is only consulted by the Claude provider. The on-device Swift
    counterpart ignores it and uses Apple's ``SystemLanguageModel``.
    """

    model: str = "claude-sonnet-4-6"  # fast/cheap default for structured reasoning
    max_tokens: int = 1024
    system_prompt: str = field(
        default=(
            "You are an on-device reasoning step in a perceive->reason pipeline. "
            "You receive structured machine-vision output and must return a concise, "
            "well-calibrated assessment. Never invent detections that are not present. "
            "State your confidence and flag when the visual evidence is insufficient."
        )
    )
