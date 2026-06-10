"""PyTorch -> Core ML conversion.

This is the Apple-stack analogue of a PyTorch -> TensorRT export: you take a
trained model from a training framework and lower it to the target runtime's
native format. Here the target is Core ML's ``mlprogram``, which the OS
schedules across CPU / GPU / Neural Engine.

The conversion itself is platform-independent (it runs in CI on Linux). What
you *cannot* do off-device is call ``mlmodel.predict(...)`` — that needs the
Core ML runtime which ships only with macOS. ``inspect_model.summarize`` gives
you everything that can be checked without the runtime.
"""
from __future__ import annotations

from pathlib import Path

import coremltools as ct
import torch

from ..config import ConvertConfig

# Map our friendly string to the coremltools enum at call time so the import
# does not explode on coremltools versions with a different target list.
_TARGETS = {
    "iOS16": ct.target.iOS16,
    "iOS17": ct.target.iOS17,
    "iOS18": ct.target.iOS18,
}
_UNITS = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
}


def _deployment_target(name: str):
    """Resolve a target name, falling back to the newest one this coremltools
    build actually exposes (handles iOS26/iOS27 on newer toolchains)."""
    if name in _TARGETS:
        return _TARGETS[name]
    enum_value = getattr(ct.target, name, None)
    if enum_value is not None:
        return enum_value
    return ct.target.iOS18  # safe floor


def convert_traced(
    traced: torch.jit.ScriptModule,
    output_path: str | Path,
    *,
    config: ConvertConfig | None = None,
    output_name: str = "logits",
) -> ct.models.MLModel:
    """Convert an already-traced TorchScript module to a Core ML package.

    Args:
        traced: Result of ``torch.jit.trace(model.eval(), example_input)``.
        output_path: Where to write the ``.mlpackage`` directory.
        config: Conversion knobs; see :class:`ConvertConfig`.
        output_name: Name to give the single output tensor.

    Returns:
        The in-memory ``MLModel`` (also saved to ``output_path``).
    """
    config = config or ConvertConfig()
    s = config.image_size

    if config.as_image_input:
        # scale=1/255 maps 0..255 pixel buffers to 0..1; bias left at 0 so the
        # app can fold mean/std normalisation into the model if desired.
        inputs = [ct.ImageType(name="image", shape=(1, 3, s, s),
                               scale=1 / 255.0, bias=[0, 0, 0])]
    else:
        inputs = [ct.TensorType(name="image", shape=(1, 3, s, s))]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=output_name)],
        convert_to="mlprogram",
        minimum_deployment_target=_deployment_target(config.deployment_target),
        compute_units=_UNITS[config.compute_units],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return mlmodel


def convert_torchvision(
    arch: str,
    output_path: str | Path,
    *,
    pretrained: bool = False,
    config: ConvertConfig | None = None,
) -> ct.models.MLModel:
    """Convert a named ``torchvision`` classifier to Core ML.

    Args:
        arch: A torchvision model factory name, e.g. ``"mobilenet_v3_small"``,
            ``"resnet18"``, ``"efficientnet_b0"``.
        output_path: Destination ``.mlpackage``.
        pretrained: If True, download ImageNet weights (needs network access).
            Defaults to False so conversion works in sandboxed CI with random
            weights — the graph is identical, which is all the converter cares
            about. Flip to True on your own machine for a usable classifier.
        config: Conversion knobs.

    Returns:
        The saved ``MLModel``.
    """
    import torchvision

    config = config or ConvertConfig()
    factory = getattr(torchvision.models, arch, None)
    if factory is None:
        raise ValueError(f"Unknown torchvision architecture: {arch!r}")

    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights).eval()

    example = torch.rand(1, 3, config.image_size, config.image_size)
    traced = torch.jit.trace(model, example)
    return convert_traced(traced, output_path, config=config)
