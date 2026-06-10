"""The *Perceive* stage: turn a trained vision model into a Core ML package.

Public surface:

* :func:`convert.convert_torchvision` — convert a named torchvision classifier.
* :func:`convert.convert_traced`      — convert any traced ``torch.jit`` module.
* :func:`optimize.compress`           — int8 / palettised weight compression.
* :func:`inspect_model.summarize`     — human-readable report of an mlpackage.

All three run on Linux, macOS and Windows. Only *running predictions* on the
resulting package requires macOS / Apple silicon (Core ML's native runtime).
"""
from . import convert, optimize, inspect_model  # noqa: F401

__all__ = ["convert", "optimize", "inspect_model"]
