"""PerceiveReason — a provider-agnostic on-device "Perceive -> Reason" pipeline
for the Apple ML stack (Core ML + Foundation Models).

The Python package covers the two stages that can be developed and verified on
any platform (including Linux CI):

* ``pipeline.coreml_convert`` — convert and optimise a vision model into a
  Core ML ``.mlpackage`` (the *Perceive* stage that runs on the Neural Engine).
* ``pipeline.reasoning``      — a ``LanguageModel`` abstraction (mirroring
  Apple's WWDC 2026 ``LanguageModelSession`` protocol) with swappable providers
  for the *Reason* stage. Ships a Claude provider and an offline mock provider.

The on-device Swift counterpart lives in ``../app`` and is compiled with Xcode.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
