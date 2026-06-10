# `pipeline/` — the Python half

Everything here runs on Linux, macOS and Windows. It owns the two stages that
do *not* require an Apple device to develop and verify.

## `coreml_convert/` — the Perceive stage

| File | Role |
|---|---|
| `convert.py` | PyTorch → Core ML `mlprogram`. `convert_torchvision(...)` for named models, `convert_traced(...)` for any TorchScript module. The Apple-stack analogue of a PyTorch→TensorRT export. |
| `optimize.py` | Post-training weight compression — `int8` linear quantisation (default) or k-means palettisation at N bits. |
| `inspect_model.py` | Runtime-free inspection: model type, I/O names, package size, compression ratio. This is what CI asserts on. |

What you can do off-device: convert, compress, inspect.
What needs macOS: `mlmodel.predict(...)` — Core ML's native runtime.

## `reasoning/` — the Reason stage

A small `LanguageModel` protocol (`language_model.py`) with two providers:

| Provider | File | Use |
|---|---|---|
| Claude | `providers/claude_provider.py` | Production reasoning via the Anthropic Messages API. Reads `ANTHROPIC_API_KEY`. |
| Mock | `providers/mock_provider.py` | Offline, deterministic. Powers CI and `--provider mock`. |

`perceive_reason.py` is the glue: detections → prompt (`prompts.py`) → provider
→ normalised result. It depends only on the protocol, so adding Gemini or a
local model is a new file, not a rewrite.

## CLI

`python -m pipeline {convert,optimize,inspect,reason,demo}` — see `cli.py` or
`python -m pipeline -h`.

## Configuration

`config.py` holds the tunable defaults (image size, deployment target, compute
units, quantisation mode, reasoning model id) as frozen dataclasses, so the CLI,
tests and examples agree and every choice is documented in one place.
