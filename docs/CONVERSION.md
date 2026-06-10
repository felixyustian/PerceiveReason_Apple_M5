# Conversion & optimisation guide

How the Perceive stage turns a trained model into a deployable Core ML package,
and how to compress it.

## 1. Convert

```python
from pipeline.coreml_convert import convert
from pipeline.config import ConvertConfig

# Named torchvision model (use pretrained=True on a machine with network).
convert.convert_torchvision(
    "mobilenet_v3_small", "Model.mlpackage",
    pretrained=True, config=ConvertConfig(image_size=224, deployment_target="iOS18"))

# Or any traced module:
import torch
traced = torch.jit.trace(my_model.eval(), torch.rand(1, 3, 224, 224))
convert.convert_traced(traced, "Model.mlpackage")
```

Notes:
- Output is an `mlprogram` (the modern Core ML format), saved as a `.mlpackage`.
- `ImageType` input (`scale=1/255`) lets the app pass a camera pixel buffer
  directly. Set `as_image_input=False` for a float `TensorType` if you
  preprocess upstream.
- **Pretrained weights need network access.** In sandboxed CI we convert with
  random weights — the graph (and therefore the conversion) is identical; only
  the numbers differ. Use `--pretrained` for a model that actually classifies.
- coremltools 9.x is tested against torch ≤ 2.7. Newer torch usually works but
  prints an "untested" warning; pin torch if you want it silent.

## 2. Optimise (compress weights)

```python
from pipeline.coreml_convert import optimize, inspect_model
from pipeline.config import OptimizeConfig
import coremltools as ct

m = ct.models.MLModel("Model.mlpackage")

# int8 linear quantisation (default, Neural-Engine friendly)
optimize.compress(m, "Model_int8.mlpackage", config=OptimizeConfig(dtype="int8"))

# or 4-bit k-means palettisation (smaller, more accuracy risk)
optimize.compress(m, "Model_4bit.mlpackage", config=OptimizeConfig(nbits_palette=4))

print(inspect_model.compression_ratio("Model.mlpackage", "Model_int8.mlpackage"))
```

Measured on MobileNetV3-Small in this repo: **5.17 MB (fp16) → 2.68 MB (int8),
≈1.93×.** Palettisation at 4 bits compresses further still.

> Compression is lossy. Always re-measure accuracy **on device** after
> quantising — that step needs the Core ML runtime (macOS) and is out of CI scope.

## 3. Inspect (off-device verification)

```python
from pipeline.coreml_convert import inspect_model
print(inspect_model.summarize("Model_int8.mlpackage").as_dict())
# {'model_type': 'mlProgram', 'inputs': ['image'], 'outputs': ['logits'], 'size_mb': 2.68}
```

This is what CI checks: the package exists, is an `mlprogram`, has the expected
I/O, and got smaller after compression.

## Why no `predict()` here

`coremltools` on Linux logs `Failed to load _MLModelProxy` — the native Core ML
runtime is macOS-only. Conversion, compression and inspection all work without
it; running the model does not. That is the single hard boundary of the Python
half, and it is by design, not a bug.
