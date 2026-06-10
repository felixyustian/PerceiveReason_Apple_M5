"""Run a Core ML prediction and benchmark it across compute units (macOS / M-series).

Usage:
    python bench.py                          # uses Model_int8.mlpackage + synthetic image
    python bench.py Model.mlpackage          # different model
    python bench.py Model_int8.mlpackage cat.jpg   # real image

This step needs the Core ML runtime, so it only works on macOS / Apple silicon
(not in the repo's Linux CI). On an M5 it exercises the Neural Engine.
"""
import sys
import time

import numpy as np
import coremltools as ct
from PIL import Image

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Model_int8.mlpackage"
IMG_PATH = sys.argv[2] if len(sys.argv) > 2 else None


def load_image(size=224):
    if IMG_PATH:
        return Image.open(IMG_PATH).convert("RGB").resize((size, size))
    # No image supplied -> synthetic one so the script always runs.
    print("No image given; using a synthetic random image.")
    arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    try:
        ml = ct.models.MLModel(MODEL)  # native Core ML runtime on macOS
    except Exception as e:
        print(f"Could not load '{MODEL}'. Convert + optimize first:\n"
              f"  python -m pipeline convert mobilenet_v3_small Model.mlpackage --pretrained\n"
              f"  python -m pipeline optimize Model.mlpackage Model_int8.mlpackage --dtype int8\n"
              f"Original error: {e}")
        sys.exit(1)

    img = load_image()

    # Single prediction -> class logits
    out = ml.predict({"image": img})
    logits = np.array(out["logits"]).ravel()
    print("top-1 class index:", int(logits.argmax()))

    # Latency benchmark: let the OS schedule (ALL) vs CPU-only
    def bench(units, n=50):
        m = ct.models.MLModel(MODEL, compute_units=units)
        m.predict({"image": img})  # warm up
        t = time.perf_counter()
        for _ in range(n):
            m.predict({"image": img})
        return (time.perf_counter() - t) / n * 1000  # ms/inference

    print("ALL      : %.2f ms" % bench(ct.ComputeUnit.ALL))
    print("CPU_ONLY : %.2f ms" % bench(ct.ComputeUnit.CPU_ONLY))


if __name__ == "__main__":
    main()