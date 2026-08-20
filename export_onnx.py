"""
Export the trained model to ONNX for in-Biome inference.

The Biome runs in Node / the browser, so ONNX + onnxruntime-web (browser) or
onnxruntime-node (server) is the path to running these weights without Python.

  python -m mycel_disposition_model.export_onnx
  -> weights/mycel-disposition.onnx
"""

from __future__ import annotations

import os

import torch

from .infer import load
from .model import N_FEATURES

HERE = os.path.dirname(__file__)


def main():
    model = load()
    dummy = torch.zeros(1, N_FEATURES, dtype=torch.float32)
    out = os.path.join(HERE, "weights", "mycel-disposition.onnx")
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["features"],
        output_names=["mood_logits", "tone", "hidden"],
        dynamic_axes={
            "features": {0: "batch"},
            "mood_logits": {0: "batch"},
            "tone": {0: "batch"},
            "hidden": {0: "batch"},
        },
        opset_version=17,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
