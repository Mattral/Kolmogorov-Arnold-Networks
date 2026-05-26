"""End-to-end: train → save → ONNX export → ONNX Runtime inference.

The fastest path from "I have data" to "I have a portable inference artifact".

Run:
    python examples/onnx_pipeline.py
"""
from __future__ import annotations

import os

# Threading hygiene (torch + onnxruntime + numpy share BLAS pools).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

torch.set_num_threads(1)

from kanx.torch import KAN, export_onnx # noqa: E402


def main():
    torch.manual_seed(0)
    # 1. Build & train
    X = torch.randn(512, 2)
    y = torch.sin(torch.pi * X[:, :1])
    model = KAN([2, 32, 1])
    model.fit(X, y, epochs=15, lr=1e-2, batch_size=64, verbose=0)

    # 2. Save the eager checkpoint
    model.save("kan.pt")
    print("Saved kan.pt")

    # 3. Export to ONNX with dynamic batch
    onnx_path = export_onnx(model, "kan.onnx", sample_input=torch.zeros(1, 2))
    print(f"Exported {onnx_path}")

    # 4. Run with ONNX Runtime — anywhere
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("Install: pip install onnxruntime")

    sess = ort.InferenceSession("kan.onnx")
    xin = np.random.randn(8, 2).astype("float32")
    onnx_out = sess.run(None, {"input": xin})[0]
    torch_out = model(torch.from_numpy(xin)).detach().numpy()
    diff = float(np.max(np.abs(onnx_out - torch_out)))
    print(f"Max ONNX↔Torch divergence: {diff:.2e}  (should be <1e-5)")


if __name__ == "__main__":
    main()
