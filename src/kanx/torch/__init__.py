"""PyTorch backend for kanx.

This is a **parallel surface** to the default TensorFlow backend. Same maths,
same numerical contracts, same configuration semantics — but expressed as
`torch.nn.Module` objects that integrate naturally with the PyTorch ecosystem
(autograd, DataLoader, DDP, TorchScript, ONNX).

Quickstart::

    from kanx.torch import KAN, Trainer
    import torch

    model = KAN([2, 64, 1])
    X = torch.randn(512, 2)
    y = torch.sin(torch.pi * X[:, :1])

    Trainer(model).fit(X, y, epochs=30, lr=1e-2)
    print(model(X[:2]))

ONNX export::

    from kanx.torch import export_onnx
    export_onnx(model, "kan.onnx", sample_input=torch.randn(1, 2))
"""
from __future__ import annotations

from .layers import KANLinear, b_spline_basis, extend_grid
from .matrix_kan import MatrixKAN, MatrixKANLinear
from .model import KAN, build_kan
from .onnx_export import export_onnx
from .symbolic import SymbolicFitter
from .trainer import Trainer

__all__ = [
    "KAN",
    "KANLinear",
    "MatrixKAN",
    "MatrixKANLinear",
    "Trainer",
    "SymbolicFitter",
    "build_kan",
    "b_spline_basis",
    "extend_grid",
    "export_onnx",
]
