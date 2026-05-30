"""ONNX export utilities for the PyTorch backend.

Usage::

    from kanx.torch import KAN, export_onnx
    import torch

    model = KAN([2, 64, 1])
    export_onnx(model, "kan.onnx", sample_input=torch.randn(1, 2))

The exported graph supports dynamic batch dimension (``dynamic_axes`` set on
both input and output to allow any batch size at runtime).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    path: str,
    sample_input: Optional[torch.Tensor] = None,
    *,
    opset_version: int = 17,
    input_name: str = "input",
    output_name: str = "output",
) -> str:
    """Export a kanx.torch.KAN (or any nn.Module) to ONNX.

    Args:
        model:         model to export.
        path:          destination filesystem path (e.g. "kan.onnx").
        sample_input:  example input used for tracing. If None, a `(1, in_features)`
                       zero tensor is built from `model[0].in_features`.
        opset_version: ONNX opset to target.
        input_name:    name of the input node.
        output_name:   name of the output node.

    Returns:
        The absolute path to the written .onnx file.
    """
    if sample_input is None:
        if not hasattr(model[0], "in_features"):
            raise ValueError(
                "sample_input is required when the first layer has no in_features attribute"
            )
        sample_input = torch.zeros(1, model[0].in_features, dtype=torch.float32)

    model.eval()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        sample_input,
        path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
    )
    return os.path.abspath(path)
