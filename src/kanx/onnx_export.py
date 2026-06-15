"""ONNX export for the TensorFlow backend (via tf2onnx).

Usage::

    from kanx import KAN
    from kanx.onnx_export import export_onnx_tf

    model = KAN([2, 64, 1])
    model(tf.zeros((1, 2)))    # build
    export_onnx_tf(model, "kan_tf.onnx")
"""
from __future__ import annotations

import os

import numpy as np
import tensorflow as tf


def export_onnx_tf(
    model: tf.keras.Model,
    path: str,
    sample_input: np.ndarray | None = None,
    *,
    opset: int = 17,
) -> str:
    """Export a kanx (TF) model to ONNX.

    Returns the absolute path of the written .onnx file.
    """
    try:
        import tf2onnx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tf2onnx is required for TF→ONNX export. "
            "Install with `pip install tf2onnx`."
        ) from exc

    # Build the model if it hasn't been called yet.
    if sample_input is None:
        # Inspect first layer for `in_features` (KAN convention)
        try:
            in_features = model.layers[0].in_features
        except AttributeError as exc:
            raise ValueError(
                "sample_input is required when the model's first layer "
                "doesn't expose `in_features`."
            ) from exc
        sample_input = np.zeros((1, in_features), dtype=np.float32)

    # Ensure the model is built.
    model(sample_input)

    # Workaround for tf2onnx + Keras 3 incompatibility: wrap as tf.function
    # and convert via the function path instead of `from_keras`.
    in_dim = sample_input.shape[1]
    spec = (tf.TensorSpec((None, in_dim), tf.float32, name="input"),)

    @tf.function(input_signature=spec)
    def _serve(x):
        return model(x, training=False)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tf2onnx.convert.from_function(
        _serve,
        input_signature=spec,
        opset=opset,
        output_path=path,
    )
    return os.path.abspath(path)
