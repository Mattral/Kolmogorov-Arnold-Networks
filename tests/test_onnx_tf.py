"""Tests for the TensorFlow → ONNX export path."""
from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from kanx import KAN
from kanx.onnx_export import export_onnx_tf


def test_tf_to_onnx_parity(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("tf2onnx")

    model = KAN([2, 16, 1])
    model(tf.zeros((1, 2)))   # build

    out_path = tmp_path / "kan_tf.onnx"
    export_onnx_tf(model, str(out_path))
    assert os.path.exists(out_path)

    sess = onnxruntime.InferenceSession(str(out_path))
    xin = np.random.RandomState(0).randn(7, 2).astype(np.float32)
    onnx_out = sess.run(None, {"input": xin})[0]
    tf_out = model(xin).numpy()
    np.testing.assert_allclose(onnx_out, tf_out, atol=1e-5)


def test_tf_to_onnx_dynamic_batch(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")

    model = KAN([2, 8, 1])
    model(tf.zeros((1, 2)))
    out_path = tmp_path / "kan_tf.onnx"
    export_onnx_tf(model, str(out_path))

    sess = onnxruntime.InferenceSession(str(out_path))
    for n in (1, 2, 8, 50):
        out = sess.run(None, {"input": np.zeros((n, 2), dtype=np.float32)})[0]
        assert out.shape == (n, 1)
