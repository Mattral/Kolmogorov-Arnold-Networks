"""Integration tests for `kanx.train` and `kanx.inference`."""
from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from kanx.config import validate_config
from kanx.inference import load_model, predict, save_model
from kanx.model import KAN
from kanx.train import build_loss, build_optimizer, train


def _synthetic(n=128, in_dim=2):
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(n, in_dim)).astype("float32")
    y = (np.sin(np.pi * X[:, :1]) + 0.3 * X[:, 1:2]).astype("float32")
    return X, y


def test_build_optimizer_and_loss():
    assert isinstance(build_optimizer("adam", 1e-3), tf.keras.optimizers.Optimizer)
    assert isinstance(build_optimizer("adamw", 1e-3), tf.keras.optimizers.Optimizer)
    assert isinstance(build_optimizer("sgd", 1e-3), tf.keras.optimizers.Optimizer)
    assert isinstance(build_loss("mse"), tf.keras.losses.Loss)
    assert isinstance(build_loss("mae"), tf.keras.losses.Loss)
    assert isinstance(build_loss("scce"), tf.keras.losses.Loss)
    with pytest.raises(ValueError):
        build_optimizer("nope", 1e-3)
    with pytest.raises(ValueError):
        build_loss("nope")


def test_train_end_to_end_persists_checkpoint(tmp_path):
    cfg = validate_config({
        "model": {"layers": [2, 16, 1], "grid_size": 5, "spline_order": 3},
        "training": {"epochs": 3, "batch_size": 32, "lr": 1e-2, "val_split": 0.0},
        "checkpoint": {
            "dir": str(tmp_path / "ckpt"),
            "filename": "m.keras",
            "save_best_only": True,
            "monitor": "loss",
        },
    })
    X, y = _synthetic()
    model, hist = train(cfg, X, y, verbose=0)
    assert isinstance(model, KAN)
    assert len(hist.history["loss"]) == 3
    expected = os.path.join(str(tmp_path / "ckpt"), "m.keras")
    assert os.path.exists(expected), f"Checkpoint not written at {expected}"


def test_predict_shapes_and_batching():
    model = KAN([2, 8, 3])
    # warm up
    _ = model(tf.zeros((1, 2)))

    y1 = predict(model, [0.1, 0.2])
    assert y1.shape == (1, 3)

    y2 = predict(model, np.zeros((5, 2), dtype="float32"))
    assert y2.shape == (5, 3)

    # Batched path
    y3 = predict(model, np.zeros((10, 2), dtype="float32"), batch_size=3)
    assert y3.shape == (10, 3)


def test_predict_rejects_bad_rank():
    model = KAN([2, 4, 1])
    _ = model(tf.zeros((1, 2)))
    with pytest.raises(ValueError):
        predict(model, np.zeros((1, 1, 2), dtype="float32"))


def test_save_load_roundtrip(tmp_path):
    model = KAN([3, 8, 2])
    x = tf.random.normal([4, 3])
    expected = model(x).numpy()
    path = str(tmp_path / "m.keras")
    save_model(model, path)
    loaded = load_model(path)
    got = loaded(x).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "no.keras"))
