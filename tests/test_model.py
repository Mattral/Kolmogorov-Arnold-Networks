"""Unit tests for `kanx.model.KAN`."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from kanx.model import KAN, build_kan


def test_kan_from_widths():
    model = KAN([2, 8, 4, 1])
    x = tf.random.normal([5, 2])
    y = model(x)
    assert y.shape == (5, 1)
    assert len(model.layers) == 3


def test_kan_from_dicts():
    model = KAN([
        {"in_features": 2, "out_features": 8},
        {"in_features": 8, "out_features": 1},
    ])
    y = model(tf.random.normal([3, 2]))
    assert y.shape == (3, 1)


def test_kan_predict_tensor_no_grad():
    model = KAN([2, 4, 1])
    x = tf.random.normal([6, 2])
    out = model.predict_tensor(x).numpy()
    assert out.shape == (6, 1)
    assert np.all(np.isfinite(out))


def test_kan_train_step_runs():
    model = build_kan([2, 16, 1])
    model.compile(optimizer="adam", loss="mse")
    X = np.random.RandomState(0).uniform(-1, 1, size=(64, 2)).astype("float32")
    y = np.sin(np.pi * X[:, :1]).astype("float32")
    hist = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    losses = hist.history["loss"]
    assert all(np.isfinite(losses))
    # Loss should typically decrease over 2 epochs on a fittable target.
    assert losses[-1] <= losses[0] * 1.5  # weak monotonicity check


def test_kan_rejects_empty():
    with pytest.raises(ValueError):
        KAN([])
    with pytest.raises(ValueError):
        KAN([10])  # single width is ambiguous


def test_kan_save_and_load(tmp_path):
    model = KAN([3, 8, 2])
    x = tf.random.normal([4, 3])
    y_before = model(x).numpy()
    path = tmp_path / "kan.keras"
    model.save(str(path))
    loaded = tf.keras.models.load_model(str(path), compile=False)
    y_after = loaded(x).numpy()
    np.testing.assert_allclose(y_after, y_before, atol=1e-5)
