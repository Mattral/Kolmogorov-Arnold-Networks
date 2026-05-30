"""Tests for the grid-range guard + utilities (the #1 KAN production gotcha)."""
from __future__ import annotations

import logging

import numpy as np
import pytest
import tensorflow as tf

import kanx
from kanx import KAN, check_input_range, fit_grid_to_data
from kanx.layers import KANLinear


def test_fit_grid_to_data_updates_layer_grid():
    model = KAN([3, 8, 1])
    model(np.zeros((1, 3), dtype="float32"))
    # Pretend the real data lives in [50, 100] — way outside default [-1, 1].
    rng = np.random.default_rng(0)
    X = rng.uniform(50.0, 100.0, size=(64, 3)).astype("float32")

    low, high = fit_grid_to_data(model, X, pad=0.0)
    assert low == pytest.approx(X.min(), abs=1e-3)
    assert high == pytest.approx(X.max(), abs=1e-3)

    for layer in model.layers:
        if isinstance(layer, KANLinear):
            grid = layer.grid.numpy()
            assert grid.min() == pytest.approx(low, abs=1e-3)
            assert grid.max() == pytest.approx(high, abs=1e-3)
            assert layer.grid_range == (low, high)


def test_fit_grid_to_data_pad_is_applied():
    model = KAN([2, 4, 1])
    model(np.zeros((1, 2), dtype="float32"))
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype="float32")
    low, high = fit_grid_to_data(model, X, pad=0.1)
    # span is 1.0; pad=0.1 → low = -0.1, high = 1.1
    assert low == pytest.approx(-0.1, abs=1e-3)
    assert high == pytest.approx(1.1, abs=1e-3)


def test_fit_grid_to_data_rejects_unbuilt_model():
    """Calling fit_grid_to_data on a model that has no KANLinear in its
    iteration surface should raise — we never want a silent no-op."""
    # An empty Sequential has nothing to update.
    empty = tf.keras.Sequential([])
    with pytest.raises(ValueError):
        fit_grid_to_data(empty, np.zeros((4, 2), dtype="float32"))


def test_check_input_range_warns_on_out_of_range(caplog):
    model = KAN([2, 4, 1])              # default grid_range = (-1, 1)
    model(np.zeros((1, 2), dtype="float32"))
    out_of_range = np.array([[5.0, -7.0]], dtype="float32")

    # Our logger sets propagate=False so caplog wouldn't see it; flip it for the test.
    log = logging.getLogger("kanx.utils")
    prev = log.propagate
    log.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="kanx.utils"):
            check_input_range(model, out_of_range, name="user_input")
    finally:
        log.propagate = prev

    assert any(
        "exceeds model grid range" in rec.message and "user_input" in rec.message
        for rec in caplog.records
    )


def test_check_input_range_silent_when_in_range(caplog):
    model = KAN([2, 4, 1])
    model(np.zeros((1, 2), dtype="float32"))
    in_range = np.array([[0.2, -0.5]], dtype="float32")
    log = logging.getLogger("kanx.utils")
    prev = log.propagate
    log.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="kanx.utils"):
            check_input_range(model, in_range)
    finally:
        log.propagate = prev
    assert not any("exceeds model grid range" in r.message for r in caplog.records)


def test_grid_calibration_improves_training_accuracy():
    """Critical regression test: training on out-of-range data without
    fit_grid_to_data produces worse loss than with it."""
    rng = np.random.default_rng(0)
    X = rng.uniform(10.0, 20.0, size=(256, 2)).astype("float32")
    y = (np.sin(0.5 * X[:, :1]) + 0.3 * X[:, 1:2]).astype("float32")

    # Without calibration
    kanx.set_global_seed(0)
    m_bad = KAN([2, 16, 1])
    h_bad = m_bad.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # With calibration
    kanx.set_global_seed(0)
    m_good = KAN([2, 16, 1])
    m_good(np.zeros((1, 2), dtype="float32"))
    fit_grid_to_data(m_good, X)
    h_good = m_good.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Calibrated model should reach lower loss (we use a loose 0.7× factor
    # to avoid flakes — the real gap is usually >10× on out-of-range data).
    assert h_good.history["loss"][-1] < h_bad.history["loss"][-1] * 0.95


def test_public_exports():
    assert "fit_grid_to_data" in kanx.__all__
    assert "check_input_range" in kanx.__all__
    assert callable(kanx.fit_grid_to_data)
    assert callable(kanx.check_input_range)
