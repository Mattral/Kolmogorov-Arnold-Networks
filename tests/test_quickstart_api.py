"""Tests for the new zero-friction APIs added in v0.1.3."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import kanx


def test_top_level_quickstart_returns_trained_model():
    """The 1-call magic moment: `kanx.quickstart()` returns a usable KAN."""
    model = kanx.quickstart(layers=(2, 8, 1), n_samples=128, epochs=2, verbose=0)
    out = model.predict(np.zeros((3, 2), dtype="float32"), verbose=0)
    assert out.shape == (3, 1)
    assert np.all(np.isfinite(out))


def test_tf_kan_fit_autocompiles():
    """`KAN([...]).fit(X, y)` should work without a manual compile()."""
    model = kanx.KAN([2, 8, 1])
    X = np.random.RandomState(0).uniform(-1, 1, (64, 2)).astype("float32")
    y = np.sin(np.pi * X[:, :1]).astype("float32")
    hist = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    assert "loss" in hist.history
    assert all(np.isfinite(hist.history["loss"]))


def test_torch_kan_fit_one_liner():
    """`kanx.torch.KAN([...]).fit(X, y)` — single call, no Trainer boilerplate."""
    from kanx.torch import KAN as TorchKAN
    torch.manual_seed(0)
    model = TorchKAN([2, 8, 1])
    X = torch.randn(64, 2)
    y = torch.sin(torch.pi * X[:, :1])
    hist = model.fit(X, y, epochs=3, batch_size=16, lr=1e-2, verbose=0)
    assert len(hist.loss) == 3
    assert hist.loss[-1] < hist.loss[0] * 1.5  # weak monotonic check


def test_version_is_present_and_correct():
    assert hasattr(kanx, "__version__")
    parts = kanx.__version__.split(".")
    assert len(parts) == 3
    for p in parts:
        assert p.isdigit()


def test_quickstart_is_exported():
    assert "quickstart" in kanx.__all__
    assert callable(kanx.quickstart)
