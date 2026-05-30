"""Performance / latency budgets.

These tests act as **regression alarms** — they don't assert specific numbers
(machines vary), but they fail loudly if the hot path becomes catastrophically
slow (e.g. someone re-introduces a Python `for` loop over the batch dimension).

Numbers are calibrated for a 2-vCPU CI runner with generous safety margins.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import tensorflow as tf

from kanx import KAN
from kanx.inference import predict


@pytest.fixture(scope="module")
def warm_model():
    model = KAN([4, 32, 1])
    model(tf.zeros((1, 4)))   # build
    return model


def _time(fn, repeats: int = 3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_forward_latency_small_batch_ms(warm_model):
    x = np.random.randn(32, 4).astype(np.float32)
    best = _time(lambda: warm_model(x), repeats=5)
    assert best < 0.5, f"forward(32, 4) took {best*1000:.1f}ms — perf regression?"


def test_forward_latency_large_batch_ms(warm_model):
    x = np.random.randn(4096, 4).astype(np.float32)
    best = _time(lambda: warm_model(x), repeats=3)
    assert best < 5.0, f"forward(4096, 4) took {best*1000:.0f}ms — perf regression?"


def test_predict_helper_overhead(warm_model):
    x = np.random.randn(64, 4).astype(np.float32)
    best = _time(lambda: predict(warm_model, x), repeats=5)
    assert best < 1.0


# pytest-benchmark is optional; if installed we publish a metric.
def test_forward_benchmarked(warm_model, benchmark):
    x = np.random.randn(64, 4).astype(np.float32)
    benchmark(lambda: warm_model(x).numpy())
