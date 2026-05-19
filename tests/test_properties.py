"""Property-based tests using Hypothesis.

These tests sample random valid inputs from the configuration space and assert
that mathematical invariants hold across the whole input domain — not just on
the hand-picked examples in the rest of the suite.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kanx import KAN
from kanx.layers import KANLinear, b_spline_basis, extend_grid

_settings = settings(
    max_examples=15,
    deadline=None,  # TF kernel launches make per-example timing noisy
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
@_settings
@given(
    grid_size=st.integers(min_value=2, max_value=10),
    k=st.integers(min_value=1, max_value=5),
    n=st.integers(min_value=1, max_value=8),
)
def test_partition_of_unity_property(grid_size, k, n):
    """For *any* valid (grid_size, k), B-splines should sum to 1 inside the inner range."""
    grid = tf.linspace(-1.0, 1.0, grid_size + 1)[None, :]
    grid_ext = extend_grid(grid, k)
    # Sample well inside the inner knot range to avoid boundary edge cases.
    x = tf.linspace(-0.8, 0.8, n)[:, None]
    basis = b_spline_basis(x, grid_ext, k)[..., 0, :]
    sums = tf.reduce_sum(basis, axis=-1).numpy()
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-4)


@_settings
@given(
    in_features=st.integers(min_value=1, max_value=6),
    out_features=st.integers(min_value=1, max_value=6),
    grid_size=st.integers(min_value=2, max_value=8),
    spline_order=st.integers(min_value=1, max_value=4),
    batch=st.integers(min_value=1, max_value=6),
)
def test_kanlinear_forward_shape_property(in_features, out_features, grid_size, spline_order, batch):
    """Forward pass shape must always be `(batch, out_features)` for any valid config."""
    layer = KANLinear(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    x = tf.random.uniform((batch, in_features), -0.9, 0.9)
    out = layer(x)
    assert out.shape == (batch, out_features)
    # And the result is finite
    assert np.all(np.isfinite(out.numpy()))


@_settings
@given(
    widths=st.lists(st.integers(min_value=1, max_value=6), min_size=2, max_size=4),
    batch=st.integers(min_value=1, max_value=4),
)
def test_kan_model_shape_property(widths, batch):
    """A KAN with any valid width tuple should produce `(batch, widths[-1])`."""
    model = KAN(widths)
    x = tf.random.uniform((batch, widths[0]), -0.9, 0.9)
    out = model(x)
    assert tuple(out.shape) == (batch, widths[-1])
    assert np.all(np.isfinite(out.numpy()))


@_settings
@given(
    batch=st.integers(min_value=1, max_value=8),
    in_features=st.integers(min_value=1, max_value=4),
    out_features=st.integers(min_value=1, max_value=4),
)
def test_kanlinear_gradients_always_finite(batch, in_features, out_features):
    """Gradients must always be finite — a guard against NaN-on-edge-case bugs."""
    layer = KANLinear(in_features, out_features)
    x = tf.random.uniform((batch, in_features), -0.5, 0.5)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(layer(x) ** 2)
    grads = tape.gradient(loss, layer.trainable_variables)
    for g in grads:
        assert g is not None
        assert np.all(np.isfinite(g.numpy()))
