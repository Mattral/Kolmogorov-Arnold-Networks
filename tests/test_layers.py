"""Unit tests for `kanx.layers` (B-spline + KANLinear)."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from kanx.layers import KANLinear, b_spline_basis, extend_grid


# ---------------------------------------------------------------------------
# extend_grid
# ---------------------------------------------------------------------------
def test_extend_grid_shape_and_values():
    grid = tf.constant([[0.0, 1.0, 2.0, 3.0]])  # (1 feature, 4 points)
    k = 2
    ext = extend_grid(grid, k).numpy()
    assert ext.shape == (1, 4 + 2 * k)
    # Uniformly spaced extension by step h=1.0
    np.testing.assert_allclose(ext[0], [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-6)


def test_extend_grid_k_zero_is_noop():
    grid = tf.constant([[0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(extend_grid(grid, 0).numpy(), grid.numpy())


def test_extend_grid_rejects_negative_k():
    with pytest.raises(ValueError):
        extend_grid(tf.constant([[0.0, 1.0]]), -1)


# ---------------------------------------------------------------------------
# b_spline_basis
# ---------------------------------------------------------------------------
def test_b_spline_basis_shape_matches_pykan_convention():
    grid_size, k = 5, 3
    grid = tf.linspace(-1.0, 1.0, grid_size + 1)[None, :]   # (1, G+1)
    grid_ext = extend_grid(grid, k)
    x = tf.random.uniform((7, 1), -1.0, 1.0)
    basis = b_spline_basis(x, grid_ext, k)
    # Expected num basis = grid_size + spline_order = 5 + 3 = 8
    assert basis.shape == (7, 1, grid_size + k)


def test_b_spline_partition_of_unity():
    """Inside the inner domain B-splines sum to 1."""
    grid_size, k = 6, 3
    grid = tf.linspace(0.0, 1.0, grid_size + 1)[None, :]
    grid_ext = extend_grid(grid, k)
    # Sample well within the inner knot range.
    x = tf.linspace(0.1, 0.9, 17)[:, None]
    basis = b_spline_basis(x, grid_ext, k)[..., 0, :]   # (17, num_basis)
    sums = tf.reduce_sum(basis, axis=-1).numpy()
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)


def test_b_spline_non_negative():
    grid_size, k = 5, 3
    grid = tf.linspace(-1.0, 1.0, grid_size + 1)[None, :]
    grid_ext = extend_grid(grid, k)
    x = tf.random.uniform((50, 1), -0.95, 0.95)
    basis = b_spline_basis(x, grid_ext, k).numpy()
    assert (basis >= -1e-6).all()


# ---------------------------------------------------------------------------
# KANLinear
# ---------------------------------------------------------------------------
def test_kanlinear_init_and_shapes():
    layer = KANLinear(in_features=10, out_features=5, grid_size=5, spline_order=3)
    layer.build((None, 10))
    assert layer.base_weight.shape == (10, 5)
    assert layer.spline_weight.shape == (10, 5, 5 + 3)
    assert layer.grid.shape == (10, 6)


def test_kanlinear_forward_shape():
    layer = KANLinear(4, 7)
    x = tf.random.normal([16, 4])
    out = layer(x)
    assert out.shape == (16, 7)


def test_kanlinear_gradients_flow():
    layer = KANLinear(3, 2)
    x = tf.random.normal([8, 3])
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(y ** 2)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert len(grads) == 2  # base_weight + spline_weight (grid is non-trainable by default)
    for g in grads:
        assert g is not None
        assert np.all(np.isfinite(g.numpy()))


def test_kanlinear_invalid_args():
    with pytest.raises(ValueError):
        KANLinear(0, 5)
    with pytest.raises(ValueError):
        KANLinear(5, 5, grid_size=0)
    with pytest.raises(ValueError):
        KANLinear(5, 5, spline_order=0)
    with pytest.raises(ValueError):
        KANLinear(5, 5, grid_range=(1.0, -1.0))


def test_kanlinear_config_roundtrip():
    layer = KANLinear(3, 4, grid_size=6, spline_order=2, regularization_factor=0.01)
    cfg = layer.get_config()
    rebuilt = KANLinear.from_config(cfg)
    assert rebuilt.in_features == 3
    assert rebuilt.out_features == 4
    assert rebuilt.grid_size == 6
    assert rebuilt.spline_order == 2
    assert rebuilt.regularization_factor == pytest.approx(0.01)


def test_kanlinear_rejects_mismatched_input():
    layer = KANLinear(4, 2)
    with pytest.raises(ValueError):
        layer.build((None, 3))
