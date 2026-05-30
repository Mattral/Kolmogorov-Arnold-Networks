"""Tests for the PyTorch backend (`kanx.torch`)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from kanx.torch import KAN, Trainer, build_kan, export_onnx
from kanx.torch.layers import KANLinear, b_spline_basis, extend_grid


# ---------------------------------------------------------------------------
# B-spline primitives — should match the TF backend's numerical contracts
# ---------------------------------------------------------------------------
def test_extend_grid_shape_and_values():
    grid = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    ext = extend_grid(grid, 2)
    assert ext.shape == (1, 8)
    np.testing.assert_allclose(
        ext.numpy()[0],
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        atol=1e-6,
    )


def test_b_spline_partition_of_unity_torch():
    grid = torch.linspace(0.0, 1.0, 7).unsqueeze(0)
    ext = extend_grid(grid, 3)
    x = torch.linspace(0.1, 0.9, 17).unsqueeze(-1)
    basis = b_spline_basis(x, ext, 3)[..., 0, :]
    sums = basis.sum(dim=-1).numpy()
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)


def test_b_spline_non_negative_torch():
    grid = torch.linspace(-1.0, 1.0, 6).unsqueeze(0)
    ext = extend_grid(grid, 3)
    x = torch.rand(50, 1) * 1.9 - 0.95
    basis = b_spline_basis(x, ext, 3).numpy()
    assert (basis >= -1e-6).all()


# ---------------------------------------------------------------------------
# KANLinear (torch)
# ---------------------------------------------------------------------------
def test_kanlinear_torch_forward_and_gradients():
    layer = KANLinear(4, 3, grid_size=5, spline_order=3)
    x = torch.randn(8, 4, requires_grad=True)
    out = layer(x)
    assert out.shape == (8, 3)
    loss = out.pow(2).mean()
    loss.backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_kanlinear_torch_num_basis_pykan_convention():
    layer = KANLinear(2, 1, grid_size=5, spline_order=3)
    assert layer.num_basis == 5 + 3
    assert layer.spline_weight.shape == (2, 1, 8)


@pytest.mark.parametrize("bad_args", [
    dict(in_features=0, out_features=1),
    dict(in_features=1, out_features=0),
    dict(in_features=2, out_features=2, grid_size=0),
    dict(in_features=2, out_features=2, spline_order=0),
    dict(in_features=2, out_features=2, grid_range=(1, -1)),
])
def test_kanlinear_torch_rejects_invalid(bad_args):
    with pytest.raises(ValueError):
        KANLinear(**bad_args)


# ---------------------------------------------------------------------------
# KAN model (torch)
# ---------------------------------------------------------------------------
def test_kan_torch_from_widths_and_dicts_equivalent_shapes():
    a = KAN([2, 8, 1])
    b = KAN([{"in_features": 2, "out_features": 8}, {"in_features": 8, "out_features": 1}])
    x = torch.randn(3, 2)
    assert a(x).shape == b(x).shape == (3, 1)


def test_kan_torch_predict_helper():
    model = KAN([2, 16, 1])
    out = model.predict([[0.1, 0.2], [0.3, 0.4]])
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1)


def test_kan_torch_save_and_load(tmp_path):
    m1 = KAN([3, 8, 2])
    x = torch.randn(4, 3)
    ref = m1(x).detach()
    p = tmp_path / "m.pt"
    m1.save(str(p))
    m2 = KAN.load(str(p))
    np.testing.assert_allclose(m2(x).detach().numpy(), ref.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def test_trainer_reduces_loss():
    torch.manual_seed(0)
    model = build_kan([2, 16, 1])
    X = torch.randn(256, 2)
    y = torch.sin(np.pi * X[:, :1]) + 0.3 * X[:, 1:2]
    hist = Trainer(model).fit(X, y, epochs=5, lr=1e-2, batch_size=64, verbose=0)
    assert len(hist.loss) == 5
    # Loss should decrease over 5 epochs on a smooth target.
    assert hist.loss[-1] < hist.loss[0] * 0.7


def test_trainer_val_split_and_early_stopping():
    torch.manual_seed(0)
    model = KAN([2, 8, 1])
    X = torch.randn(128, 2)
    y = torch.randn(128, 1)
    # lr=0 → parameters never update → val loss is constant → early-stop fires
    hist = Trainer(model).fit(
        X, y,
        epochs=50, batch_size=64, lr=0.0, val_split=0.2,
        early_stopping_patience=2, verbose=0,
    )
    assert len(hist.loss) < 50


def test_trainer_rejects_bad_optimizer():
    from kanx.torch.trainer import _build_optimizer
    with pytest.raises(ValueError):
        _build_optimizer("nope", [torch.zeros(1, requires_grad=True)], 1e-3)


# ---------------------------------------------------------------------------
# ONNX export — full numerical parity with PyTorch eager
# ---------------------------------------------------------------------------
def test_onnx_export_and_parity(tmp_path):
    import onnxruntime as ort

    torch.manual_seed(0)
    model = KAN([2, 8, 1])
    sample = torch.randn(1, 2)
    out_path = tmp_path / "kan.onnx"
    export_onnx(model, str(out_path), sample_input=sample)
    assert os.path.exists(out_path)

    sess = ort.InferenceSession(str(out_path))
    xin = np.random.RandomState(0).randn(5, 2).astype(np.float32)
    onnx_out = sess.run(None, {"input": xin})[0]
    torch_out = model(torch.from_numpy(xin)).detach().numpy()
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-5)


def test_onnx_export_dynamic_batch(tmp_path):
    import onnxruntime as ort

    model = KAN([2, 4, 1])
    out_path = tmp_path / "kan.onnx"
    export_onnx(model, str(out_path), sample_input=torch.zeros(1, 2))
    sess = ort.InferenceSession(str(out_path))
    # Dynamic batch should accept arbitrary batch sizes at inference time.
    for n in (1, 3, 13, 100):
        out = sess.run(None, {"input": np.zeros((n, 2), dtype=np.float32)})[0]
        assert out.shape == (n, 1)
