"""PyTorch B-spline primitives + KANLinear layer.

Numerical contracts (asserted by tests):
    * `b_spline_basis` returns values in [0, 1] that sum to 1 inside the
      inner knot range (partition of unity).
    * `num_basis = grid_size + spline_order` (pykan convention, identical
      to the TensorFlow backend).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
def extend_grid(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetrically extend a uniform per-feature grid by `k` knots on each side.

    Args:
        grid: ``(in_features, grid_size + 1)`` tensor.
        k:    spline polynomial degree (number of knots to pad on each side).
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return grid

    n = grid.shape[1] - 1
    h = (grid[:, -1:] - grid[:, :1]) / n
    left_offsets = torch.arange(-k, 0, device=grid.device, dtype=grid.dtype).unsqueeze(0)
    right_offsets = torch.arange(1, k + 1, device=grid.device, dtype=grid.dtype).unsqueeze(0)
    left = grid[:, :1] + left_offsets * h
    right = grid[:, -1:] + right_offsets * h
    return torch.cat([left, grid, right], dim=1)


def b_spline_basis(x: torch.Tensor, grid_ext: torch.Tensor, k: int) -> torch.Tensor:
    """Vectorized Cox-de Boor recursion. Returns ``(batch, in_features, num_basis)``."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    x_e = x.unsqueeze(-1)              # (B, F, 1)
    g_e = grid_ext.unsqueeze(0)        # (1, F, G_ext)

    # Order-0 basis (piecewise constant).
    basis = ((x_e >= g_e[..., :-1]) & (x_e < g_e[..., 1:])).to(x.dtype)

    for p in range(1, k + 1):
        L = basis.shape[-1] - 1
        g_l = g_e[..., :L]
        g_lp = g_e[..., p : p + L]
        g_r = g_e[..., 1 : L + 1]
        g_rp = g_e[..., p + 1 : p + 1 + L]

        denom_l = g_lp - g_l
        denom_r = g_rp - g_r
        # Stable division: replace 0 denominators with 1 (numerator also 0).
        denom_l = torch.where(denom_l == 0, torch.ones_like(denom_l), denom_l)
        denom_r = torch.where(denom_r == 0, torch.ones_like(denom_r), denom_r)

        left_w = (x_e - g_l) / denom_l
        right_w = (g_rp - x_e) / denom_r

        basis = left_w * basis[..., :L] + right_w * basis[..., 1 : L + 1]

    return basis


# ---------------------------------------------------------------------------
class KANLinear(nn.Module):
    """One KAN edge bank in PyTorch.

    Computes::
        out(x) = silu(x) @ W_base  +  einsum('bik,iok->bo', basis(x), W_spline)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        scale_noise: float = 0.1,
        trainable_grid: bool = False,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")
        if grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must be (low, high) with low < high")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.base_activation_name = base_activation
        self.base_activation = _resolve_activation(base_activation)
        self.grid_range = tuple(grid_range)
        self.num_basis = self.grid_size + self.spline_order

        # base_weight: Glorot uniform
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.base_weight)

        # spline_weight: small-noise init so the layer starts close to the base path
        self.spline_weight = nn.Parameter(
            torch.randn(in_features, out_features, self.num_basis) * scale_noise
        )

        # Per-feature uniform grid
        low, high = self.grid_range
        grid_init = torch.linspace(low, high, self.grid_size + 1).unsqueeze(0).expand(
            in_features, -1
        ).contiguous()
        if trainable_grid:
            self.grid = nn.Parameter(grid_init)
        else:
            self.register_buffer("grid", grid_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_activation(x) @ self.base_weight
        grid_ext = extend_grid(self.grid, self.spline_order)
        basis = b_spline_basis(x, grid_ext, self.spline_order)
        # (B, F_in, num_basis) × (F_in, F_out, num_basis) → (B, F_out)
        spline_out = torch.einsum("bik,iok->bo", basis, self.spline_weight)
        return base_out + spline_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}, "
            f"activation={self.base_activation_name}"
        )


def _resolve_activation(name: str):
    name = name.lower()
    if name == "silu" or name == "swish":
        return nn.functional.silu
    if name == "relu":
        return nn.functional.relu
    if name == "gelu":
        return nn.functional.gelu
    if name == "tanh":
        return torch.tanh
    if name == "identity" or name == "linear":
        return lambda x: x
    raise ValueError(f"Unsupported activation: {name!r}")
