"""Symbolic regression hooks for PyTorch KAN models.

This module tries to fit a simple closed-form function to each learned edge
spline function, using a linear scaling of a candidate basis.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layers import KANLinear, MatrixKANLinear, b_spline_basis, extend_grid
from .matrix_kan import b_spline_basis_matrix, extend_grid_matrix


class SymbolicFitter:
    """Post-hoc symbolic regression for PyTorch KAN edge functions."""

    CANDIDATES = {
        "identity": lambda x: x,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sqrt": lambda x: torch.sqrt(x.abs()),
        "sin": torch.sin,
        "cos": torch.cos,
        "exp": torch.exp,
        "log": lambda x: torch.log(x.abs() + 1e-8),
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _edge_target(self, layer: nn.Module, in_idx: int, out_idx: int, x: torch.Tensor) -> torch.Tensor:
        if isinstance(layer, KANLinear):
            grid_ext = extend_grid(layer.grid, layer.spline_order)
            basis = b_spline_basis(x, grid_ext, layer.spline_order)
        elif isinstance(layer, MatrixKANLinear):
            grid_ext = extend_grid_matrix(layer.grid, layer.spline_order)
            basis = b_spline_basis_matrix(x, grid_ext, layer.spline_order)
        else:
            raise ValueError(f"Unsupported layer type for symbolic fitting: {type(layer).__name__}")

        edge_weights = layer.spline_weight[in_idx, out_idx, :]
        return basis[:, in_idx, :] @ edge_weights

    def _layers(self, model: nn.Module) -> list[nn.Module]:
        return [
            layer
            for layer in model.modules()
            if isinstance(layer, (KANLinear, MatrixKANLinear))
        ]

    def _fit_edge(
        self,
        layer_idx: int,
        in_idx: int,
        out_idx: int,
        x_samples: torch.Tensor,
    ) -> tuple[str, float, dict[str, float]]:
        layers = self._layers(self.model)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"layer_idx {layer_idx} out of range")
        layer = layers[layer_idx]

        x = x_samples.detach().flatten().to(torch.float32)
        if x.ndim != 1:
            raise ValueError("x_samples must be a 1-D tensor")
        x_in = x.unsqueeze(1)

        with torch.no_grad():
            target = self._edge_target(layer, in_idx, out_idx, x_in).detach()

        best_name = "spline"
        best_r2 = 0.0
        best_params: dict[str, float] = {}

        target_mean = target.mean()
        ss_tot = torch.sum((target - target_mean) ** 2)
        if ss_tot == 0:
            return "spline", 0.0, {}

        for name, fn in self.CANDIDATES.items():
            a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            b = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            c = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            d = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            optimizer = torch.optim.LBFGS([a, b, c, d], max_iter=50, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                y_pred = a * fn(b * x + c) + d
                loss = torch.mean((y_pred - target) ** 2)
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
            except Exception:
                continue

            with torch.no_grad():
                y_pred = (a * fn(b * x + c) + d).detach()
                ss_res = torch.sum((target - y_pred) ** 2)
                r2 = 1.0 - float(ss_res / ss_tot)
                if r2 > best_r2:
                    best_r2 = r2
                    best_name = name
                    best_params = {
                        "a": float(a.item()),
                        "b": float(b.item()),
                        "c": float(c.item()),
                        "d": float(d.item()),
                    }

        return best_name, best_r2, best_params

    def fit_edge(
        self,
        layer_idx: int,
        in_idx: int,
        out_idx: int,
        x_samples: torch.Tensor,
        threshold: float = 0.99,
    ) -> tuple[str, float]:
        """Fit a candidate symbolic function to one learned spline edge."""
        name, r2, _params = self._fit_edge(layer_idx, in_idx, out_idx, x_samples)
        if r2 >= threshold:
            return name, r2
        return "spline", 0.0

    def fit_all(
        self,
        model: nn.Module | None = None,
        x_samples_per_layer: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, dict[tuple[int, int], dict[str, Any]]]:
        if model is not None:
            self.model = model
        layers = self._layers(self.model)
        results: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}

        for layer_idx, layer in enumerate(layers):
            if x_samples_per_layer is None or layer_idx not in x_samples_per_layer:
                low, high = layer.grid_range
                x_samples = torch.linspace(low, high, 200, dtype=torch.float32)
            else:
                x_samples = x_samples_per_layer[layer_idx].detach().flatten().to(torch.float32)

            layer_results: dict[tuple[int, int], dict[str, Any]] = {}
            in_features = layer.in_features
            out_features = layer.out_features
            for in_idx in range(in_features):
                for out_idx in range(out_features):
                    fn, r2, params = self._fit_edge(layer_idx, in_idx, out_idx, x_samples)
                    if r2 < 0.99:
                        fn = "spline"
                        params = {}
                    layer_results[(in_idx, out_idx)] = {
                        "fn": fn,
                        "r2": r2 if fn != "spline" else 0.0,
                        "params": params,
                    }
            results[layer_idx] = layer_results

        return results

    def to_sympy(self, fit_result: dict[str, Any]) -> str:
        try:
            import sympy as sp
        except ImportError as exc:
            raise ImportError(
                "sympy is required for SymbolicFitter.to_sympy(). "
                "Install it with `pip install kanx[symbolic]`."
            ) from exc

        if "fn" not in fit_result or "params" not in fit_result:
            raise ValueError("fit_result must be a single edge result with keys 'fn' and 'params'.")

        fn = fit_result["fn"]
        if fn == "spline":
            return "spline"

        params = fit_result["params"]
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 1.0))
        c = float(params.get("c", 0.0))
        d = float(params.get("d", 0.0))
        x = sp.symbols("x")

        if fn == "identity":
            expr = a * (b * x + c) + d
        elif fn == "square":
            expr = a * (b * x + c) ** 2 + d
        elif fn == "cube":
            expr = a * (b * x + c) ** 3 + d
        elif fn == "sqrt":
            expr = a * sp.sqrt(sp.Abs(b * x + c)) + d
        elif fn == "sin":
            expr = a * sp.sin(b * x + c) + d
        elif fn == "cos":
            expr = a * sp.cos(b * x + c) + d
        elif fn == "exp":
            expr = a * sp.exp(b * x + c) + d
        elif fn == "log":
            expr = a * sp.log(sp.Abs(b * x + c) + 1e-8) + d
        elif fn == "tanh":
            expr = a * sp.tanh(b * x + c) + d
        elif fn == "sigmoid":
            expr = a * (1 / (1 + sp.exp(-(b * x + c)))) + d
        else:
            return "spline"

        return str(sp.simplify(expr))
