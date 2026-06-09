"""Visualise the learned edge functions of a KAN.

The whole point of KAN is that each EDGE is a learned 1-D function. After
training, you can plot those functions and inspect what the model has
discovered — this is the interpretability story that distinguishes KAN
from a black-box MLP.

Run:
    python examples/visualize_edges.py     # writes edges.png next to this file
"""

from __future__ import annotations

import os

import numpy as np

from kanx import KAN, fit_grid_to_data, set_global_seed
from kanx.layers import b_spline_basis, extend_grid


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    set_global_seed(0)

    # ---- train a small KAN on a separable target -------------------------
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(2048, 2)).astype("float32")
    y = (np.sin(np.pi * X[:, :1]) + 0.6 * X[:, 1:2] ** 2).astype("float32")

    model = KAN([2, 1])
    model(np.zeros((1, 2), dtype="float32"))
    fit_grid_to_data(model, X)
    model.fit(X, y, epochs=80, batch_size=64, verbose=0)

    # ---- evaluate each edge function on a sweep --------------------------
    layer = model.layers[0]
    xs = np.linspace(-1.05, 1.05, 200).astype("float32")
    grid_ext = extend_grid(layer.grid, layer.spline_order)

    # For each input feature i, vary x_i over the sweep while others are 0.
    # The contribution of feature i to output j is spline_weight[i, j, :] . B(x_i).
    edge_fns: dict[tuple[int, int], np.ndarray] = {}
    for i in range(layer.in_features):
        xin = np.zeros((len(xs), layer.in_features), dtype="float32")
        xin[:, i] = xs
        basis = b_spline_basis(xin, grid_ext, layer.spline_order).numpy()  # (200, F, K)
        spline_w = layer.spline_weight.numpy()                              # (F, O, K)
        for j in range(layer.out_features):
            # contribution_i_j(x) = Σ_k basis[t, i, k] * spline_w[i, j, k]
            edge_fns[(i, j)] = np.einsum("tk,k->t", basis[:, i, :], spline_w[i, j, :])

    # ---- plot ------------------------------------------------------------
    rows = layer.out_features
    cols = layer.in_features
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows), squeeze=False)
    for (i, j), curve in edge_fns.items():
        ax = axes[j][i]
        ax.plot(xs, curve, color="#7C3AED", linewidth=2)
        ax.axhline(0, color="#9CA3AF", linewidth=0.5)
        ax.set_title(f"edge x_{i} → y_{j}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Learned KAN edge functions  (each panel is a learned 1-D spline)",
                 fontsize=12, weight="bold")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edges.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
