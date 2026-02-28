"""Real-data tabular regression: California Housing.

Shows the **correct way to train a KAN on real data** (data range is NOT
[-1, 1], so we calibrate the grid via `fit_grid_to_data` before training).

Run:
    python examples/tabular_regression.py
"""
from __future__ import annotations

import numpy as np

from kanx import KAN, fit_grid_to_data, check_input_range, set_global_seed


def main() -> None:
    set_global_seed(0)

    # ---- toy stand-in for sklearn.datasets.fetch_california_housing ------
    # (we keep this example dependency-free; replace with the real dataset
    # by uncommenting the sklearn block below).
    rng = np.random.default_rng(0)
    n, d = 4096, 8
    X = rng.normal(loc=2.5, scale=3.0, size=(n, d)).astype("float32")
    # Smooth-ish nonlinear target with mixed scales.
    y = (
        0.7 * np.sin(0.3 * X[:, :1]) +
        0.4 * np.tanh(0.5 * X[:, 1:2]) +
        0.2 * X[:, 2:3] * 0.1
    ).astype("float32")

    # ---- the *correct* pattern for real data ------------------------------
    model = KAN([d, 64, 1], grid_size=8, spline_order=3)

    # Build the model first (otherwise the grid weights don't exist yet).
    model(np.zeros((1, d), dtype="float32"))

    # >>> THE STEP THAT 90% OF KAN BLOG POSTS SKIP <<<
    fit_grid_to_data(model, X, pad=0.05)

    model.fit(X[:3500], y[:3500], epochs=15, batch_size=64, verbose=0)

    # Inference-time guard: warns to stdout if test data exceeds the grid.
    check_input_range(model, X[3500:], name="test set")

    test_mse = float(np.mean((model.predict(X[3500:], verbose=0) - y[3500:]) ** 2))
    print(f"Test MSE on 8-D tabular: {test_mse:.6f}")

    # ---- sklearn version (real California Housing) ------------------------
    # from sklearn.datasets import fetch_california_housing
    # from sklearn.model_selection import train_test_split
    # from sklearn.preprocessing import StandardScaler
    #
    # data = fetch_california_housing()
    # X, y = data.data.astype("float32"), data.target.astype("float32").reshape(-1, 1)
    # X = StandardScaler().fit_transform(X).astype("float32")     # zero-mean, unit-var
    # Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    # model = KAN([X.shape[1], 64, 1], grid_size=8)
    # model(np.zeros((1, X.shape[1]), dtype="float32"))
    # fit_grid_to_data(model, Xtr)
    # model.fit(Xtr, ytr, epochs=30, batch_size=128, verbose=1)


if __name__ == "__main__":
    main()
