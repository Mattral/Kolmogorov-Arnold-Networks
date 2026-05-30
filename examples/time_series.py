"""Tiny time-series forecasting example.

Builds a sliding-window feature matrix on a noisy sine wave and trains a
small KAN to predict the next value. Demonstrates:
* Window/lag features fed straight to `KAN`
* Multi-step inference loop

Run:
    python examples/time_series.py
"""
from __future__ import annotations

import numpy as np

from kanx import KAN, fit_grid_to_data, set_global_seed


def main(window: int = 8, horizon: int = 32):
    set_global_seed(0)
    t = np.linspace(0, 8 * np.pi, 1024).astype("float32")
    s = (np.sin(t) + 0.1 * np.sin(3 * t)).astype("float32")

    # Build (X, y) with X[i] = past `window` values, y[i] = next value.
    X = np.stack([s[i:i + window] for i in range(len(s) - window - 1)])
    y = s[window:-1].reshape(-1, 1)

    n_train = int(0.8 * len(X))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    model = KAN([window, 32, 1], grid_size=6)
    model(np.zeros((1, window), dtype="float32"))
    fit_grid_to_data(model, Xtr)

    model.fit(Xtr, ytr, epochs=20, batch_size=64, verbose=0)
    mse = float(np.mean((model.predict(Xte, verbose=0) - yte) ** 2))
    print(f"1-step test MSE: {mse:.6f}")

    # Rolling multi-step forecast (autoregressive).
    history = Xte[0].copy()
    forecast = []
    for _ in range(horizon):
        nxt = float(model.predict(history[None, :], verbose=0)[0, 0])
        forecast.append(nxt)
        history = np.concatenate([history[1:], np.array([nxt], dtype="float32")])
    print(f"First 5 multi-step forecasts: {[round(v, 4) for v in forecast[:5]]}")


if __name__ == "__main__":
    main()
