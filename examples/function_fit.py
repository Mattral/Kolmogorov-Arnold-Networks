"""Fit a tiny KAN to f(x) = sin(pi*x1) + x2^2 on synthetic data.

Run:
    python examples/function_fit.py
"""
from __future__ import annotations

import numpy as np

from kanx.config import validate_config
from kanx.train import train


def main() -> None:
    cfg = validate_config({
        "model": {"layers": [2, 32, 1], "grid_size": 5, "spline_order": 3},
        "training": {
            "lr": 1e-2, "epochs": 30, "batch_size": 64,
            "loss": "mse", "optimizer": "adam", "val_split": 0.2,
            "early_stopping_patience": 5,
        },
        "checkpoint": {
            "dir": "checkpoints", "filename": "function_fit.keras",
            "save_best_only": True, "monitor": "loss",
        },
    })
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(2048, 2)).astype("float32")
    y = (np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2).astype("float32")
    model, hist = train(cfg, X, y, verbose=2)
    print(f"Final train loss: {hist.history['loss'][-1]:.5f}")


if __name__ == "__main__":
    main()
