"""Binary classification on a synthetic moons-like dataset.

Demonstrates:
* `KAN` with `sparse_categorical_crossentropy` (or BCE for binary).
* `fit_grid_to_data` for inputs whose natural range is not [-1, 1].

Run:
    python examples/classification.py
"""
from __future__ import annotations

import numpy as np

from kanx import KAN, fit_grid_to_data, set_global_seed


def _moons(n: int, noise: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, size=n // 2)
    x0 = np.stack([np.cos(t),         np.sin(t)],         axis=1)
    x1 = np.stack([1 - np.cos(t),     1 - np.sin(t) - 0.5], axis=1)
    X = np.vstack([x0, x1]).astype("float32")
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype("int32")
    X += rng.normal(scale=noise, size=X.shape).astype("float32")
    perm = rng.permutation(n)
    return X[perm], y[perm]


def main():
    set_global_seed(0)
    X, y = _moons(2048)

    model = KAN([2, 32, 2])               # 2-class logits
    model(np.zeros((1, 2), dtype="float32"))
    fit_grid_to_data(model, X)            # essential

    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(X[:1700], y[:1700], epochs=15, batch_size=64, verbose=0)
    loss, acc = model.evaluate(X[1700:], y[1700:], verbose=0)
    print(f"Held-out accuracy: {acc * 100:.2f}%  |  loss: {loss:.4f}")


if __name__ == "__main__":
    main()
