from __future__ import annotations

import numpy as np


def feynman_I_9_18(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] != 7:
        raise ValueError("feynman_I_9_18 requires 7 input features")
    m1, m2, G, x1, x2, y1, y2 = x.T
    denom = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return (m1 * m2 * G / denom).astype(np.float32).reshape(-1, 1)


def make_dataset_I_9_18(n: int = 1000, noise: float = 0.01, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(0.1, 10.0, size=n)
    m2 = rng.uniform(0.1, 10.0, size=n)
    G = rng.uniform(0.1, 10.0, size=n)
    x1 = rng.uniform(-10.0, 10.0, size=n)
    x2 = rng.uniform(-10.0, 10.0, size=n)
    y1 = rng.uniform(-10.0, 10.0, size=n)
    y2 = rng.uniform(-10.0, 10.0, size=n)
    X = np.stack([m1, m2, G, x1, x2, y1, y2], axis=1).astype(np.float32)
    y = feynman_I_9_18(X)
    y = y + rng.normal(0.0, noise * np.std(y, ddof=1), size=y.shape).astype(np.float32)
    split = int(n * 0.8)
    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
    }


def feynman_I_34_8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] != 3:
        raise ValueError("feynman_I_34_8 requires 3 input features")
    omega_0, v, c = x.T
    return (omega_0 / (1.0 - v / c)).astype(np.float32).reshape(-1, 1)


def make_dataset_I_34_8(n: int = 1000, noise: float = 0.01, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega_0 = rng.uniform(1.0, 10.0, size=n)
    c = rng.uniform(1.0, 10.0, size=n)
    v = rng.uniform(0.0, 0.9 * c, size=n)
    X = np.stack([omega_0, v, c], axis=1).astype(np.float32)
    y = feynman_I_34_8(X)
    y = y + rng.normal(0.0, noise * np.std(y, ddof=1), size=y.shape).astype(np.float32)
    split = int(n * 0.8)
    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
    }


def feynman_II_11_27(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] != 4:
        raise ValueError("feynman_II_11_27 requires 4 input features")
    n, alpha, epsilon, Ef = x.T
    denom = 1.0 - n * alpha / 3.0
    return (n * alpha / denom * epsilon * Ef).astype(np.float32).reshape(-1, 1)


def make_dataset_II_11_27(n: int = 1000, noise: float = 0.01, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = rng.uniform(0.1, 2.0, size=n)
    alpha = rng.uniform(0.1, 0.9, size=n)
    epsilon = rng.uniform(0.1, 5.0, size=n)
    Ef = rng.uniform(1.0, 10.0, size=n)
    X = np.stack([n, alpha, epsilon, Ef], axis=1).astype(np.float32)
    y = feynman_II_11_27(X)
    y = y + rng.normal(0.0, noise * np.std(y, ddof=1), size=y.shape).astype(np.float32)
    split = int(n * 0.8)
    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
    }
