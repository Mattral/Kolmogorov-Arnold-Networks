"""Quickstart: build a KAN and run a single forward pass."""
from __future__ import annotations

import numpy as np

from kanx import KAN, set_global_seed


def main() -> None:
    set_global_seed(0)
    model = KAN([2, 32, 1])
    X = np.random.uniform(-1, 1, size=(8, 2)).astype("float32")
    y = model(X).numpy()
    print(f"Input: {X.shape} | Output: {y.shape}")
    print(y[:3])


if __name__ == "__main__":
    main()
