"""Zero-friction quickstart helpers.

``kanx.quickstart()`` builds, trains and returns a tiny KAN in one call —
designed to be the first thing a new user types after ``pip install kanx``::

    >>> import kanx
    >>> model = kanx.quickstart()
    [kanx.quickstart] Building KAN[2, 32, 1]…
    [kanx.quickstart] Training 20 epochs on f(x)=sin(pi*x1)+x2^2…
    [kanx.quickstart] Final train loss: 0.000312
    >>> model.predict([[0.5, 0.2]])
    array([[1.04...]], dtype=float32)

That's it. No config files, no `compile()` dance, no manual loop.
"""
from __future__ import annotations

import numpy as np

from .config import CheckpointConfig, KanxConfig, ModelConfig, TrainingConfig
from .train import train
from .utils import set_global_seed


def quickstart(
    *,
    layers=(2, 32, 1),
    n_samples: int = 1024,
    epochs: int = 20,
    seed: int = 0,
    verbose: int = 0,
):
    """Train a tiny KAN on a smooth synthetic target. Returns the trained model.

    Args:
        layers:    network widths. Default ``(2, 32, 1)``.
        n_samples: training samples drawn uniformly from [-1, 1]^d.
        epochs:    number of training epochs.
        seed:      RNG seed for reproducibility.
        verbose:   passed through to the Keras progress bar.

    Returns:
        A trained :class:`kanx.KAN` (TensorFlow). Call ``model.predict(X)``.
    """
    set_global_seed(seed)
    layers = list(layers)
    in_dim = layers[0]
    out_dim = layers[-1]

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, in_dim)).astype("float32")
    # Smooth, low-frequency target — perfectly fit by a small KAN.
    target = np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2 if in_dim >= 2 else np.sin(np.pi * X[:, :1])
    y = np.tile(target.astype("float32"), (1, out_dim))

    print(f"[kanx.quickstart] Building KAN{layers}…", flush=True)
    print(f"[kanx.quickstart] Training {epochs} epochs on a smooth synthetic target…", flush=True)

    cfg = KanxConfig(
        model=ModelConfig(layers=layers),
        training=TrainingConfig(epochs=epochs, batch_size=64, lr=1e-2, val_split=0.0, seed=seed),
        checkpoint=CheckpointConfig(dir="checkpoints", filename="quickstart.keras"),
    )
    model, history = train(cfg, X, y, verbose=verbose)
    final_loss = history.history["loss"][-1] if history.history.get("loss") else float("nan")
    print(f"[kanx.quickstart] Final train loss: {final_loss:.6f}", flush=True)
    return model
