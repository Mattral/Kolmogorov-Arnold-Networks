"""Inference utilities: checkpoint IO + batched `predict`.

The API layer (`api/app.py`) and the CLI both go through this module so that
the load / save semantics are consistent across surfaces.
"""
from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

from .layers import KANLinear  # noqa: F401  (registers custom object on import)
from .model import KAN  # noqa: F401  (registers custom object on import)
from .utils import get_logger

_LOG = get_logger("kanx.inference")

ArrayLike = np.ndarray | tf.Tensor | list


# ---------------------------------------------------------------------------
def load_model(path: str) -> tf.keras.Model:
    """Load a kanx checkpoint (`.keras` archive)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    _LOG.info("Loading model from %s", path)
    # Custom objects are auto-discovered via @register_keras_serializable.
    return tf.keras.models.load_model(path, compile=False)


def save_model(model: tf.keras.Model, path: str) -> str:
    """Persist a model to disk and return the absolute path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    abs_path = os.path.abspath(path)
    _LOG.info("Saved model to %s", abs_path)
    return abs_path


def _coerce(x: ArrayLike) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        t = x
    else:
        t = tf.convert_to_tensor(x, dtype=tf.float32)
    if t.shape.rank == 1:
        # Allow single-sample lists like [1.0, 2.0].
        t = tf.expand_dims(t, 0)
    if t.shape.rank != 2:
        raise ValueError(
            f"Inputs must be rank-1 (single sample) or rank-2 (batch); "
            f"got shape {t.shape}"
        )
    return tf.cast(t, tf.float32)


def predict(
    model: tf.keras.Model,
    x: ArrayLike,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Run inference. Returns a NumPy array of shape ``(batch, out_features)``.

    Args:
        model:      a `tf.keras.Model` (typically a `KAN`).
        x:          inputs as list / np.ndarray / tf.Tensor (rank 1 or 2).
        batch_size: optional batching for very large inputs. ``None`` => one pass.
    """
    x = _coerce(x)
    if batch_size is None or x.shape[0] <= batch_size:
        return model(x, training=False).numpy()

    out_chunks = []
    n = int(x.shape[0])
    for i in range(0, n, batch_size):
        out_chunks.append(model(x[i : i + batch_size], training=False).numpy())
    return np.concatenate(out_chunks, axis=0)
