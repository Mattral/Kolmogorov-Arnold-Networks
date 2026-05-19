"""Training pipeline for kanx.

Exposes a single ergonomic `train(config, X, y)` function that:
    1. seeds RNGs deterministically,
    2. builds a `KAN` model from the parsed `KanxConfig`,
    3. compiles it with the requested optimizer + loss,
    4. fits it on `(X, y)` with optional validation split + early stopping,
    5. persists the final / best checkpoint to disk,
    6. returns ``(model, history)``.

The function is intentionally framework-canonical: it uses
``tf.keras.Model.fit`` under the hood so that any Keras callback / TensorBoard
integration / mixed precision setup works out of the box.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import tensorflow as tf

from .callbacks import KANTensorBoardCallback
from .config import KanxConfig, validate_config
from .model import KAN
from .utils import ensure_dir, get_logger, resolve_checkpoint_path, set_global_seed

_LOG = get_logger("kanx.train")


# ---------------------------------------------------------------------------
def build_optimizer(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {name!r}")


def build_loss(name: str):
    name = name.lower()
    if name in ("mse", "mean_squared_error"):
        return tf.keras.losses.MeanSquaredError()
    if name in ("mae", "mean_absolute_error"):
        return tf.keras.losses.MeanAbsoluteError()
    if name in ("sparse_categorical_crossentropy", "scce"):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if name in ("binary_crossentropy", "bce"):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    raise ValueError(f"Unsupported loss: {name!r}")


# ---------------------------------------------------------------------------
def train(
    config: KanxConfig | dict[str, Any],
    X: np.ndarray | tf.Tensor,
    y: np.ndarray | tf.Tensor,
    *,
    verbose: int = 1,
    extra_callbacks: list | None = None,
    tensorboard: bool = False,
    log_dir: str = "logs/kanx",
) -> tuple[KAN, tf.keras.callbacks.History]:
    """Train a KAN model and persist the best checkpoint."""
    if isinstance(config, dict):
        config = validate_config(config)

    set_global_seed(config.training.seed)

    model = KAN(
        config.model.layers,
        grid_size=config.model.grid_size,
        spline_order=config.model.spline_order,
        base_activation=config.model.base_activation,
        regularization_factor=config.model.regularization_factor,
        grid_range=tuple(config.model.grid_range),
    )

    optimizer = build_optimizer(config.training.optimizer, config.training.lr)
    loss_fn = build_loss(config.training.loss)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # ---- callbacks ---------------------------------------------------------
    callbacks: list = []
    ensure_dir(config.checkpoint.dir)
    ckpt_path = resolve_checkpoint_path(
        config.checkpoint.dir, config.checkpoint.filename
    )
    monitor = (
        f"val_{config.checkpoint.monitor}"
        if config.training.val_split > 0
        and not config.checkpoint.monitor.startswith("val_")
        else config.checkpoint.monitor
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=config.checkpoint.save_best_only,
            save_weights_only=False,
            monitor=monitor,
            mode="min",
            verbose=verbose,
        )
    )

    if config.training.early_stopping_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=config.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        )
    if tensorboard:
        sample_batch = None
        if X is not None:
            sample_batch = tf.convert_to_tensor(X[:256], dtype=tf.float32)
        callbacks.append(
            KANTensorBoardCallback(
                log_dir=log_dir,
                histogram_freq=5,
                sample_batch=sample_batch,
            )
        )

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    _LOG.info(
        "Training KAN layers=%s grid=%d order=%d epochs=%d batch=%d",
        config.model.layers,
        config.model.grid_size,
        config.model.spline_order,
        config.training.epochs,
        config.training.batch_size,
    )

    history = model.fit(
        X,
        y,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        validation_split=config.training.val_split,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Always also write the final-state model so inference always works,
    # even when save_best_only never triggered (e.g. NaN losses).
    if not os.path.exists(ckpt_path):
        model.save(ckpt_path)
        _LOG.info("Saved final model to %s", ckpt_path)
    else:
        _LOG.info("Best model checkpoint at %s", ckpt_path)
    return model, history
