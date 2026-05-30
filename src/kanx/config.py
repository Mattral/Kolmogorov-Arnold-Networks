"""Configuration loader and schema validator for kanx.

We deliberately avoid pulling in heavyweight schema libraries (pydantic) —
configs are simple nested dicts validated by a hand-written `validate_config`
function. This keeps the dependency footprint tiny and the failure modes
loud and obvious (a single ValueError with a precise message).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class ModelConfig:
    layers: List[int]
    grid_size: int = 5
    spline_order: int = 3
    base_activation: str = "silu"
    grid_range: Tuple[float, float] = (-1.0, 1.0)
    regularization_factor: float = 0.0


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    loss: str = "mse"
    optimizer: str = "adam"
    val_split: float = 0.0
    early_stopping_patience: int = 0   # 0 = disabled
    seed: int = 42


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    filename: str = "kanx_model.keras"
    save_best_only: bool = True
    monitor: str = "loss"


@dataclass
class KanxConfig:
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    raw: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
def load_config(path: str) -> KanxConfig:
    """Load a YAML config from disk and validate it.

    Raises:
        FileNotFoundError if `path` does not exist.
        ValueError       on invalid schema / values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return validate_config(raw)


def validate_config(raw: Dict[str, Any]) -> KanxConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    # ---- model ------------------------------------------------------------
    if "model" not in raw or not isinstance(raw["model"], dict):
        raise ValueError("Config must contain a 'model' mapping")
    m = raw["model"]
    if "layers" not in m or not isinstance(m["layers"], list) or len(m["layers"]) < 2:
        raise ValueError("model.layers must be a list of >= 2 positive integers")
    for w in m["layers"]:
        if not isinstance(w, int) or w <= 0:
            raise ValueError(f"model.layers entries must be positive ints, got {w!r}")

    grid_range = tuple(m.get("grid_range", (-1.0, 1.0)))
    if len(grid_range) != 2 or grid_range[0] >= grid_range[1]:
        raise ValueError("model.grid_range must be [low, high] with low < high")

    model_cfg = ModelConfig(
        layers=list(m["layers"]),
        grid_size=int(m.get("grid_size", 5)),
        spline_order=int(m.get("spline_order", 3)),
        base_activation=str(m.get("base_activation", "silu")),
        grid_range=(float(grid_range[0]), float(grid_range[1])),
        regularization_factor=float(m.get("regularization_factor", 0.0)),
    )
    if model_cfg.grid_size <= 0:
        raise ValueError("model.grid_size must be > 0")
    if model_cfg.spline_order < 1:
        raise ValueError("model.spline_order must be >= 1")

    # ---- training ---------------------------------------------------------
    t = raw.get("training", {}) or {}
    train_cfg = TrainingConfig(
        lr=float(t.get("lr", 1e-3)),
        epochs=int(t.get("epochs", 50)),
        batch_size=int(t.get("batch_size", 32)),
        loss=str(t.get("loss", "mse")),
        optimizer=str(t.get("optimizer", "adam")),
        val_split=float(t.get("val_split", 0.0)),
        early_stopping_patience=int(t.get("early_stopping_patience", 0)),
        seed=int(t.get("seed", 42)),
    )
    if train_cfg.lr <= 0:
        raise ValueError("training.lr must be > 0")
    if train_cfg.epochs <= 0:
        raise ValueError("training.epochs must be > 0")
    if train_cfg.batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if not (0.0 <= train_cfg.val_split < 1.0):
        raise ValueError("training.val_split must be in [0, 1)")

    # ---- checkpoint -------------------------------------------------------
    c = raw.get("checkpoint", {}) or {}
    ckpt_cfg = CheckpointConfig(
        dir=str(c.get("dir", "checkpoints")),
        filename=str(c.get("filename", "kanx_model.keras")),
        save_best_only=bool(c.get("save_best_only", True)),
        monitor=str(c.get("monitor", "loss")),
    )

    return KanxConfig(
        model=model_cfg,
        training=train_cfg,
        checkpoint=ckpt_cfg,
        raw=raw,
    )
