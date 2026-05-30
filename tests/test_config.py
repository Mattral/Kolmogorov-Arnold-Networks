"""Unit tests for `kanx.config`."""
from __future__ import annotations

import os
import textwrap

import pytest

from kanx.config import load_config, validate_config


def test_validate_config_defaults():
    cfg = validate_config({"model": {"layers": [2, 16, 1]}})
    assert cfg.model.layers == [2, 16, 1]
    assert cfg.model.grid_size == 5
    assert cfg.model.spline_order == 3
    assert cfg.training.epochs == 50
    assert cfg.checkpoint.dir == "checkpoints"


def test_validate_config_full():
    cfg = validate_config({
        "model": {
            "layers": [4, 32, 1],
            "grid_size": 7,
            "spline_order": 2,
            "grid_range": [-2.0, 2.0],
            "regularization_factor": 1e-3,
        },
        "training": {
            "lr": 5e-4, "epochs": 10, "batch_size": 64,
            "loss": "mae", "optimizer": "adamw",
            "val_split": 0.2, "early_stopping_patience": 3,
        },
        "checkpoint": {"dir": "ckpts", "filename": "m.keras"},
    })
    assert cfg.model.grid_size == 7
    assert cfg.training.optimizer == "adamw"
    assert cfg.checkpoint.dir == "ckpts"


@pytest.mark.parametrize("bad", [
    {},
    {"model": "nope"},
    {"model": {"layers": [1]}},               # too few layers
    {"model": {"layers": [2, -1]}},           # negative
    {"model": {"layers": [2, 1.5]}},          # non-int
    {"model": {"layers": [2, 1], "grid_size": 0}},
    {"model": {"layers": [2, 1], "spline_order": 0}},
    {"model": {"layers": [2, 1], "grid_range": [1.0, -1.0]}},
    {"model": {"layers": [2, 1]}, "training": {"lr": -1}},
    {"model": {"layers": [2, 1]}, "training": {"epochs": 0}},
    {"model": {"layers": [2, 1]}, "training": {"val_split": 1.0}},
])
def test_validate_config_rejects_invalid(bad):
    with pytest.raises(ValueError):
        validate_config(bad)


def test_load_config_from_disk(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        model:
          layers: [2, 8, 1]
        training:
          epochs: 1
    """))
    cfg = load_config(str(p))
    assert cfg.training.epochs == 1


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
