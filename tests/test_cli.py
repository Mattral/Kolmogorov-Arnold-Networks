"""Tests for the `python -m kanx` CLI."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from kanx.__main__ import main


def test_cli_info(capsys):
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "kanx" in out
    assert "TensorFlow" in out


def test_cli_train_synthetic(tmp_path):
    # Write a tiny config that finishes in ~1s.
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model:\n  layers: [2, 8, 1]\n"
        "training:\n  epochs: 1\n  batch_size: 32\n  lr: 0.01\n  val_split: 0.0\n"
        f"checkpoint:\n  dir: {tmp_path}/ckpt\n  filename: m.keras\n"
        "  save_best_only: true\n  monitor: loss\n"
    )
    rc = main(["train", "--config", str(cfg), "--verbose", "0"])
    assert rc == 0
    assert os.path.exists(tmp_path / "ckpt" / "m.keras")


def test_cli_predict_roundtrip(tmp_path, capsys):
    # Train then predict using the CLI.
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model:\n  layers: [2, 4, 1]\n"
        "training:\n  epochs: 1\n  batch_size: 32\n  lr: 0.01\n  val_split: 0.0\n"
        f"checkpoint:\n  dir: {tmp_path}/ckpt\n  filename: m.keras\n"
        "  save_best_only: true\n  monitor: loss\n"
    )
    main(["train", "--config", str(cfg), "--verbose", "0"])

    input_path = tmp_path / "x.json"
    input_path.write_text(json.dumps([[0.1, 0.2], [0.3, 0.4]]))

    rc = main([
        "predict",
        "--checkpoint", str(tmp_path / "ckpt" / "m.keras"),
        "--input", str(input_path),
    ])
    assert rc == 0
    body = json.loads(capsys.readouterr().out.strip())
    assert body["shape"] == [2, 1]
    assert len(body["output"]) == 2


def test_cli_predict_missing_checkpoint(tmp_path, capsys):
    rc = main([
        "predict",
        "--checkpoint", str(tmp_path / "nope.keras"),
        "--input", str(tmp_path / "x.json"),
    ])
    assert rc == 2


def test_cli_unsupported_input_format(tmp_path):
    """`_load_x` should reject unknown file extensions."""
    from kanx.__main__ import _load_x
    bad = tmp_path / "x.txt"
    bad.write_text("0,0")
    with pytest.raises(ValueError):
        _load_x(str(bad))
