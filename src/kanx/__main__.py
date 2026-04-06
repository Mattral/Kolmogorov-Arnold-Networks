"""CLI entrypoint: ``python -m kanx --config configs/default.yaml``.

Subcommands:
    train     — train a model from a YAML config and a CSV / .npz dataset
    predict   — load a checkpoint and predict on a CSV / .npz / JSON input
    info      — print the package version and TF / device info
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np

from . import __version__
from .config import load_config
from .inference import load_model, predict
from .train import train
from .utils import get_logger, resolve_checkpoint_path

_LOG = get_logger("kanx.cli")


# ---------------------------------------------------------------------------
def _load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from either a .npz file with keys X/y, or a CSV where the
    last column is the target. Used by `kanx train`."""
    if path.endswith(".npz"):
        data = np.load(path)
        return data["X"].astype(np.float32), data["y"].astype(np.float32)
    if path.endswith(".csv"):
        arr = np.genfromtxt(path, delimiter=",", dtype=np.float32, skip_header=1)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"CSV at {path} must have >= 2 columns")
        return arr[:, :-1], arr[:, -1:]
    raise ValueError(f"Unsupported dataset format: {path} (use .npz or .csv)")


def _load_x(path: str) -> np.ndarray:
    if path.endswith(".json"):
        with open(path, "r") as f:
            return np.asarray(json.load(f), dtype=np.float32)
    if path.endswith(".npz"):
        return np.load(path)["X"].astype(np.float32)
    if path.endswith(".csv"):
        return np.genfromtxt(path, delimiter=",", dtype=np.float32, skip_header=1)
    raise ValueError(f"Unsupported input format: {path}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kanx")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="train a KAN model")
    p_train.add_argument("--config", default="configs/default.yaml")
    p_train.add_argument(
        "--data", required=False,
        help="Path to dataset (.npz with X/y or .csv with y in last column). "
             "If omitted a small synthetic regression dataset is used.",
    )
    p_train.add_argument("--verbose", type=int, default=1)
    p_train.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard logs to the configured log directory.",
    )
    p_train.add_argument(
        "--log-dir",
        default="logs/kanx",
        help="Directory for TensorBoard logs when --tensorboard is enabled.",
    )

    p_pred = sub.add_parser("predict", help="run inference on a checkpoint")
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--input", required=True)

    sub.add_parser("info", help="print version / runtime info")

    # default subcommand → train
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0].startswith("--"):
        argv = ["train"] + argv

    args = parser.parse_args(argv)

    if args.cmd == "info":
        import tensorflow as tf  # noqa
        print(f"kanx {__version__}")
        print(f"TensorFlow {tf.__version__}")
        print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
        return 0

    if args.cmd == "train":
        cfg = load_config(args.config)
        if args.data:
            X, y = _load_xy(args.data)
        else:
            # Synthetic regression: target = sin(pi*x1) + x2^2
            rng = np.random.default_rng(cfg.training.seed)
            X = rng.uniform(-1, 1, size=(512, cfg.model.layers[0])).astype(np.float32)
            y = (np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2).astype(np.float32)
            if cfg.model.layers[-1] != 1:
                # Broadcast to output dim by repeating the column.
                y = np.tile(y, (1, cfg.model.layers[-1]))
        _LOG.info("Loaded dataset: X=%s y=%s", X.shape, y.shape)
        train(
            cfg,
            X,
            y,
            verbose=args.verbose,
            tensorboard=args.tensorboard,
            log_dir=args.log_dir,
        )
        ckpt = resolve_checkpoint_path(cfg.checkpoint.dir, cfg.checkpoint.filename)
        _LOG.info("Done. Checkpoint: %s", ckpt)
        return 0

    if args.cmd == "predict":
        if not os.path.exists(args.checkpoint):
            _LOG.error("Checkpoint does not exist: %s", args.checkpoint)
            return 2
        model = load_model(args.checkpoint)
        X = _load_x(args.input)
        out = predict(model, X)
        print(json.dumps({"shape": list(out.shape), "output": out.tolist()}))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
