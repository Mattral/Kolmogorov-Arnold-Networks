from pathlib import Path

import numpy as np

from kanx import train
from kanx.config import load_config


def _small_tf_data():
    X = np.random.uniform(-1, 1, size=(128, 2)).astype(np.float32)
    y = (np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2).astype(np.float32)
    return X, y


def test_tensorboard_callback_creates_log_dir(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.model.layers = [2, 8, 1]
    cfg.training.epochs = 2
    cfg.training.batch_size = 16

    X, y = _small_tf_data()
    log_dir = tmp_path / "logs"

    model, history = train(
        cfg,
        X,
        y,
        verbose=0,
        tensorboard=True,
        log_dir=str(log_dir),
    )

    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    assert log_dir.exists()
    assert event_files, f"No TensorBoard event files found in {log_dir}"


def test_tensorboard_pytorch_creates_log_dir(tmp_path):
    try:
        import torch # noqa: F401

        from kanx.torch import KAN as TorchKAN
    except ImportError:
        return

    X = np.random.uniform(-1, 1, size=(128, 2)).astype(np.float32)
    y = (np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2).astype(np.float32)

    model = TorchKAN([2, 8, 1])
    log_dir = tmp_path / "logs"
    trainer = model.fit( # noqa: F841
        X,
        y,
        epochs=2,
        batch_size=16,
        lr=1e-3,
        tensorboard=True,
        log_dir=str(log_dir),
        verbose=0,
    )

    event_files = list(Path(log_dir).glob("**/events.out.tfevents.*"))
    assert Path(log_dir).exists()
    assert event_files, f"No TensorBoard event files found in {log_dir}"
