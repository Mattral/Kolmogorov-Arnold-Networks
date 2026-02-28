"""Train a KAN on MNIST. Requires `tensorflow-datasets`-free MNIST via Keras.

This is a kept-small example (3 epochs by default); for serious training,
edit `configs/mnist.yaml`.

Run:
    python examples/mnist_train.py
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from kanx.config import load_config
from kanx.train import train


def main(config_path: str = "configs/mnist.yaml") -> None:
    cfg = load_config(config_path)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0 - 0.5).reshape(-1, 784)
    y_train = y_train.astype("int32")
    train(cfg, x_train, y_train, verbose=2)


if __name__ == "__main__":
    main()
