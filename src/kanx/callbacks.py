from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from .layers import KANLinear


class KANTensorBoardCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_dir: str = "logs/kanx",
        histogram_freq: int = 5,
        sample_batch: tf.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.sample_batch = sample_batch
        self._writer: tf.summary.SummaryWriter | None = None

    def on_train_begin(self, logs=None):
        self._writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if self._writer is None:
            return
        logs = logs or {}
        step = epoch + 1
        with self._writer.as_default():
            if "loss" in logs:
                tf.summary.scalar("loss", float(logs["loss"]), step=step)
            if "val_loss" in logs:
                tf.summary.scalar("val_loss", float(logs["val_loss"]), step=step)

            if self.histogram_freq > 0 and step % self.histogram_freq == 0:
                for i, layer in enumerate(self.model.layers):
                    if isinstance(layer, KANLinear):
                        tf.summary.histogram(
                            f"layer_{i}/grid",
                            layer.grid,
                            step=step,
                        )

            if self.sample_batch is not None and step % 10 == 0:
                durations = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    self.model.predict(self.sample_batch, verbose=0)
                    durations.append((time.perf_counter() - t0) * 1000.0)
                tf.summary.scalar(
                    "inference_latency_ms",
                    float(np.median(durations)),
                    step=step,
                )
        self._writer.flush()
