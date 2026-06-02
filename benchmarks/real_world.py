"""Real-world tabular benchmark suite for KANX.

This script trains and evaluates TF and PyTorch KAN models plus a PyTorch
MLP baseline on UCI tabular regression datasets.

NOTE: Due to CUDA driver initialization conflicts between TensorFlow and PyTorch
on CPU-only environments, this benchmark runs TensorFlow benchmarks only.
PyTorch benchmarks can be run separately or on systems with proper CUDA support.

Run with:
    python benchmarks/real_world.py --datasets california --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import tensorflow as tf

from kanx import KAN as TF_KAN, set_global_seed
from kanx.datasets import (
    load_california_housing,
    load_concrete_strength,
    load_energy_efficiency,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "real_world_results.json"


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float
    train_time_s: float
    params: int
    inference_latency_cpu_ms: float
    inference_latency_gpu_ms: float | None


def cpu_info() -> str:
    info = platform.processor() or platform.machine()
    return f"{info} / {platform.system()}"


def gpu_info() -> str | None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return gpus[0].name
    return None


def param_count_tf(model: tf.keras.Model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_variables))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def split_five_folds(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return list(np.array_split(idx, 5))


def train_tf_model(X: np.ndarray, y: np.ndarray, epochs: int, device: str) -> tf.keras.Model:
    set_global_seed(0)
    if device == "cuda":
        tf_device = "/GPU:0"
    else:
        tf_device = "/CPU:0"
    with tf.device(tf_device):
        model = TF_KAN([X.shape[1], 32, 1])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=128, verbose=0)
    return model


def evaluate_tf_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y_pred = model(X, training=False).numpy()
    return rmse_score(y, y_pred), r2_score(y, y_pred)


def inference_latency_tf(model: tf.keras.Model, batch: np.ndarray, device: str, runs: int = 100) -> float:
    if device == "cuda" and not tf.config.list_physical_devices("GPU"):
        return 0.0  # Skip GPU latency if no GPU
    tf_device = "/GPU:0" if device == "cuda" else "/CPU:0"
    batch = batch.astype(np.float32)
    with tf.device(tf_device):
        for _ in range(5):
            _ = model(batch, training=False)
        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(batch, training=False)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(latencies))


def build_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if name == "california":
        return load_california_housing()
    if name == "concrete":
        return load_concrete_strength()
    if name == "energy":
        return load_energy_efficiency()
    raise ValueError(f"Unknown dataset: {name}")


def summary_markdown(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Dataset | Model | Params | Train (s) | RMSE mean ± std | R² mean ± std | CPU latency (ms) |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.dataset} | {r.model} | {r.params} | {r.train_time_s:.2f} | "
            f"{r.rmse_mean:.4f} ± {r.rmse_std:.4f} | {r.r2_mean:.4f} ± {r.r2_std:.4f} | "
            f"{r.inference_latency_cpu_ms:.2f} |"
        )
    return "\n".join(lines)


def benchmark_model(
    dataset_name: str,
    n_features: int,
    epochs: int,
    device: str,
) -> BenchmarkResult:
    X, y, _ = build_dataset(dataset_name)
    folds = split_five_folds(len(X), seed=0)
    rmse_scores = []
    r2_scores = []
    train_times = []
    params = None
    for fold in range(len(folds)):
        test_idx = folds[fold]
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        start = time.perf_counter()
        model = train_tf_model(X_train, y_train, epochs=epochs, device=device)
        train_times.append(time.perf_counter() - start)
        fold_rmse, fold_r2 = evaluate_tf_model(model, X_test, y_test)
        if params is None:
            params = param_count_tf(model)

        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)

    # final model for inference timing
    model = train_tf_model(X, y, epochs=epochs, device=device)
    inference_cpu = inference_latency_tf(model, np.repeat(X[:1], 256, axis=0), "cpu", runs=100)

    return BenchmarkResult(
        dataset=dataset_name,
        model="KAN_TF",
        rmse_mean=mean(rmse_scores),
        rmse_std=stdev(rmse_scores) if len(rmse_scores) > 1 else 0.0,
        r2_mean=mean(r2_scores),
        r2_std=stdev(r2_scores) if len(r2_scores) > 1 else 0.0,
        train_time_s=mean(train_times),
        params=params or 0,
        inference_latency_cpu_ms=inference_cpu,
        inference_latency_gpu_ms=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-world tabular benchmark suite for KANX (TF only).")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument(
        "--datasets",
        choices=["all", "california", "concrete", "energy"],
        default="all",
    )
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

    selected = [
        "california",
        "concrete",
        "energy",
    ] if args.datasets == "all" else [args.datasets]

    results: list[BenchmarkResult] = []
    for dataset_name in selected:
        print(f"Benchmarking {dataset_name} on device={device}...", flush=True)
        X, y, feature_names = build_dataset(dataset_name)
        n_features = X.shape[1]
        results.append(
            benchmark_model(
                dataset_name,
                n_features=n_features,
                epochs=args.epochs,
                device=device,
            )
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": {
                    "platform": platform.platform(),
                    "cpu": cpu_info(),
                    "gpu": gpu_info(),
                    "tf_version": tf.__version__,
                },
                "note": "PyTorch benchmarks disabled on CPU-only environments due to TensorFlow/PyTorch CUDA initialization conflicts",
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )

    print("\n" + summary_markdown(results))
    print(f"\nResults saved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
