"""End-to-end live validation against the supervisor-managed backend.

Covers the explicit checklist in the review request:
  * library public-API imports
  * KAN model build via widths list AND list-of-dicts
  * /api/health, /api/info, /api/predict (single + batch), /api/load (404), /api/reset
  * boundary validation: wrong feature count -> 400, oversized batch -> 413
  * checkpoint-with-fallback contract: /api/health.source startswith 'fresh:'
  * CLI: `python -m kanx info` and `python -m kanx predict`
  * Dockerfile + k8s manifests are well-formed

The pytest suite at /app/tests/ already covers in-process unit tests; this file
hits the **live** supervisor backend at http://localhost:8001.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import requests
import yaml

BASE_URL = "http://localhost:8001"
ROOT = Path(__file__).resolve().parent.parent


# -------------------- 1. Public API imports -------------------------------
def test_public_api_imports():
    from kanx import (  # noqa: F401
        KAN,
        KANLinear,
        load_config,
        load_model,
        predict,
        save_model,
        set_global_seed,
        train,
    )


# -------------------- 2. KAN build via widths + dicts ---------------------
def test_kan_build_widths_and_dicts():
    import tensorflow as tf

    from kanx import KAN

    m1 = KAN([2, 64, 1])
    y1 = m1(tf.zeros((3, 2), dtype=tf.float32))
    assert y1.shape == (3, 1)

    m2 = KAN([
        {"in_features": 2, "out_features": 64},
        {"in_features": 64, "out_features": 1},
    ])
    y2 = m2(tf.zeros((3, 2), dtype=tf.float32))
    assert y2.shape == (3, 1)


# -------------------- 3. Live REST surface --------------------------------
def test_live_health_ok_and_fallback_source():
    r = requests.get(f"{BASE_URL}/api/health", timeout=15)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["loaded"] is True
    # Supervisor backend boots from checkpoint when /app/checkpoints/kanx_model.keras
    # exists (created by `python -m kanx train`); otherwise falls back to fresh build
    # from KANX_CONFIG. Both contracts are valid per the checkpoint-with-fallback spec.
    assert body["source"].startswith(("fresh:", "checkpoint:")), body
    assert body["in_features"] == 2
    assert body["out_features"] == 1


def test_live_info_versions():
    r = requests.get(f"{BASE_URL}/api/info", timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "kanx"
    assert "tensorflow" in body and body["tensorflow"]
    assert "version" in body and body["version"]
    assert body["max_batch"] >= 1


def test_live_predict_single():
    r = requests.post(f"{BASE_URL}/api/predict", json={"x": [0.1, -0.2]}, timeout=30)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["shape"] == [1, 1]
    assert isinstance(b["output"][0][0], float)
    assert b["inference_ms"] >= 0


def test_live_predict_batch():
    xs = np.random.RandomState(0).uniform(-1, 1, size=(8, 2)).tolist()
    r = requests.post(f"{BASE_URL}/api/predict", json={"x": xs}, timeout=30)
    assert r.status_code == 200
    assert r.json()["shape"] == [8, 1]


def test_live_predict_wrong_feature_count_returns_400():
    r = requests.post(f"{BASE_URL}/api/predict", json={"x": [0.1, 0.2, 0.3]}, timeout=15)
    assert r.status_code == 400


def test_live_predict_oversized_batch_returns_413():
    # MAX_BATCH default is 4096. Send 4097 rows of 2 features each.
    xs = np.zeros((4097, 2), dtype=np.float32).tolist()
    r = requests.post(f"{BASE_URL}/api/predict", json={"x": xs}, timeout=60)
    assert r.status_code == 413, r.status_code


def test_live_load_missing_checkpoint_returns_404():
    r = requests.post(
        f"{BASE_URL}/api/load",
        json={"path": "/tmp/__definitely_missing_kanx__.keras"},
        timeout=15,
    )
    assert r.status_code == 404


def test_live_reset_yields_fresh():
    r = requests.post(f"{BASE_URL}/api/reset", timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["source"].startswith("fresh:")


# -------------------- 4. Checkpoint roundtrip identity --------------------
def test_checkpoint_roundtrip_identical_predictions():
    import tensorflow as tf

    from kanx import KAN, load_model, predict, save_model

    tf.keras.utils.set_random_seed(123)
    model = KAN([2, 16, 1])
    X = np.random.RandomState(0).uniform(-1, 1, size=(5, 2)).astype(np.float32)
    y_before = predict(model, X)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rt.keras")
        save_model(model, path)
        assert os.path.exists(path)
        reloaded = load_model(path)
        y_after = predict(reloaded, X)

    np.testing.assert_allclose(y_before, y_after, rtol=1e-5, atol=1e-6)


# -------------------- 5. CLI subcommands ----------------------------------
def _run_cli(*args, timeout=120):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT/'src'}:{ROOT}:{env.get('PYTHONPATH','')}"
    return subprocess.run(
        [sys.executable, "-m", "kanx", *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_info():
    p = _run_cli("info")
    assert p.returncode == 0, p.stderr
    assert "kanx" in p.stdout
    assert "TensorFlow" in p.stdout


def test_cli_predict_with_trained_checkpoint(tmp_path):
    # Train a tiny model on synthetic data, save, then run CLI predict.
    import tensorflow as tf

    from kanx import KAN, save_model

    tf.keras.utils.set_random_seed(7)
    model = KAN([2, 8, 1])
    model(tf.zeros((1, 2), dtype=tf.float32))  # build
    ckpt = tmp_path / "cli_model.keras"
    save_model(model, str(ckpt))

    inp = tmp_path / "x.json"
    inp.write_text(json.dumps([[0.1, 0.2], [-0.3, 0.4]]))

    p = _run_cli("predict", "--checkpoint", str(ckpt), "--input", str(inp))
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["shape"] == [2, 1]
    assert len(out["output"]) == 2


# -------------------- 6. Dockerfile + k8s manifests -----------------------
def test_dockerfile_has_required_directives():
    text = (ROOT / "Dockerfile").read_text()
    for token in ("FROM ", "WORKDIR", "COPY", "EXPOSE", "CMD"):
        assert token in text, f"Missing {token!r} in Dockerfile"
    assert "uvicorn" in text
    assert "api.app:app" in text


@pytest.mark.parametrize("manifest", ["deployment.yaml", "service.yaml", "ingress.yaml"])
def test_k8s_manifests_are_well_formed_yaml(manifest):
    path = ROOT / "k8s" / manifest
    docs = list(yaml.safe_load_all(path.read_text()))
    assert docs, f"{manifest} produced no YAML documents"
    for d in docs:
        assert isinstance(d, dict)
        assert "apiVersion" in d and "kind" in d
