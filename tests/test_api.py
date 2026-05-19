"""End-to-end FastAPI tests using `TestClient`."""
from __future__ import annotations

import os

import numpy as np
from fastapi.testclient import TestClient

# Ensure the API picks up a fresh model on import (no checkpoint required).
os.environ.setdefault("KANX_CHECKPOINT", "/tmp/__nonexistent__.keras")

from api.app import app  # noqa: E402

client = TestClient(app)


def test_health_returns_ok():
    with TestClient(app) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["loaded"] is True
        assert body["in_features"] == 2  # from configs/default.yaml


def test_info_returns_versions():
    with TestClient(app) as c:
        r = c.get("/api/info")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "kanx"
        assert "tensorflow" in body
        assert "version" in body


def test_predict_single_sample():
    with TestClient(app) as c:
        r = c.post("/api/predict", json={"x": [0.1, -0.2]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["shape"] == [1, 1]
        assert len(body["output"]) == 1
        assert isinstance(body["output"][0][0], float)


def test_predict_batch():
    with TestClient(app) as c:
        xs = np.random.RandomState(0).uniform(-1, 1, size=(4, 2)).tolist()
        r = c.post("/api/predict", json={"x": xs})
        assert r.status_code == 200
        body = r.json()
        assert body["shape"] == [4, 1]


def test_predict_wrong_feature_count():
    with TestClient(app) as c:
        r = c.post("/api/predict", json={"x": [0.1, 0.2, 0.3]})
        assert r.status_code == 400


def test_load_missing_checkpoint():
    with TestClient(app) as c:
        r = c.post("/api/load", json={"path": "/tmp/__definitely_missing__.keras"})
        assert r.status_code == 404


def test_reset_yields_fresh_model():
    with TestClient(app) as c:
        r = c.post("/api/reset")
        assert r.status_code == 200
        assert r.json()["source"].startswith("fresh:")
