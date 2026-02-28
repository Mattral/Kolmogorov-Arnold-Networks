"""Tests for the optional API-key auth + rate-limit hardening."""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def auth_client(monkeypatch):
    """Reload api.app with KANX_API_KEY enabled."""
    monkeypatch.setenv("KANX_API_KEY", "s3cret-token")
    monkeypatch.setenv("KANX_RATE_LIMIT_RPM", "0")
    monkeypatch.setenv("KANX_CHECKPOINT", "/tmp/__nope__.keras")
    import api.app as app_mod
    importlib.reload(app_mod)
    with TestClient(app_mod.app) as c:
        yield c


@pytest.fixture
def rate_limited_client(monkeypatch):
    monkeypatch.setenv("KANX_API_KEY", "")
    monkeypatch.setenv("KANX_RATE_LIMIT_RPM", "3")
    monkeypatch.setenv("KANX_CHECKPOINT", "/tmp/__nope__.keras")
    import api.app as app_mod
    importlib.reload(app_mod)
    with TestClient(app_mod.app) as c:
        yield c


def test_api_key_rejected_without_header(auth_client):
    r = auth_client.post("/api/predict", json={"x": [0.1, 0.2]})
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["detail"]


def test_api_key_accepted_with_header(auth_client):
    r = auth_client.post(
        "/api/predict",
        headers={"X-API-Key": "s3cret-token"},
        json={"x": [0.1, 0.2]},
    )
    assert r.status_code == 200
    assert r.json()["shape"] == [1, 1]


def test_api_key_wrong_value(auth_client):
    r = auth_client.post(
        "/api/predict",
        headers={"X-API-Key": "wrong"},
        json={"x": [0.1, 0.2]},
    )
    assert r.status_code == 401


def test_health_does_not_require_api_key(auth_client):
    """Probes must always succeed for k8s readiness/liveness."""
    assert auth_client.get("/api/health").status_code == 200
    assert auth_client.get("/api/info").status_code == 200


def test_rate_limit_returns_429_after_burst(rate_limited_client):
    # Limit = 3 RPM; 4th request in the same window should 429.
    ok = 0
    rate_limited = 0
    for _ in range(6):
        r = rate_limited_client.post("/api/predict", json={"x": [0.1, 0.2]})
        if r.status_code == 200:
            ok += 1
        elif r.status_code == 429:
            rate_limited += 1
    assert ok == 3
    assert rate_limited >= 1


def teardown_module():
    """Reset env so other test modules pick up the original (unsecured) app."""
    for k in ("KANX_API_KEY", "KANX_RATE_LIMIT_RPM"):
        os.environ.pop(k, None)
    import api.app as app_mod
    importlib.reload(app_mod)
