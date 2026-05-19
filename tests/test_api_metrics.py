
from fastapi.testclient import TestClient

from api.app import app


def test_metrics_endpoint_returns_200():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


def test_metrics_contains_kanx_counter():
    with TestClient(app) as client:
        predict = client.post(
            "/api/predict",
            json={"x": [[0.1, -0.2], [0.5, 0.6]]},
        )
        assert predict.status_code == 200

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        text = metrics.text
        assert "kanx_inference_total" in text


def test_metrics_contains_latency_histogram():
    with TestClient(app) as client:
        predict = client.post(
            "/api/predict",
            json={"x": [[0.1, -0.2], [0.5, 0.6]]},
        )
        assert predict.status_code == 200

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        text = metrics.text
        assert "kanx_inference_latency_seconds" in text
