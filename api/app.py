"""kanx FastAPI service.

Endpoints
---------
GET  /api/health        — liveness probe, returns model load status
GET  /api/info          — package + TF version + model summary
POST /api/predict       — run inference on a single sample or batch
POST /api/load          — (re)load a checkpoint from disk
POST /api/reset         — drop checkpoint, fall back to fresh model

Security knobs (all optional, controlled by env vars)
------------------------------------------------------
* ``KANX_API_KEY``         — if set, all ``/api/*`` (except /health and /info)
                             require ``X-API-Key`` header to match.
* ``KANX_RATE_LIMIT_RPM``  — per-IP requests-per-minute limit; 0 disables.
                             Default 0 (off) for backwards compatibility.

Concurrency
-----------
* TF predict is run in the threadpool via FastAPI's sync-route handling
  (FastAPI dispatches `def` routes to a worker thread, so the asyncio
  event loop stays responsive). For very long inferences, consider an
  explicit `asyncio.to_thread` wrapper — not needed at the current sizes.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import tensorflow as tf  # noqa: F401  (ensures TF is initialised before model load)
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from kanx import KAN, __version__
from kanx.config import load_config
from kanx.inference import load_model, predict

# ---------------------------------------------------------------------------
# Configuration via environment variables (12-factor)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = os.environ.get(
    "KANX_CONFIG",
    str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
)
DEFAULT_CHECKPOINT = os.environ.get(
    "KANX_CHECKPOINT",
    str(Path(__file__).resolve().parent.parent / "checkpoints" / "kanx_model.keras"),
)
MAX_BATCH = int(os.environ.get("KANX_MAX_BATCH", "4096"))
API_KEY = os.environ.get("KANX_API_KEY", "").strip()  # "" disables auth
RATE_LIMIT_RPM = int(os.environ.get("KANX_RATE_LIMIT_RPM", "0"))  # 0 disables


# ---------------------------------------------------------------------------
# Thread-safe model registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: tf.keras.Model | None = None
        self._source: str = "uninitialized"
        self._in_features: int | None = None
        self._out_features: int | None = None
        self._loaded_at: float = 0.0

    def set(self, model: tf.keras.Model, source: str) -> None:
        with self._lock:
            self._model = model
            self._source = source
            # Best-effort shape introspection: build with a dummy if needed.
            try:
                cfg = model.get_config()
                # KAN config carries layers.
                layers = cfg.get("layers")
                if isinstance(layers, list) and layers and isinstance(layers[0], int):
                    self._in_features = int(layers[0])
                    self._out_features = int(layers[-1])
                elif isinstance(layers, list) and layers and isinstance(layers[0], dict):
                    self._in_features = int(layers[0]["in_features"])
                    self._out_features = int(layers[-1]["out_features"])
            except Exception:
                pass
            self._loaded_at = time.time()

    @property
    def model(self) -> tf.keras.Model:
        with self._lock:
            if self._model is None:
                raise HTTPException(status_code=503, detail="Model not initialised")
            return self._model

    def status(self) -> dict:
        with self._lock:
            return {
                "loaded": self._model is not None,
                "source": self._source,
                "in_features": self._in_features,
                "out_features": self._out_features,
                "loaded_at": self._loaded_at,
            }


REGISTRY = ModelRegistry()


def _build_fresh_from_config(config_path: str) -> tf.keras.Model:
    cfg = load_config(config_path)
    model = KAN(
        cfg.model.layers,
        grid_size=cfg.model.grid_size,
        spline_order=cfg.model.spline_order,
        base_activation=cfg.model.base_activation,
        regularization_factor=cfg.model.regularization_factor,
        grid_range=tuple(cfg.model.grid_range),
    )
    # Build with a dummy input so weights exist.
    model(tf.zeros((1, cfg.model.layers[0]), dtype=tf.float32))
    return model


kanx_inference_total = Counter(
    "kanx_inference_total",
    "Total successful inference requests handled by kanx.",
    ["backend", "batch_size"],
)
kanx_inference_latency_seconds = Histogram(
    "kanx_inference_latency_seconds",
    "Inference latency for /api/predict in seconds.",
    ["backend", "batch_size"],
)


def _bucket_batch_size(batch_size: int) -> str:
    if batch_size <= 1:
        return "1"
    if batch_size <= 10:
        return "2-10"
    if batch_size <= 100:
        return "11-100"
    return "101+"


def _initialise(checkpoint: str, config: str) -> str:
    """Try checkpoint, fall back to fresh model. Returns the source label."""
    if checkpoint and os.path.exists(checkpoint):
        model = load_model(checkpoint)
        REGISTRY.set(model, source=f"checkpoint:{checkpoint}")
        return f"checkpoint:{checkpoint}"
    model = _build_fresh_from_config(config)
    REGISTRY.set(model, source=f"fresh:{config}")
    return f"fresh:{config}"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    # Accept either a single sample [f1, f2, ...] or a batch [[..], [..]].
    x: list[float] | list[list[float]] = Field(
        ..., description="Single sample (1-D list) or batch (2-D list)."
    )


class PredictResponse(BaseModel):
    output: list[list[float]]
    shape: list[int]
    inference_ms: float


class LoadRequest(BaseModel):
    path: str = Field(..., description="Filesystem path to a .keras checkpoint.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def _startup_event() -> None:
    src = _initialise(DEFAULT_CHECKPOINT, DEFAULT_CONFIG)
    print(f"[kanx-api] initialised from {src}", flush=True)


app = FastAPI(
    title="kanx — Kolmogorov-Arnold Network Inference API",
    description=(
        "Production REST surface for the kanx KAN library. "
        "Loads a checkpoint at startup with fallback to a fresh model."
    ),
    version=__version__,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.on_event("startup")
def startup() -> None:
    _startup_event()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Optional API-key + token-bucket rate limit (per-IP, in-memory)
# ---------------------------------------------------------------------------
_RATE_BUCKETS: dict[str, list[float]] = {}
_RATE_LOCK = threading.Lock()


def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _check_rate_limit(request: Request) -> None:
    if RATE_LIMIT_RPM <= 0:
        return
    client = request.client.host if request.client else "anon"
    now = time.time()
    window = 60.0
    with _RATE_LOCK:
        bucket = _RATE_BUCKETS.setdefault(client, [])
        # Drop timestamps older than the 60s window.
        cutoff = now - window
        bucket[:] = [t for t in bucket if t > cutoff]
        if len(bucket) >= RATE_LIMIT_RPM:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({RATE_LIMIT_RPM}/min)",
            )
        bucket.append(now)


# ---- routes ---------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    s = REGISTRY.status()
    return {"status": "ok" if s["loaded"] else "degraded", **s}


@app.get("/api/info")
def info() -> dict:
    s = REGISTRY.status()
    return {
        "name": "kanx",
        "version": __version__,
        "tensorflow": tf.__version__,
        "model": s,
        "max_batch": MAX_BATCH,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict_route(
    req: PredictRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> PredictResponse:
    _check_api_key(x_api_key)
    _check_rate_limit(request)
    model = REGISTRY.model
    s = REGISTRY.status()
    # Normalise to 2-D batch.
    arr = np.asarray(req.x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"x must be 1-D or 2-D; got shape {list(arr.shape)}",
        )
    if s["in_features"] is not None and arr.shape[1] != s["in_features"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {s['in_features']} features per sample, "
                f"got {arr.shape[1]}"
            ),
        )
    if arr.shape[0] > MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {arr.shape[0]} > MAX_BATCH={MAX_BATCH}",
        )

    batch_bucket = _bucket_batch_size(arr.shape[0])
    t0 = time.perf_counter()
    out = predict(model, arr, batch_size=min(MAX_BATCH, arr.shape[0]))
    dt = (time.perf_counter() - t0) * 1000.0

    kanx_inference_total.labels(backend="tf", batch_size=batch_bucket).inc()
    kanx_inference_latency_seconds.labels(
        backend="tf", batch_size=batch_bucket
    ).observe(dt / 1000.0)

    return PredictResponse(
        output=out.tolist(),
        shape=list(out.shape),
        inference_ms=round(dt, 3),
    )


@app.post("/api/load")
def load_route(
    req: LoadRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    _check_api_key(x_api_key)
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {req.path}")
    try:
        model = load_model(req.path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load: {exc!s}")
    REGISTRY.set(model, source=f"checkpoint:{req.path}")
    return {"status": "ok", **REGISTRY.status()}


@app.post("/api/reset")
def reset_route(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    _check_api_key(x_api_key)
    model = _build_fresh_from_config(DEFAULT_CONFIG)
    REGISTRY.set(model, source=f"fresh:{DEFAULT_CONFIG}")
    return {"status": "ok", **REGISTRY.status()}
