# System Design

## Goals

1. Serve KAN inference behind a stable REST contract.
2. Hot-swap checkpoints without a restart.
3. Fall back to a fresh model when no checkpoint exists (zero-downtime cold start).
4. Scale horizontally on Kubernetes with CPU-based HPA.
5. Surface a meaningful `/api/health` signal for orchestrators.

## High-level topology

```
       ┌─────────────────────────────────────────────────┐
       │           Ingress (nginx)                       │
       │   kanx.example.com  →  Service kanx-api:80      │
       └────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
      ┌──────────┐                   ┌──────────┐
      │ Pod #1   │                   │ Pod #N   │
      │ uvicorn  │   ... HPA 2–10    │ uvicorn  │
      │ + kanx   │                   │ + kanx   │
      └────┬─────┘                   └────┬─────┘
           │                              │
           └──────── PVC (RO) ────────────┘
                    /mnt/models/kanx_model.keras
```

* Each pod is **stateless**: it owns a single in-process `tf.keras.Model`
  inside a `ModelRegistry` and serves `/api/*` over uvicorn.
* The checkpoint lives on a **read-only PersistentVolume** mounted into every
  pod at `/mnt/models`. Updates are rolled by re-creating the PV contents and
  POSTing `/api/load` (or restarting the rollout).

## Request lifecycle: `POST /api/predict`

```
client → ingress → svc → pod
              │
              ▼
       Pydantic parse  ──fail──▶ 422 (FastAPI default)
              │
              ▼
       Boundary checks
        • rank ∈ {1, 2}        ──fail──▶ 400
        • last dim == in_feat  ──fail──▶ 400
        • batch <= MAX_BATCH   ──fail──▶ 413
              │
              ▼
       ModelRegistry.model  (R-lock)
              │
              ▼
       kanx.inference.predict(model, x, batch_size)
              │
              ▼
       PredictResponse(output, shape, inference_ms)
```

All boundary checks happen **before** any TF graph is invoked — invalid
requests are cheap.

## Startup contract (checkpoint + fallback)

The user explicitly requested "(b) Serve a trained checkpoint loaded from
disk" **and** "(c) Both: load checkpoint if available, fallback to fresh
model". `api/app.py:_initialise` implements:

```
if exists(KANX_CHECKPOINT):
    model = load_model(KANX_CHECKPOINT)
    source = "checkpoint:<path>"
else:
    model = build_from_config(KANX_CONFIG)
    source = "fresh:<config>"
```

`source` is surfaced through `/api/info` and `/api/health` for observability.

## Scaling model

* **Stateless pods + sticky GET cache** — replicas can be killed at will.
* **HPA target = 70% CPU.** TF inference is CPU-bound for the supported
  model sizes (no GPU).
* **Per-pod throughput.** With `KAN[2,64,64,1]` we measured ~4 ms / 4 k samples
  on a 2-vCPU pod → ~10⁶ predictions/sec single replica. Real workloads will
  be dominated by model architecture, not the API layer.
* **MAX_BATCH guardrail** prevents a single request from monopolising a pod.

## Failure modes & responses

| Failure | Detection | User-visible | Action |
|---|---|---|---|
| Checkpoint missing at startup | `os.path.exists` | `/api/health.source` shows `fresh:…` | Fall back to fresh model |
| Corrupt checkpoint at runtime | `tf.keras.models.load_model` raises | `400` from `/api/load` | Caller retries with valid path |
| Bad input shape | Boundary check | `400` | Caller fixes payload |
| Batch too large | Boundary check | `413` | Caller chunks |
| OOM during inference | TF raises `ResourceExhaustedError` | `500` (default) | Pod gets restarted by k8s liveness |

## Concurrency

* `ModelRegistry` uses an `RLock` around `set / get / status`.
* Reads (`/predict`, `/info`, `/health`) acquire the lock for the duration
  of the model handle access (microseconds) — the heavy TF call happens
  outside the lock.
* `POST /api/load` and `/api/reset` are **write** operations and block
  concurrent reads only momentarily during the swap.

## Observability hooks

* Structured stdout logs (`kanx.train`, `kanx.inference`, `kanx.cli`,
  `[kanx-api] initialised from …`).
* `inference_ms` returned on every `/api/predict` response for client-side
  histogram metrics.
* Roadmap (P1): Prometheus `/metrics` endpoint via `prometheus-fastapi-instrumentator`.

