# REST API Reference

Base URL examples below assume `http://localhost:8000` (Docker default) or
`http://localhost:8001` (supervisor backend in this preview environment).
The contract is identical on both.

## `GET /api/health`

**Purpose.** Liveness + model load status. Use this for k8s probes.

```bash
curl http://localhost:8000/api/health
```

Response:

```json
{
  "status": "ok",
  "loaded": true,
  "source": "checkpoint:/app/checkpoints/kanx_model.keras",
  "in_features": 2,
  "out_features": 1,
  "loaded_at": 1779895412.61
}
```

`status` is `"degraded"` if no model is loaded — which should never happen
because the startup hook falls back to a fresh model from `KANX_CONFIG`.

## `GET /api/info`

**Purpose.** Version, runtime and model summary.

```bash
curl http://localhost:8000/api/info
```

```json
{
  "name": "kanx",
  "version": "0.1.0",
  "tensorflow": "2.21.0",
  "model": { "...same as /health..." },
  "max_batch": 4096
}
```

## `POST /api/predict`

**Purpose.** Run inference on one sample or a batch.

```bash
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"x": [0.1, -0.2]}'                 # single sample (1-D)

curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"x": [[0.1, -0.2], [0.5, 0.7]]}'   # batch (2-D)
```

Response:

```json
{
  "output": [[0.00169], [0.22872]],
  "shape": [2, 1],
  "inference_ms": 22.368
}
```

Error responses:

| Code | Cause |
|------|-------|
| 400  | Wrong feature count, wrong rank |
| 413  | Batch size > `KANX_MAX_BATCH` |
| 422  | Pydantic body parse failure |
| 503  | Model not initialised (shouldn't happen post-startup) |

## `POST /api/load`

**Purpose.** Hot-swap the in-process model with a checkpoint on disk.

```bash
curl -X POST http://localhost:8000/api/load \
     -H "Content-Type: application/json" \
     -d '{"path": "/app/checkpoints/kanx_model.keras"}'
```

Response: same shape as `/api/health`, reflecting the newly loaded model.

| Code | Cause |
|------|-------|
| 404  | Path does not exist |
| 400  | File exists but is not a valid Keras archive |

## `POST /api/reset`

**Purpose.** Drop the loaded checkpoint and re-initialise from `KANX_CONFIG`.

```bash
curl -X POST http://localhost:8000/api/reset
```

Useful for end-to-end smoke tests and for clearing state during canary
deployments.

## Architecture notes

* All endpoints are thread-safe via `ModelRegistry`'s `RLock`.
* `/api/predict` releases the lock before the TF call begins — the lock only
  protects the model-handle swap.
* Pydantic `PredictRequest.x` is intentionally loose (`List[float] |
  List[List[float]]`) — strict shape checking happens inside the route so
  errors are precise.

