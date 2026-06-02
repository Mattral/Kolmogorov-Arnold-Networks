# Security

## Threat model

`kanx` ships **two surfaces** with different threat profiles.

### 1. The Python library (`pip install kanx`)

Threats: malicious model checkpoints, supply-chain compromise.

* Mitigations:
  * `tf.keras.models.load_model` rejects malformed archives; we wrap with a
    `FileNotFoundError` boundary check.
  * The library pins lower bounds (`tensorflow>=2.16,<3`, `numpy>=1.23`,
    `pyyaml>=6`) and exposes optional extras for FastAPI/Pydantic. Audit via
    `pip-audit` is part of the CI roadmap.
  * `yaml.safe_load` is used everywhere — never `yaml.load`.

### 2. The REST API (`api/app.py`)

Threats: unbounded inputs, untrusted file paths, denial of service.

| Threat | Mitigation |
|---|---|
| Oversized batch | `KANX_MAX_BATCH` (default 4096) returns `413` |
| Wrong feature count | Boundary check returns `400` |
| Wrong tensor rank | Boundary check returns `400` |
| Path traversal on `/api/load` | Server validates `os.path.exists` only — operator is responsible for restricting filesystem access (run as non-root, mount RO) |
| Arbitrary YAML execution | `safe_load` only, never `load` |
| CORS | Default `*` — **override in production** (see hardening) |
| Pickle gadgets in checkpoints | Only `.keras` archives accepted; `tf.keras.models.load_model(compile=False)` does not execute optimizer state |

## Hardening checklist (production)

- [ ] Run the container as a non-root user (Dockerfile USER directive — TODO P1).
- [ ] Restrict CORS to the known client origin in `api/app.py`.
- [ ] Mount `KANX_CHECKPOINT` directory **read-only** (`k8s/deployment.yaml`
      already does this via `readOnly: true`).
- [ ] Set `KANX_MAX_BATCH` based on pod memory budget.
- [ ] Put the service behind an ingress with mTLS / OAuth2-proxy.
- [ ] Enable rate limiting at the ingress (nginx `limit_req`).
- [ ] Disable `/api/reset` and `/api/load` via reverse-proxy ACLs if
      operators don't need hot-swap (defence-in-depth).
- [ ] Drop the `--reload` flag from uvicorn in prod (the Dockerfile does so).

## Supply chain

* `pyproject.toml` declares precise lower bounds.
* `requirements.txt` is the same set, used by the Docker image.
* CI installs from a clean `pip` environment per matrix job, so anything that
  silently relies on a global install will fail.
* Future P1: SBOM via `pip-licenses` + `cyclonedx-py` published on release.

## Data handling

`kanx` does **not** log request payloads. `inference_ms` is the only
per-request telemetry surfaced. Operators who want full request/response
audit must add their own middleware.

## Secrets

`kanx` has **no secrets** in v0.1 (no API keys, no DB credentials, no model
registry tokens). All configuration is via plain env vars. If you wire in
S3-backed model loading later, fetch credentials via the cloud SDK's IAM
chain — do not bake them into the image.

