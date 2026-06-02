# kanx Roadmap

Single source of truth for what's shipped and what's next.

## ✅ Shipped — v0.1.0 (May 2026)

### Library — TensorFlow backend (primary)
- ✅ `src/kanx/` (`layers`, `model`, `train`, `inference`, `config`, `utils`, `__main__`, `onnx_export`)
- ✅ Vectorized Cox-de Boor B-spline (no Python loops)
- ✅ `KANLinear` with SiLU residual + per-feature grids (Liu et al. 2024)
- ✅ `@register_keras_serializable` → safe `save_model`/`load_model`
- ✅ CLI: `python -m kanx {info,train,predict}`

### Library — PyTorch backend (parallel surface)
- ✅ `src/kanx/torch/` (`layers`, `model`, `trainer`, `onnx_export`)
- ✅ Native `torch.nn.Module` integration (autograd, DataLoader, DDP-ready)
- ✅ `Trainer` mirrors `kanx.train` semantics for one-liner training
- ✅ `KAN.save()`/`KAN.load()` checkpoint format

### ONNX export
- ✅ TF → ONNX via tf2onnx (`kanx.export_onnx_tf`)
- ✅ PyTorch → ONNX native (`kanx.torch.export_onnx`)
- ✅ Dynamic batch axis
- ✅ Numerical-parity tests (within 1e-5 of eager)

### Serving
- ✅ FastAPI service (`api/app.py`) with thread-safe `ModelRegistry`
- ✅ Endpoints: `/api/health`, `/api/info`, `/api/predict`, `/api/load`, `/api/reset`
- ✅ Lifespan-based startup with checkpoint-with-fallback contract
- ✅ Boundary validation (400 / 413 / 404 at right places)
- ✅ Supervisor-managed deployment via `backend/server.py`

### Quality
- ✅ **95 tests** across 8 files — unit, integration, E2E, property-based, performance
- ✅ **94% library coverage** (99% layers, 100% model)
- ✅ Hypothesis property tests (partition of unity, shape invariants, gradient finiteness)
- ✅ Performance regression alarms (latency budgets on forward + predict)
- ✅ Numerical-contract tests: partition of unity, non-negativity, save/load roundtrip, ONNX parity
- ✅ TF + Torch coexistence (single-threaded BLAS via `conftest.py`)

### Infra & docs
- ✅ Dockerfile + docker-compose
- ✅ Kubernetes manifests (Deployment + Service + Ingress + HPA + PVC)
- ✅ GitHub Actions CI: matrix py3.10/3.11/3.12 + lint + Docker smoke + MkDocs build
- ✅ Release pipeline (`release.yml`): PyPI (OIDC) + GHCR Docker push + GitHub Release + MkDocs gh-deploy
- ✅ **MkDocs Material site** (`mkdocs.yml` + `docs/`) — 12 pages with code-tabs, dark/light, search
- ✅ `documentations/` — 8 long-form docs (philosophy, architecture, system_design, build, security, api, testing, deployment)
- ✅ `CHANGELOG.md` (Keep-a-Changelog format)
- ✅ `notebooks/quickstart.ipynb` — Colab-ready "Train KAN in 2 minutes"
- ✅ `notebooks/LAUNCH_POST.md` — community launch copy
- ✅ Benchmark chart (`docs/assets/benchmark.png`) — KAN[2,32,1] beats MLP[2,64,64,1] by ~265× MSE with 5× fewer params

## 🟡 In progress / Next iteration

- [ ] **Adaptive grid update** (pykan-style `update_grid_from_samples`)
- [ ] **Pruning / sparsification** to drop unused edges
- [ ] **Symbolic regression** post-hoc fit per edge
- [ ] **Mixed precision** + XLA JIT on the spline einsum
- [ ] **TensorBoard callback** wired into `train()`
- [ ] **`kanx.datasets`** mini-module (Feynman, UCI tabular)
- [ ] **HuggingFace Hub integration** — `KAN.from_pretrained("user/model")`

## 🔵 Backlog (P1)

- [ ] Profiling: replace per-edge einsum with fused custom op
- [ ] Multi-GPU / `tf.distribute.MirroredStrategy` + `torch.distributed` CI smoke
- [ ] Bayesian / dropout KAN variant for uncertainty estimation
- [ ] Helm chart for `k8s/` (parameterised values)
- [ ] Prometheus `/metrics` endpoint on FastAPI
- [ ] gRPC serving alongside REST
- [ ] JAX backend (`kanx.jax`) as a third parallel surface

## 🟣 Backlog (P2)

- [ ] Interactive visualisation of learned edge functions (`kanx.viz`)
- [ ] Reproduce subset of Liu et al. (2024) benchmarks (Feynman dataset)
- [ ] Sphinx auto-API reference
- [ ] Triton inference server adapter
- [ ] Quantization-aware training pass
- [ ] Streaming inference endpoint (`/api/predict-stream` via SSE/WS)

## 📜 Decision log

- **TensorFlow primary + PyTorch secondary** — TF matches the upstream repo;
  Torch added because it's where most ML research happens (~80% of new papers).
- **ONNX as the deployment lingua-franca** — instead of integrating with each
  serving runtime, export to ONNX once and let the user pick (TensorRT,
  OpenVINO, ONNX Runtime, CoreML, …).
- **Per-feature grids** — required for adaptive grid updates (pykan parity).
- **Hand-rolled config validator** — no pydantic dep in the core lib.
- **`save_best_only=True` + final-model fallback** — inference always works.
- **Single-threaded BLAS in tests** — only way to keep TF+Torch hermetic in
  one process. Production code does not impose this.

