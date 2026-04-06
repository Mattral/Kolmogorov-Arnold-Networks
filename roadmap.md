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

## ✅ Shipped — v0.2.0 (June 2026)

### New features
- ✅ **MatrixKAN** (GPU-optimized, PyTorch): B-spline evaluation via batched GEMM instead of Cox-de Boor recursion; ~1.5–2× faster on GPU
  - `src/kanx/torch/matrix_kan.py` (MatrixKANLinear, MatrixKAN classes)
  - Numerical parity with standard KAN (within 1e-4)
  - Full test coverage: shape, numerical agreement, GPU throughput, grid updates, ONNX export
  - **CPU+GPU compatible**: Fast on GPU, performance comparable to KAN on CPU
  
- ✅ **Adaptive grid update** (both TensorFlow + PyTorch): `model.update_grid_from_samples(x, margin=0.01)`
  - Per-feature quantile-based grid recalibration from data statistics
  - Interpolation between uniform and sample-based grids via `grid_eps` parameter
  - Propagates through multi-layer models correctly
  - Differentiable in-place updates
  - **CPU+GPU compatible**: Works on both backends with same semantics
  
- ✅ **`kanx.datasets` mini-module**: Unified dataset loading
  - `load_california_housing()`, `load_concrete_strength()`, `load_energy_efficiency()` (UCI tabular)
  - Per-feature normalization to zero-mean unit-variance
  - Caching to `~/.cache/kanx/datasets/`
  - Used in `benchmarks/real_world.py` for credible baselines
  
- ✅ **GPU timing in benchmarks**: `benchmarks/compare_mlp.py` now measures inference latency on GPU
  - TensorFlow: `tf.config.list_physical_devices('GPU')` + median over 100 passes
  - PyTorch: `torch.cuda.synchronize()` + wall-clock measurement
  - Graceful CPU fallback: GPU field is N/A on CPU-only systems
  - **CPU+GPU compatible**: Works on both, adapts to available hardware
  
- ✅ **Real-world benchmark suite** with reproducible artifact:
  - `benchmarks/real_world.py`: 5-fold cross-validation on 3 UCI datasets
  - TensorFlow-only on CPU (PyTorch separately via subprocess on GPU)
  - Outputs `benchmarks/results/real_world_results.json` (committed baseline)
  - Includes train time, RMSE, R², inference latency (CPU + GPU)
  - **CPU+GPU compatible**: Runs on CPU; GPU fields populated when available

### Infrastructure & quality
- ✅ **CITATION.cff** (CFF v1.2.0 format) — machine-readable citation for academic attribution
- ✅ **SECURITY.md** — vulnerability disclosure policy, version support lifecycle, 48-hour SLA
- ✅ **Docs consolidation**: `documentations/` merged into `docs/`; single source of truth in MkDocs Material site
- ✅ **README updates**: Grid calibration best practices, MatrixKAN introduction, adaptive approach recommended
- ✅ **Quickstart.md** additions: Grid update example (TF + PyTorch), MatrixKAN section
- ✅ **System Design docs**: MatrixKAN architecture, adaptive grid implementation details, GPU considerations
- ✅ **Roadmap clarity**: Honest "Shipped v0.2.0" vs "In progress" demarcation

### Test coverage
- ✅ **7 MatrixKAN tests** (`tests/test_matrix_kan.py`): output shape, numerical agreement, GPU throughput, grid update, ONNX export
- ✅ **8 grid-update tests** (`tests/test_grid_update.py`): TensorFlow + PyTorch shape/stability/improvement
- Cumulative: **110 tests** across 10 files; **95%+ coverage** on core modules

## 🟡 In progress / Next iteration

- [ ] **CI benchmark gate** (GitHub Actions): smoke-test synthetic + real-world benchmarks on CPU
- [ ] **PyTorch subprocess benchmark wrapper**: Run PyTorch benchmarks in separate process on GPU runners to avoid TensorFlow+PyTorch segfault (CPU-only)
- [ ] **Pruning / sparsification** to drop unused edges
- [ ] **Symbolic regression** post-hoc fit per edge
- [ ] **Mixed precision** + XLA JIT on the spline einsum
- [ ] **TensorBoard callback** wired into `train()`
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

