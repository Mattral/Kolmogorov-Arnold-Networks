# Changelog

All notable changes to **kanx** are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2026-05-30

This release closes a FAANG-grade audit. Six concrete corrections:

### Added
- **`kanx.fit_grid_to_data(model, X, pad=0.05)`** — calibrates every
  `KANLinear` layer's B-spline grid to the observed input range. **Fixes
  the #1 KAN production bug** where inputs outside the default `[-1, 1]`
  grid silently zero out the spline path. Works on both TF and PyTorch
  backends.
- **`kanx.check_input_range(model, X)`** — logs a `WARNING` when input
  exceeds the model's grid range. Use as an inference-time sanity check.
- **Prominent README warning** about the grid-range gotcha with copy-paste fix.
- **Four new example scripts** in `examples/`:
  `tabular_regression.py` (8-D, non-unit range, the correct training pattern),
  `classification.py` (moons, SparseCategoricalCrossentropy),
  `time_series.py` (sliding-window autoregressive forecast),
  `visualize_edges.py` (plot the learned 1-D edge functions),
  `onnx_pipeline.py` (train → save → ONNX export → ONNXRuntime).
- **Optional API hardening** (env-controlled, off by default for backwards-compat):
  `KANX_API_KEY` enables `X-API-Key`-header auth on `/api/predict`,
  `/api/load`, `/api/reset`. `KANX_RATE_LIMIT_RPM` enables per-IP
  token-bucket rate limiting (returns `429`). Health/info remain open
  for k8s probes.
- **Branch coverage** enabled (`[tool.coverage.run] branch = true`) — line
  coverage alone is no longer the only signal.

### Fixed
- **Honest, multi-baseline benchmark.** `benchmarks/compare_mlp.py` now
  trains for **100 epochs** (not 30) and compares against three MLP
  baselines including **parameter-matched ones** (MLP[2,32,1] / 129 params
  and MLP[2,16,16,1] / 337 params), not just an overparameterised
  MLP[2,64,64,1]. The deceptive "265× MSE win" headline is gone; the new
  fair number is ~75× vs param-matched MLP and ~25× vs an MLP 10× larger
  — on a best-case smooth separable target. `results.md` now documents
  caveats prominently and `--long` mode adds 1000-epoch + early-stopping
  convergence runs.
- **Repo hygiene.** Added `.gitignore` entries for `.coverage`,
  `.coverage.*`, `htmlcov/`, `coverage.xml`, `.benchmarks/`,
  `.hypothesis/`, `test_reports/`, `site/`, `.emergent/`, `memory/`,
  `checkpoints/`, `*.keras`, `*.pt`, `*.onnx`. **These files must never be
  committed to source control.**
- **`requirements.txt`** is now core-only (tensorflow / numpy / pyyaml).
  FastAPI / uvicorn / pydantic come exclusively via the `[api]` extra.
  `pip install kanx` no longer pulls a web server into your environment.
- **Dockerfile** runs as a non-root user (`kanx`, UID 1000),
  `readOnlyRootFilesystem`-compatible, with `tmp` and `~/.cache` mounted
  as `emptyDir` in the K8s deployment.
- **K8s `Deployment`** now sets BOTH `requests` and `limits` for cpu &
  memory (without limits, HPA is meaningless). Drops all Linux
  capabilities, sets `runAsNonRoot`, `allowPrivilegeEscalation: false`,
  `seccompProfile: RuntimeDefault`.
- **K8s `PersistentVolumeClaim`** documents the need to set
  `storageClassName` for the user's environment (default-class behaviour
  is not portable across EKS/GKE/AKS).
- **K8s HPA** annotates itself as a `cpu-starter` strategy and points
  operators to the deployment doc for the correct latency-based metric.

### Tests
- New `tests/test_grid_range_guard.py` (7 tests) — guard rails,
  monkey-patch verification of the runtime warning, and a regression test
  proving calibrated training reaches lower loss than uncalibrated.
- New `tests/test_api_hardening.py` (5 tests) — API-key acceptance/refusal,
  health/info bypass auth, rate-limit returns 429 after burst.
- Total suite: **113 tests** (up from 101), all green.

## [0.1.4] - 2026-05-27

### Fixed
- **Repository URLs corrected** across `README.md`, `pyproject.toml`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `mkdocs.yml`, all docs, the Colab
  notebook, and the launch post: `Mattral/Kolmogorov-Arnold-Networks` →
  `Mattral/KANX`, and `mattral.github.io/Kolmogorov-Arnold-Networks/` →
  `mattral.github.io/KANX/`. Affects all PyPI project_urls
  (Homepage, Documentation, Repository, Issues, Changelog, Source Code,
  Colab Notebook), the badges in the README, all hyperlinks in the docs
  site, and the "Open in Colab" badge in the notebook.

## [0.1.3] - 2026-05-27

### Added
- **Zero-friction API.** `kanx.quickstart()` — build, train, return a working
  KAN in one call. Designed to be the first thing a new user types after
  `pip install kanx`.
- **`model.fit(X, y)` on TF KAN** — auto-compiles with Adam(1e-3) + MSE if
  the user hasn't called `model.compile()` first. No more `compile()` dance.
- **`model.fit(X, y)` on PyTorch KAN** — wraps `Trainer.fit()` so PyTorch
  users get the same one-line semantics as the TF backend.
- **`CONTRIBUTING.md`** with high-leverage tasks and "good first issues" pointers.
- **GitHub Pages auto-deploy** workflow (`docs.yml`) — docs site rebuilds on
  every push to `main` (not just on tag).
- **Comparison table vs other KAN libraries** in `README.md` (pykan,
  efficient-kan, mlx-kan).
- **Citations + BibTeX** for both `kanx` and the original Liu et al. (2024) paper.
- **Extra project URLs on PyPI**: Documentation, Repository, Issues, Changelog,
  Colab Notebook, Source Code.

### Fixed
- **PyPI "Documentation" link** previously redirected to a non-existent page
  inside the PyPI project namespace. Now points to the live MkDocs Material
  site at <https://mattral.github.io/KANX/>.

### Tests
- New `tests/test_quickstart_api.py` (5 tests) covering `kanx.quickstart()`,
  TF auto-compile `.fit()`, Torch one-liner `.fit()`, and version metadata.

## [0.1.2] - 2026-05-27

Initial production release.

### Added

- **Library (TensorFlow backend, primary).** `kanx.{KAN, KANLinear, train,
  predict, load_model, save_model, load_config, validate_config,
  set_global_seed}`. Vectorized Cox-de Boor B-spline basis (no Python loops),
  SiLU residual, per-feature grids, `@register_keras_serializable` for
  custom-layer-safe `save_model` / `load_model`.
- **PyTorch backend.** `kanx.torch.{KAN, KANLinear, Trainer, export_onnx,
  build_kan}`. Parallel surface to the TF backend — same maths, same
  configuration semantics, native `torch.nn.Module` integration.
- **ONNX export.** `kanx.export_onnx_tf` (via tf2onnx) and
  `kanx.torch.export_onnx` (native torch.onnx). Dynamic batch axis;
  numerical parity within 1e-5 of the eager model.
- **REST API.** `api/app.py` FastAPI service with thread-safe
  `ModelRegistry`, `/api/{health,info,predict,load,reset}`, lifespan-based
  startup, checkpoint-with-fallback contract, boundary validation.
- **CLI.** `python -m kanx {info,train,predict}` with YAML configs.
- **Deployment.** Dockerfile + docker-compose; Kubernetes manifests
  (Deployment with rolling updates, Service, Ingress, HPA, PVC).
- **CI.** GitHub Actions workflow (`ci.yml`): lint + pytest matrix
  (py3.10/3.11/3.12) + Docker smoke test.
- **Release pipeline.** `release.yml` — tag-triggered PyPI publish (OIDC
  trusted publisher), GHCR Docker push, GitHub Release, MkDocs gh-deploy.
- **Documentation.** Eight long-form docs under `documentations/`:
  philosophy, architecture, system design, build, security, API, testing,
  deployment. MkDocs Material site config (`mkdocs.yml`).
- **Tests.** 95 tests across unit, integration, end-to-end (live API),
  property-based (Hypothesis), and performance regression alarms. **94%
  library coverage**, numerical contracts (partition of unity,
  non-negativity, save/load roundtrip, ONNX parity) asserted by tests.
- **Benchmarks.** `benchmarks/compare_mlp.py` + populated `results.md`.
  KAN[2,32,1] hits 1.7×10⁻⁵ MSE vs MLP[2,64,64,1] at 4.5×10⁻³ with 5×
  fewer parameters.
- **Examples.** `quickstart.py`, `function_fit.py`, `mnist_train.py`.
- **Notebooks.** Colab-ready `notebooks/quickstart.ipynb` ("Train KAN in
  2 minutes").

### Engineering decisions

- TensorFlow remains the *primary* backend (matches the upstream repo's
  framework). PyTorch is a first-class *secondary* surface; both are tested
  in CI.
- Single-threaded BLAS in the test suite to avoid TF+Torch OpenMP segfaults.
- Hand-rolled YAML validator (no pydantic dep in the core library).
- API uses pydantic only at the request-boundary edge.

[Unreleased]: https://github.com/Mattral/KANX/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/Mattral/KANX/releases/tag/v0.1.5
[0.1.4]: https://github.com/Mattral/KANX/releases/tag/v0.1.4
[0.1.3]: https://github.com/Mattral/KANX/releases/tag/v0.1.3
[0.1.2]: https://github.com/Mattral/KANX/releases/tag/v0.1.2
