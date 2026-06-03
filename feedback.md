# KANX — GitHub Copilot Upgrade Instructions
# Top-Tier (FAANG / Frontier-Lab) Engineering Standard
# Version: 2026-06 | Target: kanx v0.2.0

---

## HOW TO USE THESE INSTRUCTIONS

Paste each numbered **PROMPT BLOCK** directly into GitHub Copilot Chat
(or any capable AI coding assistant) while you have the relevant file open.
Each block is self-contained: it states the file(s) to touch, the exact
changes required, the acceptance criteria, and a test oracle.

Work through the blocks in order — later blocks depend on earlier ones.

---

## CONTEXT BRIEF (include this at the top of every Copilot session)

```
Repository: https://github.com/Mattral/KANX
Language: Python 3.10-3.12
Primary framework: TensorFlow >= 2.16 (TF backend), PyTorch >= 2.0 (Torch backend)
Package layout: src/kanx/ (TF), src/kanx/torch/ (PyTorch)
CI: GitHub Actions (.github/workflows/ci.yml + release.yml)
Benchmarks: benchmarks/compare_mlp.py
Tests: tests/  (95 tests, 94% coverage)
Docs: docs/ (MkDocs Material source; single docs root; documentations/ removed)
Serving: api/app.py (FastAPI)
Infra: Dockerfile, docker-compose.yml, k8s/
Known issues to fix in this session: [PASTE RELEVANT BLOCK TITLE]
```

---

## BLOCK 1 — Fix README cosmetic and metadata bugs

**Open file:** `README.md`

**Prompt:**

```
The README.md in this repository has three specific bugs to fix.

1. LICENSE badge fix: The badge image URL
   `https://img.shields.io/badge/license-Apache--2.0-A78BFA?style=for-the-badge`
   currently links back to itself (the badge image). Change its href so it
   links to `./LICENSE` (the actual LICENSE file in the repo root).

2. Add a pepy.tech total-downloads badge immediately after the existing PyPI
   downloads badge. The correct shield format is:
   `[![Total Downloads](https://static.pepy.tech/badge/kanx)](https://pepy.tech/project/kanx)`

3. Add a CITATION.cff badge that links to the CITATION.cff file (to be
   created in BLOCK 2). The badge is:
   `[![Cite](https://img.shields.io/badge/cite-CITATION.cff-brightgreen)](./CITATION.cff)`

Do not change any other content. Output only the modified badge section
(the first ~15 lines of the README).
```

**Acceptance criteria:**
- `grep -n "LICENSE" README.md` shows href pointing to `./LICENSE`, not to the badge image URL.
- Three download-related badges appear in sequence: PyPI, pepy.tech total, Cite.

---

## BLOCK 2 — Add CITATION.cff and SECURITY.md

**Create new files:** `CITATION.cff`, `SECURITY.md`

**Prompt:**

```
Create two new files in the repository root.

FILE 1: CITATION.cff
Generate a valid Citation File Format (CFF) v1.2.0 file for this library.
Fields to populate:
  - cff-version: "1.2.0"
  - message: "If you use KANX, please cite both this software and the KAN paper."
  - type: software
  - title: "KANX: Production-grade Kolmogorov-Arnold Networks"
  - authors: [{family-names: "Myet", given-names: "Min Htet"}]
  - version: (read from pyproject.toml — currently 0.1.5)
  - doi: "10.5281/zenodo.20430883"
  - repository-code: "https://github.com/Mattral/KANX"
  - license: Apache-2.0
  - keywords: [kolmogorov-arnold, kan, neural-network, tensorflow, pytorch, onnx]
  - preferred-citation: a BibTeX entry for Liu et al. 2024 (arXiv:2404.19756)

FILE 2: SECURITY.md
Write a concise security policy covering:
  - Supported versions table (only latest PyPI release receives patches)
  - How to report a vulnerability (email placeholder: security@kanx-project.org,
    response SLA: 48 hours, disclosure timeline: 90 days)
  - What is NOT in scope (benchmark accuracy disputes, third-party dependencies
    outside this repo's control)
  - A note that the FastAPI service should never be exposed to the public
    internet without an auth layer and TLS termination.

Both files must pass `yamllint` (CITATION.cff) and `markdownlint` (SECURITY.md).
```

**Acceptance criteria:**
- `python -c "import yaml; yaml.safe_load(open('CITATION.cff'))"` exits 0.
- GitHub renders "Cite this repository" button on the repo landing page.
- `SECURITY.md` appears in the repo Security tab.

---

## BLOCK 3 — Consolidate docs/ and documentations/ folders

**Files to touch:** `docs/`, `documentations/`, `mkdocs.yml`, `README.md`

**Prompt:**

```
The repository has two documentation directories:
  - docs/  — MkDocs Material source (12 pages, served as GitHub Pages)
  - documentations/ — 8 long-form markdown files (philosophy, architecture,
    system_design, build, security, api, testing, deployment)

These are redundant. Merge them following this strategy:

1. Move all files from documentations/ into docs/reference/ as a new subsection.
   Rename each file to snake_case if not already:
   e.g. docs/reference/architecture.md, docs/reference/system_design.md, etc.

2. Update mkdocs.yml to add a "Reference" nav section pointing to the new paths.

3. Update any relative cross-links inside the moved files to resolve correctly
   under their new path (e.g., links that previously pointed to
   `../documentations/api.md` should now point to `./api.md` or the MkDocs
   nav slug).

4. Delete the documentations/ directory entirely.

5. Update README.md: the "Documentation" section currently lists both the
   MkDocs site and in-repo documentations/. Remove the in-repo link and
   update the table to point to the new docs/ reference paths.

Run `mkdocs build --strict` and confirm zero warnings before committing.
```

**Acceptance criteria:**
- `ls documentations/` returns "No such file or directory".
- `mkdocs build --strict` exits 0 with no warnings.
- All 20 nav entries in mkdocs.yml resolve to existing files.

---

## BLOCK 4 — Real-world benchmark suite (UCI + Feynman + throughput)

**Create new files:**
  `benchmarks/real_world.py`,
  `benchmarks/results/README.md`,
  `benchmarks/results/.gitkeep`,
  `scripts/run_benchmarks.sh`

**Prompt:**

```
The current benchmark (benchmarks/compare_mlp.py) runs on CPU only with a
synthetic 2-D regression target over 30 epochs. This is insufficient for a
FAANG-level claim. Create a new, credible benchmark suite.

FILE: benchmarks/real_world.py

Requirements:
- Download three real-world tabular datasets using scikit-learn / UCI:
    (a) California Housing (sklearn.datasets.fetch_california_housing)
        — 8 features, 20640 samples, regression
    (b) Concrete Compressive Strength
        — download from UCI URL https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls
        — 8 features, 1030 samples, regression
    (c) Energy Efficiency (UCI)
        — https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx
        — 8 features, 768 samples, regression (Y1 target)

- For each dataset, train and evaluate:
    (i)  KAN[n_features, 32, 1] — kanx TF backend, Adam, 200 epochs
    (ii) KAN[n_features, 32, 1] — kanx PyTorch backend, Adam, 200 epochs
    (iii) MLP[n_features, 64, 64, 1] — PyTorch nn.Sequential, same training budget

- Metrics to record per model per dataset:
    - Test RMSE (5-fold CV, report mean ± std)
    - Test R² (5-fold CV, mean ± std)
    - Training wall-clock time (seconds)
    - Inference latency: median of 1000 calls on a batch of 256 rows (ms)
    - Parameter count
    - If CUDA is available, repeat the inference latency on GPU and record
      separately as `inference_latency_gpu_ms`

- Output a JSON file to benchmarks/results/real_world_results.json with
  schema:
  {
    "generated_at": "<ISO-8601 timestamp>",
    "hardware": { "cpu": "<cpu_info>", "gpu": "<gpu_info or null>" },
    "results": [
      {
        "dataset": "california_housing",
        "model": "KAN_TF",
        "rmse_mean": 0.0, "rmse_std": 0.0,
        "r2_mean": 0.0, "r2_std": 0.0,
        "train_time_s": 0.0,
        "params": 0,
        "inference_latency_cpu_ms": 0.0,
        "inference_latency_gpu_ms": null
      }, ...
    ]
  }

- Also print a human-readable Markdown table to stdout that can be pasted
  directly into the README.

- Add a CLI flag --device [cpu|cuda|auto] (default auto).
- Add a CLI flag --epochs N (default 200).
- Add a CLI flag --datasets [all|california|concrete|energy] (default all).
- The script must be runnable with: python benchmarks/real_world.py

FILE: scripts/run_benchmarks.sh
A shell script that:
  1. Runs python benchmarks/compare_mlp.py --long
  2. Runs python benchmarks/real_world.py
  3. Copies both result files into benchmarks/results/
  4. Prints "Benchmark suite complete."

Add error handling: exit nonzero if either benchmark fails.

FILE: benchmarks/results/README.md
Explain what each result file contains and how to regenerate them.
Include a placeholder table that will be populated by the benchmark run.
```

**Acceptance criteria:**
- `python benchmarks/real_world.py --datasets california --epochs 10` exits 0
  and produces a valid JSON file.
- JSON schema validates against the structure above.
- The script does NOT hardcode any benchmark numbers.
- `pytest tests/ -k benchmark` passes (write at least one smoke test that
  imports and calls the module).

---

## BLOCK 5 — Commit benchmark results artifact to the repository

**File to create:** `benchmarks/results/real_world_results.json`

**Prompt:**

```
After running `python benchmarks/real_world.py` successfully on the development
machine (CPU run is sufficient for committing an initial artifact):

1. Commit the generated benchmarks/results/real_world_results.json to the
   repository. This is intentional — having a reproducible baseline artifact
   committed to the repo is a key signal of credibility.

2. Update README.md: replace the current benchmark table (which only shows
   synthetic 2-D results) with a new section titled
   "## Benchmarks — real-world tabular datasets".
   
   The new section should:
   - Show the full Markdown table printed by real_world.py
   - State clearly: hardware used, number of CV folds, epoch count, and
     that GPU numbers are "N/A (CPU run; GPU results welcome as PRs)"
   - Keep the original synthetic benchmark table but demote it to a
     subsection titled "### Synthetic baseline (original)"
   - Add a one-line note: "All benchmarks are reproducible:
     `bash scripts/run_benchmarks.sh`"

3. Update docs/benchmarks.md with the same table and regeneration instructions.

Do NOT invent or fabricate numbers. Only use the actual output of
benchmarks/real_world.py.
```

**Acceptance criteria:**
- `jq '.results | length' benchmarks/results/real_world_results.json` returns
  a value ≥ 3 (at least 3 dataset × model combinations).
- README.md contains "real-world tabular datasets" heading.
- Numbers in README match numbers in the JSON artifact.

---

## BLOCK 6 — GPU timing path in the existing compare_mlp.py benchmark

**File to modify:** `benchmarks/compare_mlp.py`

**Prompt:**

```
The existing benchmarks/compare_mlp.py runs all timing on CPU. Add a GPU
timing path with the following requirements:

1. At the top of the script, detect CUDA / MPS availability:
   ```python
   import torch
   DEVICE = (
       "cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available()
       else "cpu"
   )
   ```

2. Add a --device CLI argument (choices: cpu, cuda, mps, auto; default auto).
   When auto, use the DEVICE logic above.

3. For the PyTorch KAN model:
   - Move model and data to the selected device before timing.
   - Use torch.cuda.synchronize() (when device == "cuda") before and after
     the timed region to get wall-clock GPU time, not CPU-side scheduling time.
   - Report GPU inference latency separately from CPU latency in the output table.

4. For the TensorFlow KAN model:
   - Use tf.config.list_physical_devices('GPU') to detect GPU availability.
   - If a GPU is available, run a warm-up pass (10 iterations) then time 100
     inference calls on device and report the median.

5. Add a new column "Device" to the printed results table.

6. Do not change the --long flag behaviour or any existing test that imports
   this module.

7. Update the docstring at the top of the file to mention the --device flag.
```

**Acceptance criteria:**
- `python benchmarks/compare_mlp.py --device cpu` exits 0 and prints a table.
- `python benchmarks/compare_mlp.py --device cuda` exits 0 on a machine with
  CUDA (or prints a graceful "CUDA not available, falling back to CPU" and
  continues).
- `pytest tests/ -k compare_mlp` passes.

---

## BLOCK 7 — MatrixKAN kernel: vectorised spline via batched GEMM

**Create new file:** `src/kanx/torch/matrix_kan.py`

**Prompt:**

```
The core bottleneck of the B-spline KAN is that Cox-de Boor recursion is
serial and memory-inefficient on GPU. Implement a MatrixKAN layer for the
PyTorch backend that replaces the recursive evaluation with a fully vectorised
batched matrix multiply (no Python loops in the forward pass).

The approach:
  For spline order k and G grid intervals, precompute the (G+k) × (G+k)
  recurrence matrix M_p for each recursion level p = 1..k, then evaluate
  all B-spline bases via:
    B = x_augmented @ M_1 @ M_2 @ ... @ M_k   (batched einsum)
  This reduces the forward pass to k dense GEMMs which are GPU-friendly.

Requirements for src/kanx/torch/matrix_kan.py:

1. Class `MatrixKANLinear(nn.Module)`:
   - __init__(self, in_features, out_features, grid_size=5, spline_order=3,
               scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
               base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1,1))
   - forward(self, x) → tensor of shape (batch, out_features)
     Uses batched matrix multiply for B-spline basis evaluation:
       (a) Extend grid: shape (in_features, grid_size + 2*spline_order + 1)
       (b) Build basis via einsum chains — no Python for-loops
       (c) Apply learnable coefficients via a single batched matmul
       (d) Add SiLU residual (base_activation(x) * scale_base)
   - update_grid(self, x, margin=0.01): adaptive grid update from samples
     (matches pykan API contract)
   - get_spline_weight_at_grid_points(self): for symbolic regression hooks

2. Class `MatrixKAN(nn.Module)`:
   - __init__(self, layers_hidden: list[int], **layer_kwargs)
   - forward(self, x) → tensor

3. Numerical contract: for any input x of shape (B, in_features),
   `MatrixKANLinear.forward(x)` must agree with the existing `KANLinear`
   (in src/kanx/torch/layers.py) to within atol=1e-4 when both are
   initialised with the same random seed.

4. Export `MatrixKAN` and `MatrixKANLinear` from `src/kanx/torch/__init__.py`.

5. Write tests in `tests/test_matrix_kan.py`:
   - test_matrix_kan_output_shape: forward pass produces correct shape
   - test_matrix_kan_numerical_agreement: agrees with KANLinear within 1e-4
   - test_matrix_kan_gpu_throughput: if CUDA available, measures tokens/sec
     and asserts it is >= 2× the standard KANLinear throughput on the same
     hardware (skip with pytest.mark.skipif if no CUDA)
   - test_matrix_kan_onnx_export: ONNX export succeeds and parity holds

6. Add a benchmark entry in benchmarks/compare_mlp.py under the name
   "MatrixKAN" that times this implementation alongside the existing KAN.
```

**Acceptance criteria:**
- `pytest tests/test_matrix_kan.py -v` — all 4 tests pass (GPU test skipped
  on CPU-only machines, not failed).
- `python -c "from kanx.torch import MatrixKAN; m = MatrixKAN([2,32,1]); print('ok')"` exits 0.
- Numerical agreement test passes (atol=1e-4).

---

## BLOCK 8 — Adaptive grid update (pykan parity)

**Files to modify:** `src/kanx/layers.py`, `src/kanx/model.py`,
  `src/kanx/torch/layers.py`, `src/kanx/torch/model.py`

**Prompt:**

```
The roadmap lists `update_grid_from_samples` as in-progress (P0). Implement it
for both backends to reach pykan API parity.

For the TensorFlow backend (src/kanx/layers.py):

1. Add method `update_grid_from_samples(self, x, margin=0.01)` to
   `KANLinear`:
   - x: tf.Tensor of shape (batch, in_features)
   - Compute per-feature adaptive grid using quantiles of x along axis=0.
   - grid_eps=0.02 interpolation between uniform and sample-based as in pykan.
   - Recompute self.grid (a tf.Variable) in-place using tf.Variable.assign().
   - This must be differentiable: after update, forward() uses the new grid.
   
2. Expose `model.update_grid_from_samples(x)` on the top-level `KAN` class
   (src/kanx/model.py) so it calls each layer's method in sequence.

For the PyTorch backend (src/kanx/torch/layers.py, src/kanx/torch/model.py):

3. Add the same method to `KANLinear` (PyTorch version).
   Use torch.quantile for the adaptive grid computation.
   Update self.grid (an nn.Buffer registered via register_buffer) in-place
   using tensor.copy_().

4. Expose on PyTorch `KAN` model similarly.

Tests to add in tests/test_grid_update.py:
   - test_tf_grid_update_changes_grid: after update_grid_from_samples,
     assert the grid values differ from the initial uniform grid.
   - test_tf_grid_update_preserves_output_shape: forward() still works.
   - test_torch_grid_update_changes_grid: same for PyTorch.
   - test_torch_grid_update_is_numerically_stable: grid values are strictly
     increasing after update with any non-degenerate x.
   - test_grid_update_improves_loss: train a KAN for 5 epochs, call
     update_grid_from_samples, train 5 more epochs; assert final loss is
     lower than without the update (test with seed for reproducibility).

Update docs/quickstart.md to show grid update usage in a code snippet.
```

**Acceptance criteria:**
- `pytest tests/test_grid_update.py -v` — all 5 tests pass.
- `python -c "from kanx import KAN; import numpy as np; m = KAN([2,8,1]); m.update_grid_from_samples(np.random.randn(100,2).astype('float32'))"` exits 0.

---

## BLOCK 9 — TensorBoard callback

**Files to create/modify:** `src/kanx/callbacks.py`, `src/kanx/train.py`,
  `src/kanx/torch/trainer.py`

**Prompt:**

```
Add TensorBoard logging to both backends (roadmap item: "TensorBoard callback
wired into train()").

FILE: src/kanx/callbacks.py
Create a `KANTensorBoardCallback` that extends `tf.keras.callbacks.Callback`:
  - Logs loss and val_loss scalars at each epoch.
  - Logs per-layer spline grid histograms every `histogram_freq` epochs
    (default 5). Iterate over model.layers, check for KANLinear instances,
    and call tf.summary.histogram("layer_{i}/grid", layer.grid, step=epoch).
  - Logs inference latency (median over 100 forward passes on a fixed 256-row
    batch from the training data) every 10 epochs as a scalar.
  - Constructor: __init__(self, log_dir="logs/kanx", histogram_freq=5)

Modify src/kanx/train.py:
  - The `fit()` / `train()` function should accept an optional
    `tensorboard: bool = False` argument.
  - When True, instantiate KANTensorBoardCallback and append it to the
    callbacks list before calling model.fit().
  - Also accept `log_dir: str = "logs/kanx"`.

For the PyTorch trainer (src/kanx/torch/trainer.py):
  - Accept `tensorboard: bool = False` and `log_dir: str = "logs/kanx"`.
  - When True, use torch.utils.tensorboard.SummaryWriter to log:
      - train_loss at each epoch
      - val_loss at each epoch (if validation data provided)
      - per-layer spline coefficient norms (as scalars) every 5 epochs
      - inference latency every 10 epochs

Add tests in tests/test_callbacks.py:
  - test_tensorboard_callback_creates_log_dir: after training 2 epochs with
    tensorboard=True, assert the log_dir directory exists and contains at
    least one events.out.tfevents.* file.
  - test_tensorboard_does_not_crash_without_gpu: same test on CPU.

Update CLI: add --tensorboard flag to `python -m kanx train`.
Update docs/quickstart.md with a one-paragraph "Monitoring with TensorBoard"
section and the command `tensorboard --logdir logs/kanx`.
```

**Acceptance criteria:**
- `pytest tests/test_callbacks.py -v` passes.
- `python -m kanx train --config configs/default.yaml --tensorboard` exits 0
  and creates `logs/kanx/`.

---

## BLOCK 10 — kanx.datasets mini-module (Feynman + UCI)

**Create new files:** `src/kanx/datasets/__init__.py`,
  `src/kanx/datasets/feynman.py`, `src/kanx/datasets/tabular.py`

**Prompt:**

```
Implement the `kanx.datasets` mini-module (roadmap P0 item).

FILE: src/kanx/datasets/feynman.py
Expose three Feynman benchmark functions used in the original KAN paper:
  - feynman_I_9_18(x):  F = m1*m2*G / ((x1-x2)^2 + (y1-y2)^2)  — 6 features
  - feynman_I_34_8(x):  omega = omega_0 / (1 - v/c)              — 3 features
  - feynman_II_11_27(x): Pol = n*alpha/(1 - n*alpha/3) * epsilon * Ef — 4 features
For each:
  - Implement a Python function matching the formula.
  - Provide a `make_dataset(n=1000, noise=0.01, seed=42)` factory that returns
    a dict {"X_train", "y_train", "X_test", "y_test"} as numpy arrays.
    Input ranges should match the Feynman SR benchmark defaults.

FILE: src/kanx/datasets/tabular.py
Expose loaders for the three UCI datasets used in the real-world benchmark:
  - load_california_housing() → (X, y, feature_names)
  - load_concrete_strength() → (X, y, feature_names)
  - load_energy_efficiency() → (X, y, feature_names)
Each function downloads the dataset on first call (using urllib or requests),
caches it to ~/.cache/kanx/datasets/<name>.npz, and loads from cache
on subsequent calls. Normalise X to zero mean unit variance. Normalise y.

FILE: src/kanx/datasets/__init__.py
Re-export all public symbols from feynman.py and tabular.py.
Update src/kanx/__init__.py to expose `from kanx import datasets`.

Tests in tests/test_datasets.py:
  - test_feynman_I_9_18_shape: make_dataset returns arrays of expected shape.
  - test_feynman_output_range: y values are finite (no NaN/Inf).
  - test_california_housing_loads: X shape is (N, 8).
  - test_datasets_cached: second call returns same data without network access
    (mock urllib or assert .npz cache file exists after first call).

Update benchmarks/real_world.py to import from kanx.datasets instead of
re-implementing download logic inline.
```

**Acceptance criteria:**
- `python -c "from kanx.datasets import make_dataset, load_california_housing; print('ok')"` exits 0.
- `pytest tests/test_datasets.py -v` — all 4 tests pass.

---

## BLOCK 11 — HuggingFace Hub integration

**Files to modify:** `src/kanx/model.py`, `src/kanx/torch/model.py`,
  `pyproject.toml`

**Prompt:**

```
Implement `KAN.push_to_hub()` and `KAN.from_pretrained()` for both backends
(roadmap item: "HuggingFace Hub integration").

For the TensorFlow backend (src/kanx/model.py):
1. Add classmethod `from_pretrained(cls, repo_id: str, revision="main", **kwargs) -> KAN`:
   - Uses `huggingface_hub.hf_hub_download` to fetch "model.keras" and
     "config.yaml" from the Hub repo.
   - Reconstructs the model from config.yaml, then loads weights from model.keras.
   - Returns a ready-to-use KAN instance.
2. Add method `push_to_hub(self, repo_id: str, commit_message="Upload KANX model",
                            private=False)`:
   - Saves model.keras and config.yaml to a temp dir.
   - Uses `huggingface_hub.HfApi().upload_folder` to push both files.
   - Creates the Hub repo if it does not exist.

For the PyTorch backend (src/kanx/torch/model.py):
3. Same interface, but saves/loads model.pt (torch.save state_dict) +
   config.yaml.

Dependency: add `huggingface_hub>=0.21` to pyproject.toml under a new
optional extra `[hub]`. Add it to `[all]` as well.

Tests in tests/test_hub.py (use pytest-mock to mock HfApi and hf_hub_download
so no real network calls occur):
  - test_push_to_hub_calls_upload: verify push_to_hub calls
    HfApi().upload_folder with the correct repo_id.
  - test_from_pretrained_loads_weights: mock hf_hub_download to return a
    locally saved model file; verify from_pretrained returns a KAN instance
    with the same architecture as the original.
  - test_from_pretrained_torch: same for PyTorch backend.

Update docs/quickstart.md with a "Share your model" section showing
push_to_hub / from_pretrained usage.
```

**Acceptance criteria:**
- `pip install "kanx[hub]"` installs huggingface_hub.
- `pytest tests/test_hub.py -v` — all 3 tests pass (no real network calls).

---

## BLOCK 12 — Prometheus /metrics endpoint on FastAPI

**Files to modify:** `api/app.py`, `api/requirements.txt` (or
  `pyproject.toml`), `k8s/`

**Prompt:**

```
Add a Prometheus /metrics endpoint to the FastAPI serving layer (roadmap P1
item).

1. Add `prometheus-fastapi-instrumentator>=6` to pyproject.toml under the
   `api` extra.

2. In api/app.py:
   - Import Instrumentator from prometheus_fastapi_instrumentator.
   - In the app lifespan (or on_event startup), call:
       Instrumentator().instrument(app).expose(app, endpoint="/metrics")
   - Add a custom counter metric `kanx_inference_total` that increments on
     every successful POST /api/predict call, with labels:
       backend (tf | torch), batch_size (bucketed: 1, 2-10, 11-100, 101+)
   - Add a custom histogram metric `kanx_inference_latency_seconds` that
     records the wall-clock time of the model.predict() call inside
     /api/predict.

3. Update the API endpoint table in README.md to add:
   GET /metrics  — Prometheus metrics scrape endpoint

4. Add a Prometheus scrape config snippet to docs/deployment.md showing how
   to add the kanx service as a scrape target.

5. Update k8s/production-stack.yaml:
   - Add annotation `prometheus.io/scrape: "true"` to the Service spec.
   - Add annotation `prometheus.io/port: "8000"` and
     `prometheus.io/path: "/metrics"`.

Tests in tests/test_api_metrics.py:
  - test_metrics_endpoint_returns_200: GET /metrics returns 200.
  - test_metrics_contains_kanx_counter: after one POST /api/predict,
    GET /metrics response body contains "kanx_inference_total".
  - test_metrics_contains_latency_histogram: body contains
    "kanx_inference_latency_seconds".
```

**Acceptance criteria:**
- `pytest tests/test_api_metrics.py -v` — all 3 tests pass.
- `curl http://localhost:8000/metrics` returns text/plain with Prometheus
  format when the server is running.

---

## BLOCK 13 — CI/CD: add benchmark gate + real-world benchmark job

**File to modify:** `.github/workflows/ci.yml`

**Prompt:**

```
Add two new CI jobs to .github/workflows/ci.yml.

JOB 1: benchmark-smoke
  Runs on: ubuntu-latest
  Trigger: push to main and pull_request
  Steps:
    1. Checkout code.
    2. Set up Python 3.11.
    3. pip install -e ".[dev,torch,onnx]"
    4. Run: python benchmarks/compare_mlp.py (quick, 30 epochs).
    5. Run: python benchmarks/real_world.py --datasets california --epochs 5
       (fast smoke — 5 epochs just to verify the pipeline runs, not for
       publishable numbers).
    6. Assert that benchmarks/results/real_world_results.json exists (created
       by step 5).
    7. Upload benchmarks/results/ as a GitHub Actions artifact named
       "benchmark-results-${{ github.sha }}" with retention-days: 30.

JOB 2: coverage-gate
  Runs on: ubuntu-latest
  Trigger: push to main and pull_request
  Steps:
    1. Checkout.
    2. Python 3.11.
    3. pip install -e ".[dev,torch,onnx]"
    4. pytest tests/ --cov=kanx --cov-report=xml --cov-fail-under=92
    5. Upload coverage.xml as artifact.
  Fail the PR if coverage drops below 92%.

Ensure both jobs are listed in the `needs` of the existing release job so
releases can only proceed if both pass.

Also update the CI badge in README.md to use the new ci.yml status badge URL
format if it changed.
```

**Acceptance criteria:**
- `act -j benchmark-smoke` passes locally (or confirmed by a PR with green CI).
- `act -j coverage-gate` fails when coverage drops below 92% and passes above.

---

## BLOCK 14 — Symbolic regression hooks (post-hoc edge function fitting)

**Create new file:** `src/kanx/torch/symbolic.py`

**Prompt:**

```
Implement symbolic regression hooks (roadmap item: "Symbolic regression
post-hoc fit per edge").

FILE: src/kanx/torch/symbolic.py

Provide a `SymbolicFitter` class that, given a trained MatrixKAN or KAN
model, attempts to identify closed-form symbolic expressions for each learned
spline edge function.

Requirements:

1. Maintain a library of candidate symbolic functions:
   {"identity": lambda x: x,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "sqrt": lambda x: torch.sqrt(x.abs()),
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
    "log": lambda x: torch.log(x.abs() + 1e-8),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid}

2. Method `fit_edge(self, layer_idx, in_idx, out_idx, x_samples, threshold=0.99)`
   - Evaluates the trained spline φ_{out,in}(x) on x_samples (a 1-D tensor of
     values in the layer's grid range).
   - For each candidate function f, fits a linear combination
     a*f(b*x + c) + d by minimizing MSE using torch.optim.LBFGS (50 steps).
   - Returns the best-fit function name and R² score if R² > threshold,
     else returns ("spline", 0.0).

3. Method `fit_all(self, model, x_samples_per_layer=None)` → dict:
   - Iterates over all layers and all (in, out) edge pairs.
   - Returns a nested dict:
     {layer_idx: {(in_idx, out_idx): {"fn": "sin", "r2": 0.997, "params": {...}}}}

4. Method `to_sympy(self, fit_result)` → str:
   - Converts the fit result dict to a human-readable SymPy expression string.
   - Requires sympy as an optional import. If not installed, raises
     ImportError with a helpful message: `pip install kanx[symbolic]`.

Add `sympy>=1.12` to a new `[symbolic]` optional extra in pyproject.toml.
Add it to `[all]`.

Export SymbolicFitter from src/kanx/torch/__init__.py.

Tests in tests/test_symbolic.py:
  - test_fit_sin: train a 1-input, 1-output KAN on y=sin(x), call fit_edge,
    assert returned function is "sin" and R² > 0.99.
  - test_fit_all_returns_dict: fit_all on a KAN[2,4,1] returns a dict with
    correct structure.
  - test_to_sympy_returns_string: to_sympy output is a non-empty string.
```

**Acceptance criteria:**
- `pytest tests/test_symbolic.py -v` passes.
- `python -c "from kanx.torch import SymbolicFitter; print('ok')"` exits 0.

---

## BLOCK 15 — Final audit: pyproject.toml, README badges, and version bump

**Files to modify:** `pyproject.toml`, `README.md`, `CHANGELOG.md`

**Prompt:**

```
Perform a final housekeeping pass to bring the repository to v0.2.0 standard.

1. pyproject.toml — version bump to 0.2.0.

2. pyproject.toml — add all new optional extras introduced in Blocks 11-14:
   hub = ["huggingface_hub>=0.21"]
   symbolic = ["sympy>=1.12"]
   Update `all` to include hub and symbolic.

3. pyproject.toml — add `mypy>=1.8` and `types-PyYAML` to the `dev` extra.
   Add a [tool.mypy] section:
     python_version = "3.10"
     strict = false
     ignore_missing_imports = true
     disallow_untyped_defs = true
     warn_return_any = true

4. README.md — update the install table to show the new extras:
   pip install "kanx[hub]"       # HuggingFace Hub push/pull
   pip install "kanx[symbolic]"  # Symbolic regression
   pip install "kanx[all]"       # Everything

5. CHANGELOG.md — add a [0.2.0] section following Keep-a-Changelog format:
   ### Added
   - Real-world benchmark suite (California Housing, Concrete Strength, Energy)
   - GPU timing path in compare_mlp.py
   - MatrixKAN: vectorised batched-GEMM B-spline layer
   - Adaptive grid update (update_grid_from_samples) for both backends
   - TensorBoard callback
   - kanx.datasets mini-module (Feynman + UCI tabular)
   - HuggingFace Hub integration (push_to_hub / from_pretrained)
   - Prometheus /metrics endpoint
   - Symbolic regression hooks (SymbolicFitter)
   - CITATION.cff
   - SECURITY.md
   ### Changed
   - Merged documentations/ into docs/reference/
   - Benchmark section expanded with real-world results
   ### Fixed
   - LICENSE badge linked to image instead of LICENSE file
   - Missing pepy.tech total-downloads badge

6. Run `python -m build` and verify the wheel builds without errors.
   Run `twine check dist/*` and verify it passes.
```

**Acceptance criteria:**
- `python -m build` exits 0.
- `twine check dist/*` exits 0 with no errors or warnings.
- `python -c "import kanx; print(kanx.__version__)"` prints "0.2.0".
- `grep '"version"' pyproject.toml | head -1` shows 0.2.0.

---

## MASTER CHECKLIST

Copy this into a GitHub Issue titled "v0.2.0 upgrade tracker" and check off
each item as you complete the corresponding block:

- [x] BLOCK 1: README badge fixes
- [x] BLOCK 2: CITATION.cff + SECURITY.md
- [x] BLOCK 3: Docs consolidation (docs/ + documentations/ → docs/)
- [ ] BLOCK 4: Real-world benchmark suite code
- [ ] BLOCK 5: Benchmark results artifact committed
- [ ] BLOCK 6: GPU timing in compare_mlp.py
- [ ] BLOCK 7: MatrixKAN vectorised layer
- [ ] BLOCK 8: Adaptive grid update (pykan parity)
- [x] BLOCK 9: TensorBoard callback
- [ ] BLOCK 10: kanx.datasets mini-module
- [x] BLOCK 11: HuggingFace Hub integration
- [x] BLOCK 12: Prometheus /metrics endpoint
- [ ] BLOCK 13: CI benchmark gate + coverage gate
- [x] BLOCK 14: Symbolic regression hooks
- [ ] BLOCK 15: Version bump to 0.2.0 + final audit

---

## TECHNICAL RATIONALE NOTES
(for reference when discussing with reviewers)

### Why MatrixKAN matters
The Cox-de Boor recursion is O(k · G · in · out) sequential tensor ops.
On GPU this serialises to k back-to-back kernels with intermediate allocations.
The batched GEMM formulation collapses this to k GEMMs on pre-built matrices,
reducing kernel launch overhead from O(k) to O(1) per layer and making the
forward pass GPU-memory-bandwidth-bound rather than launch-overhead-bound.
For k=3, G=5: expected 2–4× throughput improvement on A100.

### Why real-world benchmarks matter
The current 265× MSE claim on `y = sin(πx₁) + cos(2πx₂)` is a synthetic
function that KANs are structurally designed to learn well (smooth, low-rank,
product-separable). FAANG ML engineers will immediately ask: does this hold on
tabular data with collinear features, heterogeneous scales, and label noise?
The California Housing, Concrete, and Energy datasets test these conditions.
Expected realistic outcome: KAN ≈ MLP on R², with worse training throughput
and better extrapolation — this is an honest and defensible result.

### Why committed benchmark artifacts matter
Static badges with numbers not backed by committed run artifacts are a red
flag in code review. A `benchmarks/results/real_world_results.json` file that
can be regenerated with one command signals engineering discipline, not
marketing.

### Why CITATION.cff matters for academic discoverability
GitHub renders a "Cite this repository" button only when CITATION.cff is
present. Citation count on Zenodo + GitHub is a proxy signal that hiring
managers at research labs use when evaluating candidates' community impact.

---

*End of KANX Copilot Upgrade Instructions — v0.2.0 target* 
