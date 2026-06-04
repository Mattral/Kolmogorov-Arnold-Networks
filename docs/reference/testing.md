# Testing Strategy

## Test pyramid

```
                ▲
                │
           ┌────┴──────┐
           │    e2e    │  tests/test_api.py        — FastAPI TestClient
           ├───────────┤
           │integration│ tests/test_train_inference.py — train→ckpt→predict
           ├───────────┤
           │   unit    │  tests/test_layers.py     — B-spline + KANLinear
           │           │  tests/test_model.py      — KAN sequential
           │           │  tests/test_config.py     — YAML + dataclasses
           └───────────┘
```

* **Unit (30 tests)** — math, shapes, validation, gradient flow.
* **Integration (9 tests)** — full train loop on synthetic data, checkpoint
  persistence, save/load roundtrip, optimiser/loss factories.
* **End-to-end (7 tests)** — FastAPI TestClient hits `/api/{health,info,
  predict,load,reset}` with valid and invalid payloads.

Total: **113 tests, all green**.

## Conventions

* `tests/conftest.py` adds `src/` and the repo root to `sys.path` and
  silences TF stderr. There is no global `monkeypatch` of TF — tests are
  hermetic at process scope.
* Random data uses a *fixed* `np.random.RandomState` or
  `np.random.default_rng(seed)` so failures are reproducible.
* No `time.sleep`, no flakes. Inference latency is measured with
  `time.perf_counter` and **not** asserted (machines vary).
* Each test asserts **one** behaviour. If a fix breaks 5 tests, the failure
  message tells you exactly which contract regressed.

## Numerical correctness

Two non-trivial maths invariants are asserted directly:

1. **Partition of unity.** Inside the inner knot range,
   `Σᵢ Bᵢᵏ(x) == 1` for all `x`. `tests/test_layers.py::test_b_spline_partition_of_unity`.
2. **Non-negativity.** Every basis value is `>= 0` up to numerical noise.
   `tests/test_layers.py::test_b_spline_non_negative`.

Plus structural invariants:

3. **Basis count.** `num_basis == grid_size + spline_order` (pykan
   convention).
4. **Shape correctness.** Forward, gradient and serialization paths all
   produce the documented shapes.
5. **Save/load roundtrip.** A saved-then-loaded `KAN` produces *identical*
   outputs (atol=1e-5) on a fixed input.

## Coverage

CI computes coverage via `pytest --cov=src/kanx` (`pytest-cov`). The current
suite covers every public function in the library; the small uncovered
fraction is `__main__.py` error paths exercised only by integration scripts.

## Running tests

```bash
bash scripts/test.sh             # default: full suite + coverage report
pytest tests/test_layers.py -v   # one file
pytest -k partition_of_unity     # one test
```

## What is *not* tested

* GPU codepaths — the library inherits TF's GPU support but the CI runners
  are CPU-only. GPU smoke tests are listed in `roadmap.md`.
* Long-running MNIST convergence — `examples/mnist_train.py` exists as a
  reproducible script; we don't gate CI on its accuracy.
* Docker image — built and smoke-tested in CI but only as a separate job;
  not part of the pytest suite.

