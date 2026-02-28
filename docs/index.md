# kanx 🚀

<p align="center">
  <em>Production-grade <strong>Kolmogorov-Arnold Networks</strong> — TensorFlow + PyTorch + ONNX</em>
</p>

<p align="center">
  <img src="assets/benchmark.png" alt="KAN vs MLP benchmark" width="720"/>
</p>

!!! tip "**Honest, param-matched benchmark — KAN wins on smooth, separable targets**"
    The chart below shows real numbers from a fair, multi-baseline run.
    On non-smooth or high-dimensional data, the picture is far more nuanced.
    KANs are not universally better than MLPs.

---

## ⚡ The 30-second magic moment

```python
import kanx

# Build, train, predict — in ONE call. No config files. No compile dance.
model = kanx.quickstart()                    # trains a tiny KAN on synthetic data
model.predict([[0.5, 0.2]])                  # → array([[1.04…]])
```

Want more control? Same simplicity, your data:

=== "🔷 TensorFlow (default)"

    ```python
    from kanx import KAN
    import numpy as np

    X = np.random.uniform(-1, 1, (1024, 2)).astype("float32")
    y = np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2

    model = KAN([2, 64, 1])
    model.fit(X, y, epochs=30, verbose=0)    # auto-compiles with Adam+MSE
    model.predict(X[:3])
    ```

=== "🔶 PyTorch"

    ```python
    from kanx.torch import KAN
    import torch

    model = KAN([2, 64, 1])
    X = torch.randn(1024, 2); y = torch.sin(torch.pi * X[:, :1])
    model.fit(X, y, epochs=30, lr=1e-2)      # one-liner, same semantics
    model.predict([[0.5, 0.2]])
    ```

=== "🌐 REST API"

    ```bash
    docker run --rm -p 8000:8000 ghcr.io/mattral/kanx:latest
    curl -X POST http://localhost:8000/api/predict \
         -H 'content-type: application/json' \
         -d '{"x": [[0.1, -0.2], [0.5, 0.7]]}'
    ```

=== "🔄 ONNX"

    ```python
    from kanx.torch import KAN, export_onnx
    model = KAN([2, 64, 1])
    export_onnx(model, "kan.onnx")           # dynamic batch axis
    ```

---

## Install

```bash
pip install kanx                # core (TensorFlow)
pip install "kanx[torch]"       # +PyTorch backend
pip install "kanx[api]"         # +FastAPI service
pip install "kanx[all]"
```

---

## What to read next

| Page | What's inside |
|------|---------------|
| [Quickstart](quickstart.md)     | Train your first KAN in 60 seconds |
| [Philosophy](philosophy.md)     | Why kanx exists & non-goals |
| [Architecture](architecture.md) | Package layout, module contracts |
| [System Design](system_design.md) | Serving topology, scaling, failure modes |
| [REST API](api.md)              | Endpoint reference + curl examples |
| [Testing](testing.md)           | Test pyramid + numerical invariants |
| [Deployment](deployment.md)     | CI/CD, rollout, observability |
| [Benchmarks](benchmarks.md)     | KAN vs MLP — methodology + numbers |
| [Roadmap](roadmap.md)           | Shipped + P0/P1/P2 backlog |
| [Changelog](changelog.md)       | Release notes |

---

## License

[Apache 2.0](https://github.com/Mattral/KANX/blob/main/LICENSE).
