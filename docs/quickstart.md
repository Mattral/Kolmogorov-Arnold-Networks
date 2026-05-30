# Quickstart

Three calls to train, save, and predict.

## 1. Install

```bash
pip install "kanx[torch,api,onnx]"
```

## 2. ⚡ One-call magic

```python
import kanx
model = kanx.quickstart()                   # build + train + return
model.predict([[0.5, 0.2]])                 # → array([[1.04…]])
```

## 3. Your data

=== "TensorFlow"

    ```python
    from kanx import KAN
    import numpy as np

    X = np.random.uniform(-1, 1, (1024, 2)).astype("float32")
    y = (np.sin(np.pi * X[:, :1]) + X[:, 1:2] ** 2).astype("float32")

    model = KAN([2, 64, 64, 1])
    model.fit(X, y, epochs=30, verbose=0)   # auto-compiles
    ```

=== "PyTorch"

    ```python
    import torch
    from kanx.torch import KAN

    model = KAN([2, 64, 64, 1])
    X = torch.randn(1024, 2)
    y = torch.sin(torch.pi * X[:, :1]) + X[:, 1:2] ** 2
    model.fit(X, y, epochs=30, lr=1e-2, val_split=0.1)
    model.save("kan.pt")
    ```

## 3. Serve

```bash
# (option A) Local
uvicorn api.app:app --port 8000

# (option B) Docker
docker run --rm -p 8000:8000 \
    -e KANX_CHECKPOINT=/app/checkpoints/kanx_model.keras \
    -v $(pwd)/checkpoints:/app/checkpoints \
    ghcr.io/mattral/kanx:latest
```

```bash
curl -X POST http://localhost:8000/api/predict \
     -H 'content-type: application/json' \
     -d '{"x": [[0.1, -0.2], [0.5, 0.7]]}'
```

```json
{ "output": [[0.31], [0.84]], "shape": [2, 1], "inference_ms": 22.4 }
```

## 4. Export to ONNX

=== "From PyTorch"

    ```python
    from kanx.torch import export_onnx
    export_onnx(model, "kan.onnx", sample_input=torch.zeros(1, 2))
    ```

=== "From TensorFlow"

    ```python
    from kanx import export_onnx_tf
    export_onnx_tf(model, "kan.onnx")
    ```

Both exports include a dynamic batch dimension and have been verified to
produce outputs identical to the eager model within 1e-5.

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("kan.onnx")
out = sess.run(None, {"input": np.zeros((4, 2), dtype=np.float32)})
```

## Next steps

- [Architecture](architecture.md) — how the package is laid out
- [REST API](api.md) — full endpoint reference
- [Deployment](deployment.md) — production rollouts
