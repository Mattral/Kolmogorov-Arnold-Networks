import numpy as np
import torch

from kanx.torch import KAN, SymbolicFitter, Trainer


def test_fit_sin():
    model = KAN([1, 1])
    x = np.linspace(-1.0, 1.0, 200, dtype=np.float32).reshape(-1, 1)
    y = np.sin(x)

    Trainer(model, device="cpu").fit(
        x,
        y,
        epochs=80,
        batch_size=32,
        lr=1e-2,
        verbose=0,
    )

    fitter = SymbolicFitter(model)
    fn, r2 = fitter.fit_edge(0, 0, 0, torch.linspace(-1.0, 1.0, 200))

    assert fn == "sin"
    assert r2 > 0.99


def test_fit_all_returns_dict():
    model = KAN([2, 4, 1])
    fitter = SymbolicFitter(model)
    results = fitter.fit_all()

    assert isinstance(results, dict)
    assert 0 in results
    assert 1 in results
    assert isinstance(results[0], dict)
    assert isinstance(results[1], dict)
    assert (0, 0) in results[0]
    assert (0, 0) in results[1] or (0, 0) in results[0]


def test_to_sympy_returns_string():
    fitter = SymbolicFitter(KAN([1, 1]))
    result = {
        "fn": "sin",
        "r2": 0.995,
        "params": {"a": 1.0, "b": 2.0, "c": 0.5, "d": 0.1},
    }
    expr = fitter.to_sympy(result)
    assert isinstance(expr, str)
    assert expr
