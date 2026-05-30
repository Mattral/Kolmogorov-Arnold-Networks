"""High-level training loop for the PyTorch backend.

Mirrors `kanx.train.train` (TF) at the surface level so users can switch
backends without rewriting their training script.

Usage::

    from kanx.torch import KAN, Trainer
    model = KAN([2, 64, 1])
    Trainer(model).fit(X, y, epochs=30, lr=1e-2, batch_size=64)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import KAN


@dataclass
class TrainHistory:
    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)


class Trainer:
    """Minimal, opinionated training loop.

    The point of `Trainer` is **not** to be a feature-rich PyTorch Lightning
    competitor — it's to give a one-liner that matches the TF backend's
    `kanx.train(cfg, X, y)` ergonomics.

    Args:
        model:      a `kanx.torch.KAN` (or any `nn.Module`).
        device:     "cpu", "cuda", or a `torch.device`. Default auto-detect.
        loss_fn:    a callable `(preds, targets) -> scalar`. Default MSE.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str | torch.device] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.history = TrainHistory()

    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        *,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        optimizer: str = "adam",
        val_split: float = 0.0,
        early_stopping_patience: int = 0,
        verbose: int = 1,
    ) -> TrainHistory:
        X = _as_tensor(X)
        y = _as_tensor(y)

        # train / val split
        if val_split > 0:
            n_val = max(1, int(len(X) * val_split))
            X_train, y_train = X[:-n_val], y[:-n_val]
            X_val, y_val = X[-n_val:], y[-n_val:]
        else:
            X_train, y_train = X, y
            X_val = y_val = None

        ds = TensorDataset(X_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = _build_optimizer(optimizer, self.model.parameters(), lr)

        best_loss = float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            n_seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                opt.step()
                running += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)
            train_loss = running / max(1, n_seen)
            self.history.loss.append(train_loss)

            val_str = ""
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    vp = self.model(X_val.to(self.device))
                    vloss = float(self.loss_fn(vp, y_val.to(self.device)).item())
                self.history.val_loss.append(vloss)
                val_str = f" | val_loss={vloss:.6f}"
                monitor = vloss
            else:
                monitor = train_loss

            if verbose:
                print(
                    f"epoch {epoch:3d}/{epochs} | loss={train_loss:.6f}{val_str}",
                    flush=True,
                )

            # Early stopping
            if early_stopping_patience > 0:
                if monitor < best_loss - 1e-9:
                    best_loss = monitor
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stopping_patience:
                        if verbose:
                            print(f"early-stop at epoch {epoch}", flush=True)
                        break
        return self.history


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name!r}")


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)
