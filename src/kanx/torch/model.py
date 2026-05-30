"""High-level PyTorch KAN model + ergonomic builders."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import torch
from torch import nn

from .layers import KANLinear

LayersSpec = Union[Sequence[int], Sequence[dict]]


class KAN(nn.Sequential):
    """PyTorch KAN model.

    Two equivalent constructors::

        KAN([2, 64, 1])                                # widths
        KAN([{"in_features": 2, "out_features": 64},   # per-layer dicts
             {"in_features": 64, "out_features": 1}])
    """

    def __init__(
        self,
        layers: LayersSpec,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        grid_range=(-1.0, 1.0),
    ) -> None:
        modules: List[nn.Module] = []
        defaults = dict(
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=base_activation,
            grid_range=grid_range,
        )

        layers = list(layers)
        if len(layers) == 0:
            raise ValueError("`layers` must not be empty")
        first = layers[0]
        if isinstance(first, dict):
            for cfg in layers:
                modules.append(KANLinear(**{**defaults, **cfg}))
        else:
            widths = [int(w) for w in layers]
            if len(widths) < 2:
                raise ValueError("widths must contain at least input and output dims")
            for i in range(len(widths) - 1):
                modules.append(
                    KANLinear(widths[i], widths[i + 1], **defaults)
                )

        super().__init__(*modules)
        self._layers_spec = layers
        self._defaults = defaults

    # ---- convenience ergonomics --------------------------------------------
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
        device: str | None = None,
        loss_fn=None,
    ):
        """Zero-friction fit: ``model.fit(X, y)`` — no optimizer or compile dance.

        Internally wraps :class:`kanx.torch.Trainer`; returns its
        ``TrainHistory`` so you get ``hist.loss`` and ``hist.val_loss``.
        """
        # Local import to avoid an import cycle.
        from .trainer import Trainer
        return Trainer(self, device=device, loss_fn=loss_fn).fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            val_split=val_split,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose,
        )

    @torch.no_grad()
    def predict(
        self,
        x,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Inference. Accepts list / numpy array / tensor. Returns a CPU tensor."""
        was_training = self.training
        self.eval()
        try:
            xt = _as_tensor(x)
            if batch_size is None or xt.shape[0] <= batch_size:
                return self(xt).cpu()
            chunks = []
            for i in range(0, xt.shape[0], batch_size):
                chunks.append(self(xt[i : i + batch_size]).cpu())
            return torch.cat(chunks, dim=0)
        finally:
            self.train(was_training)

    def save(self, path: str) -> str:
        """Save state_dict + architecture spec to a single .pt file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "layers_spec": self._layers_spec,
                "defaults": self._defaults,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "KAN":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt["layers_spec"], **ckpt["defaults"])
        model.load_state_dict(ckpt["state_dict"])
        return model


def build_kan(layers: Iterable, **kwargs) -> KAN:
    """Functional builder mirroring `KAN.__init__`."""
    return KAN(list(layers), **kwargs)


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"Inputs must be rank-1 or rank-2; got shape {tuple(t.shape)}")
    return t.float()
