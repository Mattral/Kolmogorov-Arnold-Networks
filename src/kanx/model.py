"""High-level KAN model.

`KAN` is a thin convenience wrapper around `tf.keras.Sequential` that wires
together a list of `KANLinear` layers from a list of widths
(e.g. ``[2, 64, 64, 1]``) or a list of per-layer config dicts.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import tensorflow as tf

from .layers import KANLinear


LayersSpec = Union[Sequence[int], Sequence[dict]]


@tf.keras.utils.register_keras_serializable(package="kanx")
class KAN(tf.keras.Sequential):
    """Sequential KAN.

    Two equivalent ways to build it:

    >>> KAN([2, 64, 64, 1])                                       # widths
    >>> KAN([                                                     # per-layer
    ...     {"in_features": 2,  "out_features": 64},
    ...     {"in_features": 64, "out_features": 64},
    ...     {"in_features": 64, "out_features": 1},
    ... ])
    """

    def __init__(
        self,
        layers: LayersSpec,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "silu",
        regularization_factor: float = 0.0,
        grid_range=(-1.0, 1.0),
        name: str = "kan",
        **kwargs,
    ):
        super().__init__(name=name)
        self._layers_spec = list(layers)
        self._default_kwargs = {
            "grid_size": grid_size,
            "spline_order": spline_order,
            "base_activation": base_activation,
            "regularization_factor": regularization_factor,
            "grid_range": grid_range,
        }
        self._extra_kwargs = kwargs

        if len(self._layers_spec) == 0:
            raise ValueError("`layers` must not be empty")

        first = self._layers_spec[0]
        if isinstance(first, dict):
            for cfg in self._layers_spec:
                self.add(KANLinear(**{**self._default_kwargs, **cfg}))
        else:
            widths: List[int] = [int(w) for w in self._layers_spec]
            if len(widths) < 2:
                raise ValueError(
                    "When passing widths, provide at least 2 values "
                    "(input_dim, output_dim)."
                )
            for i in range(len(widths) - 1):
                self.add(
                    KANLinear(
                        in_features=widths[i],
                        out_features=widths[i + 1],
                        **self._default_kwargs,
                    )
                )

    # ---- convenience -------------------------------------------------------
    def fit(self, x=None, y=None, *args, **kwargs):
        """Zero-friction fit: auto-compiles with Adam(1e-3) + MSE if not compiled.

        This makes ``KAN([2, 32, 1]).fit(X, y, epochs=20)`` work out of the
        box without a separate ``model.compile(...)`` call.
        """
        if getattr(self, "optimizer", None) is None:
            self.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.MeanSquaredError(),
            )
        return super().fit(x, y, *args, **kwargs)

    def predict_tensor(self, x):
        """Tensor-in, tensor-out inference (no Keras progress bars / batching)."""
        return self(x, training=False)

    def update_grid_from_samples(self, x: tf.Tensor, margin: float = 0.01) -> None:
        """Update grids on all layers from input samples.
        
        For the first layer, update grid directly from input x.
        For subsequent layers, propagate x through prior layers.
        
        Args:
            x: (batch, in_features) input tensor to fit grid to.
            margin: margin applied to grid boundaries.
        """
        kan_layers = [layer for layer in self.layers if isinstance(layer, KANLinear)]
        if not kan_layers:
            return
        
        # Update first layer from raw input
        kan_layers[0].update_grid_from_samples(x, margin=margin)
        
        # Update remaining layers by propagating through prior layers
        current_x = x
        for i in range(1, len(kan_layers)):
            # Propagate through all layers up to this point
            for j in range(i):
                current_x = kan_layers[j](current_x)
            # Update this layer
            kan_layers[i].update_grid_from_samples(current_x, margin=margin)

    def get_config(self):
        return {
            "layers": self._layers_spec,
            **self._default_kwargs,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


def build_kan(
    layers: Iterable[int],
    *,
    grid_size: int = 5,
    spline_order: int = 3,
    base_activation: str = "silu",
    regularization_factor: float = 0.0,
    grid_range=(-1.0, 1.0),
) -> KAN:
    """Functional builder, mirrors `KAN.__init__` arguments."""
    return KAN(
        list(layers),
        grid_size=grid_size,
        spline_order=spline_order,
        base_activation=base_activation,
        regularization_factor=regularization_factor,
        grid_range=grid_range,
    )
