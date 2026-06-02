from __future__ import annotations

from .feynman import (
    feynman_I_9_18,
    feynman_I_34_8,
    feynman_II_11_27,
    make_dataset_I_9_18,
    make_dataset_I_34_8,
    make_dataset_II_11_27,
)
from .tabular import (
    load_california_housing,
    load_concrete_strength,
    load_energy_efficiency,
)

__all__ = [
    "feynman_I_9_18",
    "feynman_I_34_8",
    "feynman_II_11_27",
    "make_dataset_I_9_18",
    "make_dataset_I_34_8",
    "make_dataset_II_11_27",
    "load_california_housing",
    "load_concrete_strength",
    "load_energy_efficiency",
    "make_dataset",
]

_DEFAULT_FUZZY_DATASETS = {
    "feynman_I_9_18": make_dataset_I_9_18,
    "feynman_I_34_8": make_dataset_I_34_8,
    "feynman_II_11_27": make_dataset_II_11_27,
}


def make_dataset(name: str = "feynman_I_9_18", **kwargs):
    """Return a named Feynman benchmark dataset.

    Parameters
    ----------
    name:
        One of the supported Feynman dataset factories.
    kwargs:
        Passed through to the selected dataset factory.
    """
    if name not in _DEFAULT_FUZZY_DATASETS:
        raise ValueError(
            f"Unknown dataset name: {name}. Supported names: {sorted(_DEFAULT_FUZZY_DATASETS)}"
        )
    return _DEFAULT_FUZZY_DATASETS[name](**kwargs)
