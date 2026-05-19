from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from kanx import KAN


def test_push_to_hub_calls_upload():
    model = KAN([2, 8, 1])
    model(np.zeros((1, 2), dtype=np.float32))

    with patch("huggingface_hub.HfApi.create_repo") as create_repo, patch(
        "huggingface_hub.HfApi.upload_folder"
    ) as upload_folder:
        model.push_to_hub("user/repo", commit_message="test upload", private=True)

    create_repo.assert_called_once_with(repo_id="user/repo", private=True, exist_ok=True)
    upload_folder.assert_called_once()


def test_from_pretrained_loads_weights(tmp_path):
    model = KAN([2, 8, 1])
    model(np.zeros((1, 2), dtype=np.float32))

    model_path = Path(tmp_path) / "model.keras"
    config_path = Path(tmp_path) / "config.yaml"
    model.save(model_path)
    yaml.safe_dump({"model": model.get_config()}, open(config_path, "w", encoding="utf-8"))

    def fake_hf_hub_download(repo_id, filename, revision="main"):
        if filename == "model.keras":
            return str(model_path)
        if filename == "config.yaml":
            return str(config_path)
        raise ValueError(filename)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
        loaded = KAN.from_pretrained("user/repo")

    assert loaded._layers_spec == model._layers_spec
    assert loaded._defaults == model._defaults


def test_from_pretrained_torch(tmp_path):
    try:
        import torch

        from kanx.torch import KAN as TorchKAN
    except ImportError:
        return

    model = TorchKAN([2, 8, 1])
    model(torch.zeros((1, 2), dtype=torch.float32))

    model_path = Path(tmp_path) / "model.pt"
    config_path = Path(tmp_path) / "config.yaml"
    model.save(str(model_path))
    yaml.safe_dump(
        {"model": {**model._defaults, "layers": model._layers_spec, "name": model.__class__.__name__}},
        open(config_path, "w", encoding="utf-8"),
    )

    def fake_hf_hub_download(repo_id, filename, revision="main"):
        if filename == "model.pt":
            return str(model_path)
        if filename == "config.yaml":
            return str(config_path)
        raise ValueError(filename)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
        loaded = TorchKAN.from_pretrained("user/repo")

    assert loaded._layers_spec == model._layers_spec
    assert loaded._defaults == model._defaults
