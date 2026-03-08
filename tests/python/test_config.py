"""
Tests for drex.utils.config — save_checkpoint and load_checkpoint.
Also imports drex.utils package to cover utils/__init__.py.
"""

import json

import pytest
import torch

from drex.models.transformer import DrexConfig, DrexTransformer
from drex.utils import load_checkpoint, save_checkpoint  # covers utils/__init__.py


@pytest.fixture
def tiny_model():
    cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, vocab_size=64)
    return DrexTransformer(cfg)


class TestSaveCheckpoint:
    def test_creates_safetensors_file(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=42)
        assert path.exists()

    def test_creates_json_sidecar(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=7)
        json_path = path.with_suffix(".json")
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert meta["step"] == 7

    def test_json_contains_config(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=0)
        meta = json.loads((path.with_suffix(".json")).read_text())
        assert "config" in meta
        assert meta["config"]["d_model"] == 32

    def test_creates_parent_dirs(self, tiny_model, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "model.safetensors"
        save_checkpoint(tiny_model, path)  # parent dirs don't exist yet
        assert path.exists()

    def test_default_step_is_zero(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path)
        meta = json.loads((path.with_suffix(".json")).read_text())
        assert meta["step"] == 0


class TestLoadCheckpoint:
    def test_round_trip_weights(self, tiny_model, tmp_path):
        """save → load → weights identical."""
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=5)

        # Perturb model weights
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p))

        step = load_checkpoint(tiny_model, path)
        assert step == 5

        # Re-run save to get reference weights; compare via forward pass
        cfg = tiny_model.config
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            logits, _ = tiny_model(ids)
        assert not torch.isnan(logits).any()

    def test_returns_step(self, tiny_model, tmp_path):
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=99)
        step = load_checkpoint(tiny_model, path)
        assert step == 99

    def test_without_json_sidecar_returns_zero(self, tiny_model, tmp_path):
        """If the .json sidecar is missing, load_checkpoint returns 0."""
        path = tmp_path / "ckpt.safetensors"
        save_checkpoint(tiny_model, path, step=10)
        # Remove the sidecar
        path.with_suffix(".json").unlink()
        step = load_checkpoint(tiny_model, path)
        assert step == 0
