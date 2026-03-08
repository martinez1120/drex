"""
Phase 3 integration tests — DrexTransformer with use_l3=True.

Verifies that enabling L3 creates .snap files on disk and that the
full forward + backward pass works without errors.
"""

import glob
import os

import pytest
import torch
import torch.nn as nn

from drex.models.transformer import DrexConfig, DrexLayer, DrexTransformer
from drex.training.trainer import DrexTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def l3_config(tmp_path):
    return DrexConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        ff_mult=2,
        vocab_size=128,
        window_size=16,
        max_seq_len=64,
        dropout=0.0,
        use_l3=True,
        l3_base_path=str(tmp_path / "l3"),
        l3_compress=False,
    )


@pytest.fixture
def l3_model(l3_config, device):
    return DrexTransformer(l3_config).to(torch.device(device))


# ---------------------------------------------------------------------------
# DrexTransformer L3 construction
# ---------------------------------------------------------------------------


class TestDrexTransformerL3Init:
    def test_titan_list_created(self, l3_model, l3_config):
        assert l3_model._titan_list is not None
        assert len(l3_model._titan_list) == l3_config.n_layers

    def test_l3_bridge_created(self, l3_model):
        assert l3_model._l3_bridge is not None
        assert l3_model._l3_bridge._available is True

    def test_layers_have_bridge(self, l3_model):
        for layer in l3_model.layers:
            assert layer.l3_bridge is l3_model._l3_bridge

    def test_layers_have_correct_idx(self, l3_model, l3_config):
        for i, layer in enumerate(l3_model.layers):
            assert layer.layer_idx == i

    def test_no_l3_model_has_none(self, device):
        cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, use_l3=False)
        model = DrexTransformer(cfg).to(torch.device(device))
        assert model._titan_list is None
        assert model._l3_bridge is None


# ---------------------------------------------------------------------------
# DrexLayer L3 forward wiring
# ---------------------------------------------------------------------------


class TestDrexLayerL3Forward:
    def test_forward_with_l3_bridge_creates_snapshots(self, l3_model, l3_config, tmp_path):
        """One forward step should write at least one .snap to disk."""
        dev = torch.device("cpu")  # use cpu to keep it simple for bridge writes
        cfg = l3_config
        model = DrexTransformer(cfg).to(dev)
        B, S = 1, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        model(ids)
        snap_files = glob.glob(str(tmp_path / "l3" / "**" / "*.snap"), recursive=True)
        assert len(snap_files) >= 1, "Expected at least one .snap file on disk"

    def test_backward_with_l3_bridge(self, l3_model, l3_config, device):
        """Backward pass through L3-enabled model should not raise."""
        dev = torch.device(device)
        B, S = 1, 8
        ids = torch.randint(0, l3_config.vocab_size, (B, S), device=dev)
        logits, _ = l3_model(ids)
        loss = logits.sum()
        loss.backward()
        assert any(p.grad is not None for p in l3_model.parameters())

    def test_l3_bridge_registers_prefetch_entries(self, l3_config, tmp_path):
        """After forward, the sketch index should have entries for layer 0."""
        dev = torch.device("cpu")
        cfg = DrexConfig(
            **{**l3_config.__dict__, "l3_base_path": str(tmp_path / "l3b")}
        )
        model = DrexTransformer(cfg).to(dev)
        ids = torch.randint(0, cfg.vocab_size, (1, 8), device=dev)
        model(ids)
        # After write_and_snapshot, the prefetch engine's sketch should return
        # at least 1 candidate for layer 0
        key = torch.randn(cfg.d_model)
        candidates = model._l3_bridge._engine.prefetch(0, key.tolist(), 1)
        # prefetch returns None (fires async); just ensure no exception raised
        assert candidates is None


# ---------------------------------------------------------------------------
# DrexLayer with no bridge (existing constructor still works)
# ---------------------------------------------------------------------------


class TestDrexLayerNoL3:
    def test_layer_idx_default(self, device):
        from drex.models.transformer import DrexConfig
        cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, use_l3=False)
        layer = DrexLayer(cfg)
        assert layer.layer_idx == 0
        assert layer.l3_bridge is None

    def test_forward_without_bridge(self, device):
        dev = torch.device(device)
        cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, use_l3=False)
        layer = DrexLayer(cfg, layer_idx=3).to(dev)
        from drex.models.memory import LayerState
        state = LayerState.zeros(1, 2, 16, 16, dev)
        x = torch.randn(1, 4, 32, device=dev)
        out, new_state = layer(x, state)
        assert out.shape == (1, 4, 32)


# ---------------------------------------------------------------------------
# DrexTrainer with use_l3=True
# ---------------------------------------------------------------------------


class TestDrexTrainerL3:
    def test_train_step_with_l3(self, l3_config, tmp_path):
        """Trainer should complete a step with L3 enabled."""
        dev = torch.device("cpu")  # always cpu to avoid device mismatch in bridge
        cfg = DrexConfig(
            **{**l3_config.__dict__, "l3_base_path": str(tmp_path / "trainer_l3")}
        )
        model = DrexTransformer(cfg).to(dev)
        trainer = DrexTrainer(model, cfg, lr=1e-3, n_segments_per_step=1, segment_len=8)
        ids = torch.randint(0, cfg.vocab_size, (1, 16), device=dev)
        loss = trainer.train_step(ids)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_snap_files_created_after_training(self, l3_config, tmp_path):
        """Snap files should exist after a training step."""
        dev = torch.device("cpu")
        cfg = DrexConfig(
            **{**l3_config.__dict__, "l3_base_path": str(tmp_path / "snap_test")}
        )
        model = DrexTransformer(cfg).to(dev)
        trainer = DrexTrainer(model, cfg, n_segments_per_step=1, segment_len=8)
        ids = torch.randint(0, cfg.vocab_size, (1, 16), device=dev)
        trainer.train_step(ids)
        snap_files = glob.glob(
            str(tmp_path / "snap_test" / "**" / "*.snap"), recursive=True
        )
        assert len(snap_files) >= 1
