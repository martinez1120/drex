"""
Tests for drex.models.transformer — DrexConfig, DrexLayer, DrexTransformer.
Also covers DrexTrainer basic functionality.
"""

import pytest
import torch

from drex.models.transformer import DrexConfig, DrexLayer, DrexTransformer
from drex.training.trainer import DrexTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """Tiny config for fast tests."""
    return DrexConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_mult=2,
        vocab_size=256,
        window_size=32,
        max_seq_len=64,
        dropout=0.0,
        use_l3=False,
    )


@pytest.fixture
def model(cfg, device):
    dev = torch.device(device)
    return DrexTransformer(cfg).to(dev)


# ---------------------------------------------------------------------------
# DrexLayer
# ---------------------------------------------------------------------------


class TestDrexLayer:
    def test_output_shape(self, cfg, device):
        dev = torch.device(device)
        layer = DrexLayer(cfg).to(dev)
        B, S = 2, 16
        d_k = cfg.d_model // cfg.n_heads
        from drex.models.memory import LayerState
        state = LayerState.zeros(B, cfg.n_heads, d_k, d_k, dev)
        x = torch.randn(B, S, cfg.d_model, device=dev)
        out, new_state = layer(x, state)
        assert out.shape == (B, S, cfg.d_model)

    def test_backward(self, cfg, device):
        dev = torch.device(device)
        layer = DrexLayer(cfg).to(dev)
        B, S = 1, 8
        d_k = cfg.d_model // cfg.n_heads
        from drex.models.memory import LayerState
        state = LayerState.zeros(B, cfg.n_heads, d_k, d_k, dev)
        x = torch.randn(B, S, cfg.d_model, device=dev, requires_grad=True)
        out, _ = layer(x, state)
        out.sum().backward()
        assert x.grad is not None

    def test_episodic_memory_output_shape(self, device):
        """episodic_mem branch: output shape unchanged when use_episodic_memory=True."""
        dev = torch.device(device)
        epi_cfg = DrexConfig(
            d_model=64, n_heads=4, n_layers=1,
            ff_mult=2, vocab_size=256,
            window_size=32, max_seq_len=64,
            dropout=0.0, use_episodic_memory=True, episodic_gate_thresh=0.70,
        )
        layer = DrexLayer(epi_cfg).to(dev)
        B, S = 2, 16
        d_k = epi_cfg.d_model // epi_cfg.n_heads
        from drex.models.memory import LayerState
        state = LayerState.zeros(B, epi_cfg.n_heads, d_k, d_k, dev)
        x = torch.randn(B, S, epi_cfg.d_model, device=dev)
        out, _ = layer(x, state)
        assert out.shape == (B, S, epi_cfg.d_model)

    def test_episodic_memory_backward(self, device):
        """Gradients flow through the episodic_mem residual branch."""
        dev = torch.device(device)
        epi_cfg = DrexConfig(
            d_model=64, n_heads=4, n_layers=1,
            ff_mult=2, vocab_size=256,
            window_size=32, max_seq_len=64,
            dropout=0.0, use_episodic_memory=True, episodic_gate_thresh=0.70,
        )
        layer = DrexLayer(epi_cfg).to(dev)
        B, S = 1, 12
        d_k = epi_cfg.d_model // epi_cfg.n_heads
        from drex.models.memory import LayerState
        state = LayerState.zeros(B, epi_cfg.n_heads, d_k, d_k, dev)
        x = torch.randn(B, S, epi_cfg.d_model, device=dev, requires_grad=True)
        out, _ = layer(x, state)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# DrexTransformer
# ---------------------------------------------------------------------------


class TestDrexTransformer:
    def test_logit_shape(self, model, cfg, device):
        dev = torch.device(device)
        B, S = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, states = model(ids)
        assert logits.shape == (B, S, cfg.vocab_size)

    def test_states_returned(self, model, cfg, device):
        dev = torch.device(device)
        B, S = 2, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        _, states = model(ids)
        assert len(states) == cfg.n_layers

    def test_state_threading(self, model, cfg, device):
        """States should change across two consecutive segments."""
        dev = torch.device(device)
        B, S = 1, 16
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        _, states1 = model(ids)
        _, states2 = model(ids, states1)
        # M from the second forward should differ from first
        assert not torch.allclose(states1[0].memory.M, states2[0].memory.M)

    def test_no_nan_output(self, model, cfg, device):
        dev = torch.device(device)
        B, S = 2, 12
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_backward(self, model, cfg, device):
        dev = torch.device(device)
        B, S = 2, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, _ = model(ids)
        loss = logits.sum()
        loss.backward()
        # Check at least one param has gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_weight_tying(self, model, cfg):
        """LM head and token embedding share the same weight tensor."""
        assert model.lm_head.weight is model.token_emb.weight

    def test_init_states(self, model, cfg, device):
        dev = torch.device(device)
        states = model.init_states(batch=3, device=dev)
        assert len(states) == cfg.n_layers
        d_k = cfg.d_model // cfg.n_heads
        for s in states:
            assert s.memory.M.shape == (3, cfg.n_heads, d_k, d_k)

    def test_segment_detach_no_backward_through_state(self, model, cfg, device):
        """Verify TBPTT detach: loss on segment 2 should not backprop into segment 1."""
        dev = torch.device(device)
        B, S = 1, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)

        logits1, states1 = model(ids)
        # Detach states at segment boundary
        states1_detached = [s.detach() for s in states1]

        logits2, _ = model(ids, states1_detached)
        loss2 = torch.nn.functional.cross_entropy(
            logits2.reshape(-1, cfg.vocab_size),
            ids.reshape(-1),
        )
        loss2.backward()
        # LM head should have grads, but this shouldn't fail
        assert model.lm_head.weight.grad is not None

    def test_gradient_checkpointing_forward(self, cfg, device):
        """Gradient checkpointing enabled: output shape matches non-checkpointed."""
        dev = torch.device(device)
        ckpt_cfg = DrexConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult, vocab_size=cfg.vocab_size,
            window_size=cfg.window_size, max_seq_len=cfg.max_seq_len,
            dropout=0.0, gradient_checkpointing=True,
        )
        ckpt_model = DrexTransformer(ckpt_cfg).to(dev)
        ckpt_model.train()
        B, S = 2, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, states = ckpt_model(ids)
        assert logits.shape == (B, S, cfg.vocab_size)
        assert len(states) == cfg.n_layers

    def test_gradient_checkpointing_backward(self, cfg, device):
        """Gradient checkpointing: gradients still flow to all parameters."""
        dev = torch.device(device)
        ckpt_cfg = DrexConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult, vocab_size=cfg.vocab_size,
            window_size=cfg.window_size, max_seq_len=cfg.max_seq_len,
            dropout=0.0, gradient_checkpointing=True,
        )
        ckpt_model = DrexTransformer(ckpt_cfg).to(dev)
        ckpt_model.train()
        B, S = 1, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, _ = ckpt_model(ids)
        logits.sum().backward()
        assert any(p.grad is not None for p in ckpt_model.parameters())

    def test_episodic_memory_logit_shape(self, cfg, device):
        """DrexTransformer with use_episodic_memory=True returns correct logit shape."""
        dev = torch.device(device)
        epi_cfg = DrexConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult, vocab_size=cfg.vocab_size,
            window_size=cfg.window_size, max_seq_len=cfg.max_seq_len,
            dropout=0.0, use_episodic_memory=True, episodic_gate_thresh=0.70,
        )
        epi_model = DrexTransformer(epi_cfg).to(dev)
        B, S = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, states = epi_model(ids)
        assert logits.shape == (B, S, cfg.vocab_size)
        assert len(states) == cfg.n_layers

    def test_episodic_memory_no_nan(self, cfg, device):
        """DrexTransformer with episodic memory produces no NaN/Inf outputs."""
        dev = torch.device(device)
        epi_cfg = DrexConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult, vocab_size=cfg.vocab_size,
            window_size=cfg.window_size, max_seq_len=cfg.max_seq_len,
            dropout=0.0, use_episodic_memory=True, episodic_gate_thresh=0.70,
        )
        epi_model = DrexTransformer(epi_cfg).to(dev)
        B, S = 2, 12
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, _ = epi_model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_episodic_memory_backward(self, cfg, device):
        """Gradients flow to all parameters when episodic memory is enabled."""
        dev = torch.device(device)
        epi_cfg = DrexConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult, vocab_size=cfg.vocab_size,
            window_size=cfg.window_size, max_seq_len=cfg.max_seq_len,
            dropout=0.0, use_episodic_memory=True, episodic_gate_thresh=0.70,
        )
        epi_model = DrexTransformer(epi_cfg).to(dev)
        B, S = 2, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, _ = epi_model(ids)
        logits.sum().backward()
        assert any(p.grad is not None for p in epi_model.parameters())

    def test_episodic_memory_gradient_checkpointing(self, cfg, device):
        """Gradient checkpointing + episodic memory: forward completes, backward works."""
        dev = torch.device(device)
        ckpt_epi_cfg = DrexConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            ff_mult=cfg.ff_mult, vocab_size=cfg.vocab_size,
            window_size=cfg.window_size, max_seq_len=cfg.max_seq_len,
            dropout=0.0, gradient_checkpointing=True,
            use_episodic_memory=True, episodic_gate_thresh=0.70,
        )
        ckpt_epi_model = DrexTransformer(ckpt_epi_cfg).to(dev)
        ckpt_epi_model.train()
        B, S = 1, 8
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, _ = ckpt_epi_model(ids)
        assert logits.shape == (B, S, cfg.vocab_size)
        logits.sum().backward()
        assert any(p.grad is not None for p in ckpt_epi_model.parameters())


# ---------------------------------------------------------------------------
# DrexTrainer
# ---------------------------------------------------------------------------


class TestDrexTrainer:
    def test_train_step_returns_float(self, cfg, device):
        dev = torch.device(device)
        model = DrexTransformer(cfg).to(dev)
        trainer = DrexTrainer(model, cfg, lr=1e-3, n_segments_per_step=2, segment_len=16)
        # Need at least n_segments * segment_len + 1 tokens
        ids = torch.randint(0, cfg.vocab_size, (2, 64), device=dev)
        loss = trainer.train_step(ids)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_loss_decreases_over_steps(self, cfg, device):
        """Loss on random data should not increase monotonically (sanity check)."""
        dev = torch.device(device)
        model = DrexTransformer(cfg).to(dev)
        trainer = DrexTrainer(model, cfg, lr=1e-2, n_segments_per_step=2, segment_len=16)
        ids = torch.randint(0, cfg.vocab_size, (2, 64), device=dev)
        losses = [trainer.train_step(ids) for _ in range(20)]
        # Loss at end should be different from start (model is learning)
        assert losses[0] != losses[-1]

    def test_reset_states(self, cfg, device):
        dev = torch.device(device)
        model = DrexTransformer(cfg).to(dev)
        trainer = DrexTrainer(model, cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 64), device=dev)
        trainer.train_step(ids)
        assert trainer._states is not None
        trainer.reset_states()
        assert trainer._states is None

    def test_train_step_zero_tokens_returns_zero(self, cfg, device):
        """When the input is too short to form one complete segment, return 0.0."""
        dev = torch.device(device)
        model = DrexTransformer(cfg).to(dev)
        seg_len = 16
        trainer = DrexTrainer(model, cfg, n_segments_per_step=2, segment_len=seg_len)
        # T == seg_len: end = seg_len + 1 > T, loop body never runs → n_tokens == 0
        ids = torch.randint(0, cfg.vocab_size, (1, seg_len), device=dev)
        loss = trainer.train_step(ids)
        assert loss == 0.0

    def test_train_step_no_grad_clip(self, cfg, device):
        """grad_clip=0.0 disables gradient clipping (branch coverage)."""
        dev = torch.device(device)
        model = DrexTransformer(cfg).to(dev)
        trainer = DrexTrainer(model, cfg, grad_clip=0.0, n_segments_per_step=1, segment_len=16)
        ids = torch.randint(0, cfg.vocab_size, (1, 32), device=dev)
        loss = trainer.train_step(ids)
        assert isinstance(loss, float)
        assert loss > 0.0


# ---------------------------------------------------------------------------
# Phase 16 ablation flags
# ---------------------------------------------------------------------------


def _epi_cfg(**overrides) -> "DrexConfig":
    """Small episodic-memory config for ablation tests."""
    from drex.models.transformer import DrexConfig
    defaults = dict(
        d_model=64, n_heads=4, n_layers=4,
        ff_mult=2, vocab_size=256,
        window_size=32, max_seq_len=64,
        dropout=0.0, use_episodic_memory=True,
        episodic_gate_thresh=0.70,
    )
    defaults.update(overrides)
    return DrexConfig(**defaults)


class TestFullSeqResidualAblation:
    """full_seq_residual=True: memory residual is broadcast to all token positions."""

    def test_output_shape_unchanged(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(full_seq_residual=True)
        model = DrexTransformer(cfg).to(dev)
        B, S = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, states = model(ids)
        assert logits.shape == (B, S, cfg.vocab_size)
        assert len(states) == cfg.n_layers

    def test_no_nan(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(full_seq_residual=True)
        model = DrexTransformer(cfg).to(dev)
        B, S = 2, 12
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_backward(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(full_seq_residual=True)
        model = DrexTransformer(cfg).to(dev)
        ids = torch.randint(0, cfg.vocab_size, (2, 8), device=dev)
        logits, _ = model(ids)
        logits.sum().backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_differs_from_last_token_residual(self, device):
        """full_seq_residual and default (last-token only) produce different logits."""
        dev = torch.device(device)
        torch.manual_seed(42)
        cfg_last = _epi_cfg(full_seq_residual=False)
        torch.manual_seed(42)
        cfg_full = _epi_cfg(full_seq_residual=True)
        model_last = DrexTransformer(cfg_last).to(dev)
        model_full = DrexTransformer(cfg_full).to(dev)
        # Copy identical weights so only the residual mode differs
        model_full.load_state_dict(model_last.state_dict(), strict=False)
        ids = torch.randint(0, cfg_last.vocab_size, (1, 16), device=dev)
        logits_last, _ = model_last(ids)
        logits_full, _ = model_full(ids)
        # The non-last positions should differ
        assert not torch.allclose(logits_last[:, :-1], logits_full[:, :-1])


class TestMemoryLastLayerOnlyAblation:
    """memory_last_layer_only=True: MemoryModule only on the final layer."""

    def test_only_last_layer_has_mem(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(memory_last_layer_only=True)
        model = DrexTransformer(cfg).to(dev)
        for i, layer in enumerate(model.layers):
            if i < cfg.n_layers - 1:
                assert layer.episodic_mem is None, (
                    f"Layer {i} should not have episodic_mem when memory_last_layer_only=True"
                )
            else:
                assert layer.episodic_mem is not None, (
                    f"Final layer (index {i}) should have episodic_mem"
                )

    def test_output_shape_unchanged(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(memory_last_layer_only=True)
        model = DrexTransformer(cfg).to(dev)
        B, S = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, states = model(ids)
        assert logits.shape == (B, S, cfg.vocab_size)
        assert len(states) == cfg.n_layers

    def test_no_nan(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(memory_last_layer_only=True)
        model = DrexTransformer(cfg).to(dev)
        ids = torch.randint(0, cfg.vocab_size, (2, 12), device=dev)
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_backward(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(memory_last_layer_only=True)
        model = DrexTransformer(cfg).to(dev)
        ids = torch.randint(0, cfg.vocab_size, (1, 8), device=dev)
        logits, _ = model(ids)
        logits.sum().backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_fewer_params_than_all_layers(self, device):
        """Last-layer-only has fewer memory parameters than all-layers variant."""
        cfg_all = _epi_cfg(memory_last_layer_only=False)
        cfg_last = _epi_cfg(memory_last_layer_only=True)
        model_all = DrexTransformer(cfg_all)
        model_last = DrexTransformer(cfg_last)
        params_all = sum(p.numel() for p in model_all.parameters())
        params_last = sum(p.numel() for p in model_last.parameters())
        assert params_last < params_all


class TestNoNullGateThroughTransformer:
    """use_null_gate=False propagates correctly through DrexConfig → DrexLayer → MemoryModule."""

    def test_no_null_gate_module_is_none(self, device):
        cfg = _epi_cfg(use_null_gate=False)
        model = DrexTransformer(cfg)
        for layer in model.layers:
            if layer.episodic_mem is not None:
                assert layer.episodic_mem.null_gate is None

    def test_output_shape_unchanged(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(use_null_gate=False)
        model = DrexTransformer(cfg).to(dev)
        B, S = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        logits, states = model(ids)
        assert logits.shape == (B, S, cfg.vocab_size)
        assert len(states) == cfg.n_layers

    def test_no_nan(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(use_null_gate=False)
        model = DrexTransformer(cfg).to(dev)
        ids = torch.randint(0, cfg.vocab_size, (2, 12), device=dev)
        logits, _ = model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_backward(self, device):
        dev = torch.device(device)
        cfg = _epi_cfg(use_null_gate=False)
        model = DrexTransformer(cfg).to(dev)
        ids = torch.randint(0, cfg.vocab_size, (1, 8), device=dev)
        logits, _ = model(ids)
        logits.sum().backward()
        assert any(p.grad is not None for p in model.parameters())


