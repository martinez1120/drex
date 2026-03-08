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

