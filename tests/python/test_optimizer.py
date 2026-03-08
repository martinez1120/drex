"""
Tests for drex.training.optimizer — build_optimizer and cosine_schedule_with_warmup.
"""

import math

import pytest
import torch

from drex.models.transformer import DrexConfig, DrexTransformer
from drex.training.optimizer import build_optimizer, cosine_schedule_with_warmup


@pytest.fixture
def tiny_model():
    cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, vocab_size=64)
    return DrexTransformer(cfg)


@pytest.fixture
def tiny_optim(tiny_model):
    return build_optimizer(tiny_model, lr=1e-3, weight_decay=0.1)


class TestBuildOptimizer:
    def test_returns_two_groups(self, tiny_optim):
        assert len(tiny_optim.param_groups) == 2

    def test_decay_group_has_weight_decay(self, tiny_optim):
        assert tiny_optim.param_groups[0]["weight_decay"] == pytest.approx(0.1)

    def test_no_decay_group_zero_decay(self, tiny_optim):
        assert tiny_optim.param_groups[1]["weight_decay"] == pytest.approx(0.0)

    def test_no_decay_covers_bias_and_norm(self, tiny_model):
        """All bias and norm parameters land in the no-decay group."""
        optim = build_optimizer(tiny_model)
        # Collect all parameter objects in the no-decay group
        no_decay_params = set(id(p) for p in optim.param_groups[1]["params"])
        for name, param in tiny_model.named_parameters():
            if "bias" in name or "norm" in name.lower():
                assert id(param) in no_decay_params, f"Expected {name!r} in no-decay group"

    def test_decay_covers_weight_matrices(self, tiny_model):
        """Linear weight matrices (no norm/bias in name) land in the decay group."""
        optim = build_optimizer(tiny_model)
        decay_params = set(id(p) for p in optim.param_groups[0]["params"])
        for name, param in tiny_model.named_parameters():
            if "bias" not in name and "norm" not in name.lower():
                assert id(param) in decay_params, f"Expected {name!r} in decay group"

    def test_total_params_match_model(self, tiny_model):
        """Every requires_grad param appears in exactly one group."""
        optim = build_optimizer(tiny_model)
        grouped = set()
        for g in optim.param_groups:
            for p in g["params"]:
                grouped.add(id(p))
        model_params = {id(p) for p in tiny_model.parameters() if p.requires_grad}
        assert grouped == model_params

    def test_custom_lr_and_betas(self, tiny_model):
        optim = build_optimizer(tiny_model, lr=5e-4, weight_decay=0.05, betas=(0.8, 0.9))
        assert optim.defaults["lr"] == pytest.approx(5e-4)
        assert optim.defaults["betas"] == (0.8, 0.9)


class TestCosineScheduleWithWarmup:
    def _get_lambda(self, warmup, total, min_lr=0.1, tiny_optim=None):
        if tiny_optim is None:
            cfg = DrexConfig(d_model=32, n_heads=2, n_layers=1, ff_mult=2, vocab_size=64)
            model = DrexTransformer(cfg)
            tiny_optim = build_optimizer(model, lr=1e-3)
        sched = cosine_schedule_with_warmup(tiny_optim, warmup, total, min_lr)
        return sched.lr_lambdas[0]

    def test_warmup_phase_zero(self):
        """At step 0 with warmup_steps=10, factor = 0/10 = 0."""
        fn = self._get_lambda(warmup=10, total=100)
        assert fn(0) == pytest.approx(0.0)

    def test_warmup_phase_midpoint(self):
        """At step 5 with warmup_steps=10, factor = 5/10 = 0.5."""
        fn = self._get_lambda(warmup=10, total=100)
        assert fn(5) == pytest.approx(0.5)

    def test_warmup_full(self):
        """At step = warmup_steps, factor ≈ 1.0 (transitions to cosine)."""
        fn = self._get_lambda(warmup=10, total=100)
        # step=10 still falls into warmup branch (< not <=): factor = 10/10 = 1
        assert fn(10) == pytest.approx(1.0)

    def test_cosine_phase_midpoint(self):
        """At step 50 with warmup=0, total=100, progress=0.5 → cosine midpoint."""
        fn = self._get_lambda(warmup=0, total=100)
        progress = 50 / 100
        expected = 0.1 + 0.5 * (1.0 - 0.1) * (1.0 + math.cos(math.pi * progress))
        assert fn(50) == pytest.approx(expected)

    def test_at_total_steps_returns_min_lr(self):
        """At step == total_steps, scheduler returns min_lr_ratio."""
        fn = self._get_lambda(warmup=5, total=50, min_lr=0.05)
        assert fn(50) == pytest.approx(0.05)

    def test_past_total_steps_returns_min_lr(self):
        """Steps beyond total_steps also return min_lr_ratio."""
        fn = self._get_lambda(warmup=5, total=50, min_lr=0.05)
        assert fn(200) == pytest.approx(0.05)

    def test_schedule_integrates_with_step(self, tiny_optim):
        """Calling scheduler.step() correctly updates the optimizer LR."""
        base_lr = tiny_optim.param_groups[0]["lr"]  # capture before scheduler zeroes it
        sched = cosine_schedule_with_warmup(tiny_optim, warmup_steps=2, total_steps=10)
        # After LambdaLR init, lr = base_lr * fn(0) = 0
        assert tiny_optim.param_groups[0]["lr"] == pytest.approx(0.0)
        sched.step()  # step 1 → warmup: 1/2 = 0.5
        assert tiny_optim.param_groups[0]["lr"] == pytest.approx(base_lr * 0.5)
