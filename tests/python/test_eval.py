"""
Tests for drex.eval — PasskeyBenchmark.

Also covers drex/eval/__init__.py via the package-level import.
"""

import unittest.mock

import pytest
import torch

from drex.eval import PasskeyBenchmark  # covers eval/__init__.py
from drex.models.transformer import DrexConfig, DrexTransformer


@pytest.fixture
def cfg():
    return DrexConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_mult=2,
        vocab_size=128,
        window_size=32,
        max_seq_len=512,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(cfg, device):
    return DrexTransformer(cfg).to(torch.device(device))


@pytest.fixture
def bench(tiny_model, device):
    return PasskeyBenchmark(
        model=tiny_model,
        context_lengths=[200, 400],
        n_trials=2,
        device=torch.device(device),
        segment_len=32,
    )


class TestPasskeyBenchmark:
    def test_init_default_device(self, tiny_model):
        """device=None defaults to cpu."""
        b = PasskeyBenchmark(tiny_model, context_lengths=[100])
        assert b.device == torch.device("cpu")

    def test_make_prompt_normal_context(self, bench):
        """Normal context length: passkey embedded at ~50%."""
        tokens, passkey = bench._make_prompt(400, seed=0)
        assert isinstance(tokens, list)
        assert len(passkey) == 5
        assert passkey.isdigit()
        # All tokens clamped to ASCII range
        assert all(0 <= t <= 127 for t in tokens)

    def test_make_prompt_short_context_clamps_pos(self, bench):
        """Very short context triggers passkey_pos < len(prefix_toks) branch."""
        tokens, passkey = bench._make_prompt(50, seed=1)
        assert isinstance(tokens, list)
        assert passkey.isdigit()

    def test_make_prompt_distractor_zero_length(self, bench):
        """context so small that distractor_len_after hits max(0, ...) = 0."""
        tokens, passkey = bench._make_prompt(10, seed=2)
        assert isinstance(tokens, list)

    def test_greedy_generate_length(self, bench, tiny_model, device):
        """_greedy_generate returns exactly n_tokens integers."""
        dev = torch.device(device)
        prompt = torch.randint(0, 128, (1, 40), device=dev)
        generated = bench._greedy_generate(prompt, n_tokens=5)
        assert len(generated) == 5
        assert all(isinstance(t, int) for t in generated)

    def test_greedy_generate_multisegment(self, bench, device):
        """Prompt longer than segment_len exercises the segmented forward loop."""
        dev = torch.device(device)
        prompt = torch.randint(0, 128, (1, 100), device=dev)  # > segment_len=32
        generated = bench._greedy_generate(prompt, n_tokens=3)
        assert len(generated) == 3

    def test_run_returns_dict(self, bench):
        """run() returns a dict keyed by context_length with float values."""
        results = bench.run()
        assert set(results.keys()) == {200, 400}
        for v in results.values():
            assert 0.0 <= v <= 1.0

    def test_run_correct_prediction_branch(self, tiny_model, device):
        """Cover the `correct += 1` branch by mocking _greedy_generate."""
        dev = torch.device(device)
        bench_single = PasskeyBenchmark(
            model=tiny_model,
            context_lengths=[200],
            n_trials=1,  # one trial so seed=0 is the only trial
            device=dev,
            segment_len=32,
        )
        tokens, passkey = bench_single._make_prompt(200, seed=0)
        passkey_tokens = [ord(c) for c in passkey]

        with unittest.mock.patch.object(bench_single, "_greedy_generate", return_value=passkey_tokens):
            results = bench_single.run()

        assert results[200] == 1.0
