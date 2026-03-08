"""
Tests for drex.eval.babilong — BABILongBenchmark.
Also imports drex.eval package to cover eval/__init__.py new export.
"""

import unittest.mock

import pytest
import torch

from drex.eval import BABILongBenchmark  # covers the new eval/__init__.py export
from drex.models.transformer import DrexConfig, DrexTransformer


@pytest.fixture
def cfg():
    return DrexConfig(
        d_model=32,
        n_heads=2,
        n_layers=1,
        ff_mult=2,
        vocab_size=128,
        window_size=16,
        max_seq_len=512,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(cfg, device):
    return DrexTransformer(cfg).to(torch.device(device))


@pytest.fixture
def bench(tiny_model, device):
    return BABILongBenchmark(
        model=tiny_model,
        context_lengths=[150, 300],
        tasks=(1, 2, 3, 4, 5),
        n_trials=1,
        device=torch.device(device),
        segment_len=32,
    )


class TestBABILongBenchmarkInit:
    def test_default_device(self, tiny_model):
        b = BABILongBenchmark(tiny_model, context_lengths=[100])
        assert b.device == torch.device("cpu")

    def test_custom_device(self, tiny_model, device):
        dev = torch.device(device)
        b = BABILongBenchmark(tiny_model, context_lengths=[100], device=dev)
        assert b.device == dev


class TestEmbedInContext:
    def test_normal_context_false_branch(self, bench):
        """fact_pos >= len(prefix_toks) — the clamping branch is NOT taken."""
        tokens, answer = bench._embed_in_context(300, "Mary went to the garden. ", "Where?", "garden")
        assert isinstance(tokens, list)
        assert all(0 <= t <= 127 for t in tokens)
        assert isinstance(answer, str)

    def test_short_context_true_branch(self, bench):
        """Very short context forces fact_pos < len(prefix_toks) → clamp."""
        tokens, answer = bench._embed_in_context(10, "X. ", "Q?", "A")
        assert isinstance(tokens, list)

    def test_dist_after_clamped_to_zero(self, bench):
        """With minimal context, dist_after = max(0, negative) = 0."""
        tokens, answer = bench._embed_in_context(30, "Some facts. ", "Question?", "ans")
        assert isinstance(tokens, list)


class TestTaskGenerators:
    """Each _make_taskN method is called with normal and short context."""

    @pytest.mark.parametrize("ctx_len", [20, 300])
    def test_task1(self, bench, ctx_len):
        tokens, answer = bench._make_task1(ctx_len, seed=0)
        assert isinstance(tokens, list)
        assert isinstance(answer, str) and len(answer) > 0

    @pytest.mark.parametrize("ctx_len", [20, 300])
    def test_task2(self, bench, ctx_len):
        tokens, answer = bench._make_task2(ctx_len, seed=1)
        assert isinstance(tokens, list) and len(tokens) > 0
        assert answer in ["garden", "office", "kitchen", "hallway"]

    @pytest.mark.parametrize("ctx_len", [20, 300])
    def test_task3(self, bench, ctx_len):
        tokens, answer = bench._make_task3(ctx_len, seed=2)
        assert answer in ["garden", "office", "kitchen", "hallway"]

    @pytest.mark.parametrize("ctx_len", [20, 300])
    def test_task4(self, bench, ctx_len):
        tokens, answer = bench._make_task4(ctx_len, seed=3)
        assert answer in ["Mary", "John", "Sandra", "Daniel"]

    @pytest.mark.parametrize("ctx_len", [20, 300])
    def test_task5(self, bench, ctx_len):
        tokens, answer = bench._make_task5(ctx_len, seed=4)
        assert answer == "2"


class TestGreedyGenerate:
    def test_returns_correct_length(self, bench, device):
        dev = torch.device(device)
        prompt = torch.randint(0, 128, (1, 20), device=dev)
        generated = bench._greedy_generate(prompt, n_tokens=4)
        assert len(generated) == 4
        assert all(isinstance(t, int) for t in generated)

    def test_multisegment_prompt(self, bench, device):
        """Prompts longer than segment_len exercise the segmented loop."""
        dev = torch.device(device)
        prompt = torch.randint(0, 128, (1, 90), device=dev)  # > segment_len=32
        generated = bench._greedy_generate(prompt, n_tokens=2)
        assert len(generated) == 2


class TestRun:
    def test_returns_nested_dict(self, bench):
        results = bench.run()
        assert set(results.keys()) == {1, 2, 3, 4, 5}
        for task_dict in results.values():
            assert set(task_dict.keys()) == {150, 300}
            for v in task_dict.values():
                assert 0.0 <= v <= 1.0

    def test_correct_prediction_branch(self, tiny_model, device):
        """Cover `correct += 1` by mocking _greedy_generate."""
        dev = torch.device(device)
        b = BABILongBenchmark(
            tiny_model,
            context_lengths=[150],
            tasks=(1,),
            n_trials=1,
            device=dev,
            segment_len=32,
        )
        _, answer = b._make_task1(150, seed=0)
        answer_tokens = [ord(c) for c in answer]

        with unittest.mock.patch.object(b, "_greedy_generate", return_value=answer_tokens):
            results = b.run()

        assert results[1][150] == 1.0
