"""
Tests for drex.eval.passkey — PasskeyBenchmark.
"""

import pytest
import torch

import drex.eval  # ensures drex/eval/__init__.py is covered
from drex.eval.passkey import PasskeyBenchmark, _DISTRACTOR, _PREFIX, _QUESTION
from drex.models.transformer import DrexConfig, DrexTransformer


# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model(device):
    """Untrained tiny DrexTransformer with ASCII-range vocab for passkey tests."""
    cfg = DrexConfig(
        d_model=32,
        n_heads=2,
        n_layers=1,
        ff_mult=2,
        vocab_size=128,
        window_size=64,
        max_seq_len=512,
        dropout=0.0,
        use_l3=False,
    )
    dev = torch.device(device)
    return DrexTransformer(cfg).to(dev)


@pytest.fixture(scope="module")
def bench(tiny_model, device):
    return PasskeyBenchmark(
        model=tiny_model,
        context_lengths=[256, 512],
        n_trials=2,
        device=torch.device(device),
        segment_len=64,
    )


# ---------------------------------------------------------------------------
# _make_prompt
# ---------------------------------------------------------------------------


class TestMakePrompt:
    def test_returns_tokens_and_passkey(self, bench):
        tokens, passkey = bench._make_prompt(context_len=400, seed=0)
        assert isinstance(tokens, list)
        assert isinstance(passkey, str)
        assert len(passkey) == 5
        assert passkey.isdigit()

    def test_passkey_embedded_in_tokens(self, bench):
        tokens, passkey = bench._make_prompt(context_len=400, seed=42)
        text = "".join(chr(t) for t in tokens if 32 <= t < 127)
        assert passkey in text

    def test_tokens_in_valid_range(self, bench):
        tokens, _ = bench._make_prompt(context_len=300, seed=7)
        assert all(0 <= t <= 127 for t in tokens)

    def test_short_context_triggers_passkey_pos_adjustment(self, bench):
        """For very short context the passkey_pos < len(prefix_toks) branch is hit."""
        # prefix_toks is ~76 chars; passkey_toks ~45; passkey_pos=50//2-45=5 < 76
        tokens, passkey = bench._make_prompt(context_len=100, seed=1)
        text = "".join(chr(t) for t in tokens if 32 <= t < 127)
        assert passkey in text

    def test_distractor_after_can_be_zero(self, bench):
        """Very short context: distractor_len_after = max(0, negative) → 0."""
        # With context_len=150, there may be no room for distractors after passkey
        tokens, passkey = bench._make_prompt(context_len=150, seed=3)
        text = "".join(chr(t) for t in tokens if 32 <= t < 127)
        assert passkey in text


# ---------------------------------------------------------------------------
# _greedy_generate
# ---------------------------------------------------------------------------


class TestGreedyGenerate:
    def test_generates_correct_number_of_tokens(self, bench):
        tokens, _ = bench._make_prompt(context_len=256, seed=0)
        prompt = torch.tensor([tokens], dtype=torch.long, device=bench.device)
        generated = bench._greedy_generate(prompt, n_tokens=5)
        assert len(generated) == 5

    def test_generated_tokens_in_valid_range(self, bench):
        tokens, _ = bench._make_prompt(context_len=256, seed=0)
        prompt = torch.tensor([tokens], dtype=torch.long, device=bench.device)
        generated = bench._greedy_generate(prompt, n_tokens=3)
        for tok in generated:
            assert 0 <= tok < bench.model.config.vocab_size

    def test_prompt_longer_than_segment_uses_multiple_segments(self, bench):
        """Prompt larger than segment_len exercises the segment loop."""
        tokens, _ = bench._make_prompt(context_len=256, seed=5)
        # 256 tokens > segment_len=64, so at least 4 segments processed
        prompt = torch.tensor([tokens], dtype=torch.long, device=bench.device)
        generated = bench._greedy_generate(prompt, n_tokens=5)
        assert len(generated) == 5


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_returns_dict_with_correct_keys(self, bench):
        results = bench.run()
        assert set(results.keys()) == {256, 512}

    def test_accuracies_are_in_range(self, bench):
        results = bench.run()
        for acc in results.values():
            assert 0.0 <= acc <= 1.0

    def test_run_single_context_length(self, tiny_model, device):
        """Smoke-test with a single context length and 1 trial."""
        b = PasskeyBenchmark(
            model=tiny_model,
            context_lengths=[256],
            n_trials=1,
            device=torch.device(device),
            segment_len=64,
        )
        results = b.run()
        assert 256 in results
        assert 0.0 <= results[256] <= 1.0

    def test_run_correct_answer_increments_accuracy(self, tiny_model, device):
        """
        Monkeypatch _greedy_generate to return the exact passkey chars.
        Exercises the `if passkey in gen_str` branch (line 155).
        """
        b = PasskeyBenchmark(
            model=tiny_model,
            context_lengths=[256],
            n_trials=1,
            device=torch.device(device),
            segment_len=64,
        )
        # Determine the passkey that will be used for seed=0
        _, passkey = b._make_prompt(256, seed=0)
        passkey_ascii = [ord(c) for c in passkey]

        # Replace generation with a stub that always returns the passkey
        b._greedy_generate = lambda prompt, n_tokens: passkey_ascii[:n_tokens]

        results = b.run()
        assert results[256] == 1.0

