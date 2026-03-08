"""
drex.eval.passkey — Passkey retrieval benchmark.

Inserts a 5-digit passkey at a controlled depth in a distractor context,
then evaluates whether the model can retrieve it at the end of a long sequence.

Reference: Mohtashami & Jaggi, "Landmark Attention", 2023.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import torch

from drex.models.memory import LayerState
from drex.models.transformer import DrexTransformer


_DISTRACTOR = (
    "The quick brown fox jumps over the lazy dog. "
)

_PREFIX = "There is an important number planted in the following text. Find it and memorize it. "
_QUESTION = " What is the important number that was planted in the text? The important number is: "


class PasskeyBenchmark:
    """
    Evaluate passkey retrieval accuracy over a range of context lengths.

    The benchmark encodes text as raw character-level integers (0-127) to
    keep the vocab simple and avoid needing a trained tokenizer.
    The model must output the 5-digit passkey verbatim.
    """

    def __init__(
        self,
        model: DrexTransformer,
        context_lengths: list[int],
        n_trials: int = 20,
        device: Optional[torch.device] = None,
        segment_len: int = 512,
    ) -> None:
        self.model = model
        self.context_lengths = context_lengths
        self.n_trials = n_trials
        self.device = device or torch.device("cpu")
        self.segment_len = segment_len

    # ------------------------------------------------------------------

    def _make_prompt(self, context_len: int, seed: int) -> tuple[list[int], str]:
        """
        Build a token list (char-level ASCII) of approximately context_len tokens
        with a 5-digit passkey embedded at ~50% depth.
        Returns (token_ids, passkey_str).
        """
        rng = random.Random(seed)
        passkey = str(rng.randint(10_000, 99_999))

        # Build the distractor body first to estimate token count
        prefix_toks = [ord(c) for c in _PREFIX]
        passkey_sentence = f"The important number is: {passkey}. Remember it. "
        passkey_toks = [ord(c) for c in passkey_sentence]
        question_toks = [ord(c) for c in _QUESTION]

        # Target: embed passkey at ~50% of context_len
        passkey_pos = (context_len // 2) - len(passkey_toks)
        if passkey_pos < len(prefix_toks):
            passkey_pos = len(prefix_toks)

        distractor_len_before = passkey_pos - len(prefix_toks)
        distractor_len_after = (
            context_len
            - len(prefix_toks)
            - len(passkey_toks)
            - distractor_len_before
            - len(question_toks)
            - 5  # space for answer
        )
        distractor_len_after = max(0, distractor_len_after)

        def _distractors(n: int) -> list[int]:
            out = []
            src = _DISTRACTOR
            while len(out) < n:
                out.extend(ord(c) for c in src)
            return out[:n]

        tokens = (
            prefix_toks
            + _distractors(distractor_len_before)
            + passkey_toks
            + _distractors(distractor_len_after)
            + question_toks
        )

        # Clamp all token values to valid ASCII range
        tokens = [min(t, 127) for t in tokens]
        return tokens, passkey

    def _greedy_generate(self, prompt_ids: torch.Tensor, n_tokens: int) -> list[int]:
        """Autoregressively generate n_tokens after prompt using segment-wise forward."""
        self.model.eval()
        B = prompt_ids.shape[0]
        device = self.device

        # Process prompt in segments
        states = self.model.init_states(B, device)
        T = prompt_ids.shape[1]

        with torch.no_grad():
            for start in range(0, T, self.segment_len):
                seg = prompt_ids[:, start : start + self.segment_len]
                if seg.shape[1] == 0:  # pragma: no cover
                    break
                logits, states = self.model(seg, states)

            # states now contains memory after full prompt

            # Generate tokens one by one
            generated: list[int] = []
            # Last token of prompt serves as first input
            last_tok = prompt_ids[:, -1:]  # (1, 1)

            for _ in range(n_tokens):
                logits, states = self.model(last_tok, states)
                next_id = logits[0, -1].argmax().item()
                generated.append(int(next_id))
                last_tok = torch.tensor([[next_id]], device=device)

        return generated

    def run(self) -> dict[int, float]:
        """
        Run the benchmark.

        Returns a dict mapping context_length → accuracy (0.0 – 1.0).
        """
        results: dict[int, float] = {}

        for ctx_len in self.context_lengths:
            correct = 0
            for trial in range(self.n_trials):
                token_ids, passkey = self._make_prompt(ctx_len, seed=trial)
                prompt = torch.tensor([token_ids], dtype=torch.long, device=self.device)

                # Generate 5 tokens (the passkey answer)
                generated = self._greedy_generate(prompt, n_tokens=5)
                gen_str = "".join(chr(t) for t in generated if 32 <= t < 127)

                if passkey in gen_str:
                    correct += 1

            results[ctx_len] = correct / self.n_trials

        return results
