"""
drex.eval.babilong — Synthetic BABILong-style long-context Q&A benchmark.

Implements 5 task families of increasing reasoning complexity (in the spirit of
the bAbI corpus) without requiring a downloaded dataset. All tasks use the same
character-level tokenisation as PasskeyBenchmark.

Task 1: Single supporting fact   — entity's current location
Task 2: Double supporting fact   — entity's location after one move
Task 3: Triple supporting fact   — entity's location after two moves
Task 4: Object possession        — entity received an object; who has it?
Task 5: Count after drop         — N pick-ups, 1 drop; how many remain?
"""

from __future__ import annotations

import random
from typing import Optional

import torch

from drex.models.transformer import DrexTransformer


_DISTRACTOR = "The quick brown fox jumps over the lazy dog. "
_PREFIX = "Pay attention to the following story. "
_QUESTION_PREFIX = " Question: "

_ENTITIES = ["Mary", "John", "Sandra", "Daniel"]
_LOCATIONS = ["garden", "office", "kitchen", "hallway"]
_OBJECTS = ["milk", "book", "football", "apple"]


class BABILongBenchmark:
    """
    Character-level BABILong-style benchmark.

    Embeds factual statements at a controlled depth in a distractor context,
    then asks the model a short question and evaluates exact-match accuracy.
    """

    def __init__(
        self,
        model: DrexTransformer,
        context_lengths: list[int],
        tasks: tuple[int, ...] = (1, 2, 3, 4, 5),
        n_trials: int = 10,
        device: Optional[torch.device] = None,
        segment_len: int = 512,
    ) -> None:
        self.model = model
        self.context_lengths = context_lengths
        self.tasks = tasks
        self.n_trials = n_trials
        self.device = device or torch.device("cpu")
        self.segment_len = segment_len

    # ------------------------------------------------------------------
    # Shared context builder
    # ------------------------------------------------------------------

    def _embed_in_context(
        self,
        context_len: int,
        facts: str,
        question: str,
        answer: str,
    ) -> tuple[list[int], str]:
        """
        Embed `facts` at ~50% depth in a distractor context of ~context_len chars.
        Returns (token_ids: list[int], answer: str).
        """
        prefix_toks = [ord(c) for c in _PREFIX]
        fact_toks = [ord(c) for c in facts]
        q_toks = [ord(c) for c in (_QUESTION_PREFIX + question)]

        fact_pos = (context_len // 2) - len(fact_toks)
        if fact_pos < len(prefix_toks):
            fact_pos = len(prefix_toks)

        dist_before = fact_pos - len(prefix_toks)
        dist_after = max(
            0,
            context_len
            - len(prefix_toks)
            - len(fact_toks)
            - dist_before
            - len(q_toks)
            - len(answer),
        )

        def _distractors(n: int) -> list[int]:
            out: list[int] = []
            while len(out) < n:
                out.extend(ord(c) for c in _DISTRACTOR)
            return out[:n]

        tokens = (
            prefix_toks
            + _distractors(dist_before)
            + fact_toks
            + _distractors(dist_after)
            + q_toks
        )
        return [min(t, 127) for t in tokens], answer

    # ------------------------------------------------------------------
    # Task generators
    # ------------------------------------------------------------------

    def _make_task1(self, context_len: int, seed: int) -> tuple[list[int], str]:
        """Task 1: Single supporting fact — current location."""
        rng = random.Random(seed)
        entity = rng.choice(_ENTITIES)
        location = rng.choice(_LOCATIONS)
        facts = f"{entity} travelled to the {location}. "
        question = f"Where is {entity}?"
        return self._embed_in_context(context_len, facts, question, location)

    def _make_task2(self, context_len: int, seed: int) -> tuple[list[int], str]:
        """Task 2: Two supporting facts — location after one move."""
        rng = random.Random(seed)
        entity = rng.choice(_ENTITIES)
        locs = rng.sample(_LOCATIONS, 2)
        facts = (
            f"{entity} travelled to the {locs[0]}. "
            f"{entity} moved to the {locs[1]}. "
        )
        question = f"Where is {entity}?"
        return self._embed_in_context(context_len, facts, question, locs[1])

    def _make_task3(self, context_len: int, seed: int) -> tuple[list[int], str]:
        """Task 3: Three supporting facts — location after two moves."""
        rng = random.Random(seed)
        entity = rng.choice(_ENTITIES)
        locs = rng.sample(_LOCATIONS, 3)
        facts = (
            f"{entity} went to the {locs[0]}. "
            f"{entity} went to the {locs[1]}. "
            f"{entity} went to the {locs[2]}. "
        )
        question = f"Where is {entity}?"
        return self._embed_in_context(context_len, facts, question, locs[2])

    def _make_task4(self, context_len: int, seed: int) -> tuple[list[int], str]:
        """Task 4: Object possession — entity received an object."""
        rng = random.Random(seed)
        giver, receiver = rng.sample(_ENTITIES, 2)
        obj = rng.choice(_OBJECTS)
        facts = f"{giver} gave the {obj} to {receiver}. "
        question = f"Who has the {obj}?"
        return self._embed_in_context(context_len, facts, question, receiver)

    def _make_task5(self, context_len: int, seed: int) -> tuple[list[int], str]:
        """Task 5: Counting — how many objects remain after one drop."""
        rng = random.Random(seed)
        entity = rng.choice(_ENTITIES)
        picked = rng.sample(_OBJECTS, 3)
        dropped = picked[0]
        facts = (
            f"{entity} picked up the {picked[0]}. "
            f"{entity} picked up the {picked[1]}. "
            f"{entity} picked up the {picked[2]}. "
            f"{entity} dropped the {dropped}. "
        )
        question = f"How many objects does {entity} have?"
        answer = "2"
        return self._embed_in_context(context_len, facts, question, answer)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _greedy_generate(self, prompt_ids: torch.Tensor, n_tokens: int) -> list[int]:
        """Autoregressively generate n_tokens after the prompt."""
        self.model.eval()
        B = prompt_ids.shape[0]
        states = self.model.init_states(B, self.device)
        T = prompt_ids.shape[1]

        with torch.no_grad():
            for start in range(0, T, self.segment_len):
                seg = prompt_ids[:, start : start + self.segment_len]
                if seg.shape[1] == 0:  # pragma: no cover
                    break
                logits, states = self.model(seg, states)

            generated: list[int] = []
            last_tok = prompt_ids[:, -1:]
            for _ in range(n_tokens):
                logits, states = self.model(last_tok, states)
                next_id = int(logits[0, -1].argmax().item())
                generated.append(next_id)
                last_tok = torch.tensor([[next_id]], device=self.device)

        return generated

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> dict[int, dict[int, float]]:
        """
        Evaluate all tasks over all context lengths.

        Returns:
            ``{task_id: {context_length: accuracy}}``
        """
        _task_fn = {
            1: self._make_task1,
            2: self._make_task2,
            3: self._make_task3,
            4: self._make_task4,
            5: self._make_task5,
        }
        results: dict[int, dict[int, float]] = {}

        for task_id in self.tasks:
            make_fn = _task_fn[task_id]
            task_results: dict[int, float] = {}

            for ctx_len in self.context_lengths:
                correct = 0
                for trial in range(self.n_trials):
                    token_ids, answer = make_fn(ctx_len, seed=trial)
                    prompt = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                    n_gen = len(answer) + 2  # a few extra chars for safety
                    generated = self._greedy_generate(prompt, n_tokens=n_gen)
                    gen_str = "".join(chr(t) for t in generated if 32 <= t < 127)
                    if answer in gen_str:
                        correct += 1

                task_results[ctx_len] = correct / self.n_trials

            results[task_id] = task_results

        return results
