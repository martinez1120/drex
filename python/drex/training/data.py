"""
drex.training.data — segment-level dataset utilities for training DrexTransformer.

Token sequences longer than a GPT context window are sliced into consecutive
overlapping segments. DrexTrainer processes each segment sequentially and
threads the recurrent state across boundaries.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset


def tokenize_chars(text: str, vocab_size: int = 256) -> list[int]:
    """
    Character-level tokeniser: return ASCII ordinals clamped to [0, vocab_size).

    Suitable for small-scale experiments and benchmarks that don't need a
    trained sub-word vocabulary.
    """
    return [min(ord(c), vocab_size - 1) for c in text]


class SegmentDataset(Dataset):
    """
    Fixed-length sliding-window dataset over a pre-tokenised integer sequence.

    Each item is a ``(segment_len + 1,)`` int64 tensor; the first ``segment_len``
    tokens are the model input, the last ``segment_len`` tokens (shifted by 1)
    are the language-model target.  Use ``collate_fn`` to split them in the
    DataLoader.

    Args:
        tokens:       flat list of token ids (already tokenised)
        segment_len:  number of input tokens per segment (context length)
        stride:       step between consecutive segments; defaults to segment_len
                      (non-overlapping).  Set stride < segment_len for overlap.
    """

    def __init__(
        self,
        tokens: list[int],
        segment_len: int,
        stride: Optional[int] = None,
    ) -> None:
        self.tokens = tokens
        self.segment_len = segment_len
        self.stride = stride if stride is not None else segment_len

        # How many complete (input + target) windows fit?
        n = len(tokens)
        self._len = max(0, (n - segment_len - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._len:
            raise IndexError(f"index {idx} out of range for dataset of length {self._len}")
        start = idx * self.stride
        chunk = self.tokens[start : start + self.segment_len + 1]
        return torch.tensor(chunk, dtype=torch.long)


def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a list of ``(T+1,)`` tensors into ``(B, T)`` input and target pairs.

    Designed to be passed as ``collate_fn`` to ``torch.utils.data.DataLoader``.
    """
    stacked = torch.stack(batch)       # (B, T+1)
    return stacked[:, :-1], stacked[:, 1:]
