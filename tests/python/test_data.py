"""
Tests for drex.training.data — tokenize_chars, SegmentDataset, collate_fn.
Also imports drex.training package to cover training/__init__.py.
"""

import pytest
import torch

from drex.training import (  # covers training/__init__.py new exports
    SegmentDataset,
    build_optimizer,
    collate_fn,
    cosine_schedule_with_warmup,
    tokenize_chars,
)


class TestTokenizeChars:
    def test_basic_ascii(self):
        tokens = tokenize_chars("abc", vocab_size=256)
        assert tokens == [97, 98, 99]

    def test_clamp_above_vocab(self):
        """Characters with ordinal >= vocab_size are clamped."""
        tokens = tokenize_chars("\xff", vocab_size=128)  # ord=255, clamped to 127
        assert tokens == [127]

    def test_empty_string(self):
        assert tokenize_chars("", vocab_size=256) == []

    def test_all_printable(self):
        text = "Hello, World!"
        tokens = tokenize_chars(text)
        assert all(0 <= t < 256 for t in tokens)
        assert len(tokens) == len(text)


class TestSegmentDataset:
    def test_len_non_overlapping(self):
        tokens = list(range(100))
        ds = SegmentDataset(tokens, segment_len=10)  # stride defaults to 10
        # (100 - 10 - 1) // 10 + 1 = 89 // 10 + 1 = 8 + 1 = 9
        assert len(ds) == 9

    def test_len_with_stride(self):
        tokens = list(range(100))
        ds = SegmentDataset(tokens, segment_len=10, stride=5)
        # (100 - 10 - 1) // 5 + 1 = 89 // 5 + 1 = 17 + 1 = 18
        assert len(ds) == 18

    def test_len_too_short_is_zero(self):
        """Sequence shorter than segment_len + 1 produces an empty dataset."""
        ds = SegmentDataset(list(range(5)), segment_len=10)
        assert len(ds) == 0

    def test_getitem_returns_correct_slice(self):
        tokens = list(range(50))
        ds = SegmentDataset(tokens, segment_len=10)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.long
        assert item.shape == (11,)  # segment_len + 1
        assert item.tolist() == list(range(11))

    def test_getitem_stride(self):
        tokens = list(range(50))
        ds = SegmentDataset(tokens, segment_len=10, stride=5)
        item1 = ds[0]
        item2 = ds[1]
        assert item1.tolist() == list(range(11))
        assert item2.tolist() == list(range(5, 16))

    def test_getitem_negative_index_raises(self):
        ds = SegmentDataset(list(range(50)), segment_len=10)
        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_getitem_out_of_range_raises(self):
        ds = SegmentDataset(list(range(50)), segment_len=10)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]

    def test_stride_none_defaults_to_segment_len(self):
        ds = SegmentDataset(list(range(50)), segment_len=10, stride=None)
        assert ds.stride == 10

    def test_stride_explicit(self):
        ds = SegmentDataset(list(range(50)), segment_len=10, stride=3)
        assert ds.stride == 3


class TestCollateFn:
    def test_output_shapes(self):
        batch = [torch.arange(11)] * 4  # 4 items of length 11
        src, tgt = collate_fn(batch)
        assert src.shape == (4, 10)
        assert tgt.shape == (4, 10)

    def test_shifted_by_one(self):
        tokens = torch.tensor([0, 1, 2, 3, 4])
        src, tgt = collate_fn([tokens])
        assert src.tolist() == [[0, 1, 2, 3]]
        assert tgt.tolist() == [[1, 2, 3, 4]]

    def test_batch_size_one(self):
        item = torch.arange(6)
        src, tgt = collate_fn([item])
        assert src.shape == (1, 5)
        assert tgt.shape == (1, 5)
