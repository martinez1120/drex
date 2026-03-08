"""
Tests for the drex._sys Rust extension (Phase 2 smoke — runs now).
"""

import struct
import tempfile
import pytest

# Skip entire module if Rust extension not built
import drex
pytestmark = pytest.mark.skipif(
    not drex._RUST_AVAILABLE,
    reason="drex._sys Rust extension not built — run `maturin develop` first",
)

import drex._sys as sys


def f32(v: float) -> float:
    """Round-trip through f32 precision."""
    return struct.unpack("f", struct.pack("f", v))[0]


class TestSnapshotStore:
    def test_write_read_roundtrip(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        weights = [float(i) for i in range(32)]
        store.write(0, 0, 1, weights)
        recovered = store.read(0, 0, 1)
        assert len(recovered) == 32
        for orig, got in zip(weights, recovered):
            assert abs(got - f32(orig)) < 1e-9

    def test_write_read_zstd(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path), compress=True)
        weights = [1.0, 2.0, 3.0, 4.0]
        store.write(1, 2, 10, weights)
        recovered = store.read(1, 2, 10)
        assert recovered == weights

    def test_exists(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        assert not store.exists(0, 0, 99)
        store.write(0, 0, 99, [1.0, 2.0])
        assert store.exists(0, 0, 99)

    def test_delete(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        store.write(0, 0, 5, [1.0])
        assert store.exists(0, 0, 5)
        store.delete(0, 0, 5)
        assert not store.exists(0, 0, 5)

    def test_delete_nonexistent_is_ok(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        store.delete(0, 0, 999)  # should not raise

    def test_read_nonexistent_raises(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        with pytest.raises(RuntimeError, match="not found"):
            store.read(0, 0, 999)

    def test_list_snapshots(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        store.write(0, 0, 1, [1.0])
        store.write(0, 1, 2, [2.0])
        store.write(1, 0, 3, [3.0])
        metas = store.list_snapshots()
        assert len(metas) == 3
        steps = {m.step for m in metas}
        assert steps == {1, 2, 3}

    def test_list_by_layer(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        store.write(0, 0, 1, [1.0])
        store.write(0, 1, 2, [2.0])
        store.write(1, 0, 3, [3.0])
        layer0 = store.list_by_layer(0)
        assert len(layer0) == 2
        layer1 = store.list_by_layer(1)
        assert len(layer1) == 1

    def test_meta_fields(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        store.write(2, 3, 42, [1.0, 2.0, 3.0])
        metas = store.list_by_layer(2)
        assert len(metas) == 1
        m = metas[0]
        assert m.layer == 2
        assert m.head == 3
        assert m.step == 42
        assert m.n_weights == 3
        assert m.size_bytes > 0
        assert not m.compressed

    def test_large_weights(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        weights = [float(i) * 0.001 for i in range(10_000)]
        store.write(0, 0, 1, weights)
        recovered = store.read(0, 0, 1)
        assert len(recovered) == 10_000
        for o, r in zip(weights, recovered):
            assert abs(r - f32(o)) < 1e-9

    def test_read_mmap_roundtrip(self, tmp_path):
        """read_mmap on an uncompressed store returns identical weights to write."""
        store = sys.SnapshotStore(str(tmp_path))
        weights = [float(i) * 0.25 for i in range(64)]
        store.write(0, 0, 1, weights)
        recovered = store.read_mmap(0, 0, 1)
        assert recovered == weights

    def test_read_mmap_compressed_fallback(self, tmp_path):
        """read_mmap on a compressed store falls back to regular read."""
        store = sys.SnapshotStore(str(tmp_path), compress=True)
        weights = [1.0, 2.0, 3.0]
        store.write(0, 0, 1, weights)
        recovered = store.read_mmap(0, 0, 1)
        assert recovered == weights

    def test_read_mmap_nonexistent_raises(self, tmp_path):
        """read_mmap raises RuntimeError for a missing snapshot."""
        store = sys.SnapshotStore(str(tmp_path))
        with pytest.raises(RuntimeError, match="not found"):
            store.read_mmap(0, 0, 999)


class TestMemoryTierManager:
    def test_demote_promote(self, tmp_path):
        mgr = sys.MemoryTierManager(str(tmp_path))
        weights = [1.0, 2.0, 3.0]
        mgr.demote(0, 0, 1, weights)
        result = mgr.promote(0, 0, 1)
        assert result == weights

    def test_snapshot_count(self, tmp_path):
        mgr = sys.MemoryTierManager(str(tmp_path))
        assert mgr.snapshot_count() == 0
        mgr.demote(0, 0, 1, [1.0])
        assert mgr.snapshot_count() == 1
        mgr.demote(0, 1, 2, [2.0])
        assert mgr.snapshot_count() == 2

    def test_record_attention_score(self, tmp_path):
        mgr = sys.MemoryTierManager(str(tmp_path))
        # Should not raise; eviction tracking is internal
        mgr.record_attention_score(0, 0, 1, 0.75)
        mgr.record_attention_score(0, 0, 2, 0.25)

    def test_eviction_under_budget(self, tmp_path):
        mgr = sys.MemoryTierManager(str(tmp_path), l3_max_snapshots=10)
        mgr.demote(0, 0, 1, [1.0])
        evicted = mgr.run_eviction()
        assert evicted == []  # under budget, nothing evicted

    def test_eviction_over_budget(self, tmp_path):
        # Set a tiny budget so eviction fires
        mgr = sys.MemoryTierManager(
            str(tmp_path),
            l3_max_snapshots=2,
            eviction_batch_size=1,
        )
        mgr.demote(0, 0, 1, [1.0])
        mgr.demote(0, 1, 2, [2.0])
        mgr.demote(1, 0, 3, [3.0])
        mgr.record_attention_score(0, 0, 3, 0.1)  # low score = first to evict

        evicted = mgr.run_eviction()
        assert len(evicted) >= 1  # at least one evicted


class TestPrefetchEngine:
    def test_register_and_prefetch_and_consume(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        store.write(0, 0, 1, [0.5] * 16)

        engine = sys.PrefetchEngine(store, d_model=16, max_cache_entries=32, sketch_rank=4)
        engine.register_snapshot(0, 0, 1, [1.0] * 16)
        engine.prefetch(0, [1.0] * 16, k=1)

        import time
        time.sleep(0.2)  # let the async tokio task complete

        result = engine.consume_prefetched(0, 0, 1)
        assert result is not None, "expected prefetch hit after 200ms"
        assert result == [0.5] * 16

    def test_consume_miss_returns_none(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        engine = sys.PrefetchEngine(store, d_model=16, max_cache_entries=8)
        result = engine.consume_prefetched(0, 0, 999)
        assert result is None

    def test_clear(self, tmp_path):
        store = sys.SnapshotStore(str(tmp_path))
        store.write(0, 0, 1, [1.0])
        engine = sys.PrefetchEngine(store, d_model=4, max_cache_entries=8, sketch_rank=2)
        engine.register_snapshot(0, 0, 1, [1.0, 0.0, 0.0, 0.0])
        engine.prefetch(0, [1.0, 0.0, 0.0, 0.0], k=1)
        import time; time.sleep(0.1)
        engine.clear()
        result = engine.consume_prefetched(0, 0, 1)
        assert result is None  # cache was cleared
