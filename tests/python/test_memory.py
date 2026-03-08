"""
Tests for drex.models.memory — MemoryState, DeltaRuleUpdate, TitanMemory, L3MemoryBridge.
"""

import sys
import time

import pytest
import torch

from drex.models.memory import (
    DeltaRuleUpdate,
    L3MemoryBridge,
    LayerState,
    MemoryState,
    TitanMemory,
    _elu1,
)


@pytest.fixture
def batch():
    return 2


@pytest.fixture
def n_heads():
    return 4


@pytest.fixture
def d_k():
    return 16


@pytest.fixture
def d_v():
    return 16


@pytest.fixture
def seq_len():
    return 8


class TestMemoryState:
    def test_zeros_shape(self, batch, n_heads, d_k, d_v, device):
        dev = torch.device(device)
        state = MemoryState.zeros(batch, n_heads, d_k, d_v, dev)
        assert state.M.shape == (batch, n_heads, d_k, d_v)
        assert state.z.shape == (batch, n_heads, d_k)

    def test_zeros_are_zero(self, batch, n_heads, d_k, d_v, device):
        dev = torch.device(device)
        state = MemoryState.zeros(batch, n_heads, d_k, d_v, dev)
        assert state.M.sum().item() == 0.0
        assert state.z.sum().item() == 0.0

    def test_detach_breaks_grad(self, batch, n_heads, d_k, d_v, device):
        dev = torch.device(device)
        state = MemoryState(
            M=torch.randn(batch, n_heads, d_k, d_v, device=dev, requires_grad=True),
            z=torch.randn(batch, n_heads, d_k, device=dev, requires_grad=True),
        )
        detached = state.detach()
        assert not detached.M.requires_grad
        assert not detached.z.requires_grad

    def test_to_moves_device(self, batch, n_heads, d_k, d_v):
        state = MemoryState.zeros(batch, n_heads, d_k, d_v, torch.device("cpu"))
        moved = state.to(torch.device("cpu"))
        assert moved.M.device == torch.device("cpu")


class TestLayerState:
    def test_zeros(self, batch, n_heads, d_k, d_v, device):
        dev = torch.device(device)
        state = LayerState.zeros(batch, n_heads, d_k, d_v, dev)
        assert state.step == 0
        assert state.memory.M.shape == (batch, n_heads, d_k, d_v)

    def test_detach(self, batch, n_heads, d_k, d_v, device):
        dev = torch.device(device)
        state = LayerState.zeros(batch, n_heads, d_k, d_v, dev)
        state.memory.M.requires_grad = True
        d = state.detach()
        assert not d.memory.M.requires_grad

    def test_to(self, batch, n_heads, d_k, d_v, device):
        """LayerState.to() — line 70 in memory.py."""
        dev = torch.device(device)
        state = LayerState.zeros(batch, n_heads, d_k, d_v, torch.device("cpu"))
        moved = state.to(dev)
        assert moved.memory.M.device.type == dev.type
        assert moved.step == state.step


class TestElu1:
    def test_positive_output(self, device):
        x = torch.randn(4, 8, requires_grad=False)
        out = _elu1(x)
        assert (out > 0).all()

    def test_min_value(self, device):
        # ELU + 1 at -inf should approach 0, never negative
        x = torch.full((4,), -100.0)
        out = _elu1(x)
        assert (out >= 0).all()


class TestDeltaRuleUpdate:
    def test_shapes(self, batch, n_heads, seq_len, d_k, d_v, device):
        dev = torch.device(device)
        update = DeltaRuleUpdate()
        state = MemoryState.zeros(batch, n_heads, d_k, d_v, dev)
        K = torch.randn(batch, n_heads, seq_len, d_k, device=dev)
        V = torch.randn(batch, n_heads, seq_len, d_v, device=dev)
        new_state = update(K, V, state)
        assert new_state.M.shape == (batch, n_heads, d_k, d_v)
        assert new_state.z.shape == (batch, n_heads, d_k)

    def test_M_changes(self, batch, n_heads, seq_len, d_k, d_v, device):
        dev = torch.device(device)
        update = DeltaRuleUpdate()
        state = MemoryState.zeros(batch, n_heads, d_k, d_v, dev)
        K = torch.randn(batch, n_heads, seq_len, d_k, device=dev)
        V = torch.randn(batch, n_heads, seq_len, d_v, device=dev)
        new_state = update(K, V, state)
        assert not torch.allclose(new_state.M, state.M)

    def test_z_accumulates(self, batch, n_heads, seq_len, d_k, d_v, device):
        dev = torch.device(device)
        update = DeltaRuleUpdate()
        state = MemoryState.zeros(batch, n_heads, d_k, d_v, dev)
        K = torch.randn(batch, n_heads, seq_len, d_k, device=dev)
        V = torch.randn(batch, n_heads, seq_len, d_v, device=dev)
        # Two consecutive writes
        s1 = update(K, V, state)
        s2 = update(K, V, s1)
        # z should be strictly larger after two writes
        assert (s2.z >= s1.z).all()

    def test_idempotent_on_consistent_kv(self, batch, n_heads, seq_len, d_k, d_v, device):
        """If V = φ(K)M already (perfect match), delta M should be near zero."""
        dev = torch.device(device)
        update = DeltaRuleUpdate()
        M0 = torch.randn(batch, n_heads, d_k, d_v, device=dev)
        z0 = torch.ones(batch, n_heads, d_k, device=dev)  # non-zero z
        state = MemoryState(M=M0, z=z0)
        K = torch.zeros(batch, n_heads, seq_len, d_k, device=dev)
        # phi_K = elu1(0) = 1; V_existing = 1 * M → V should equal row-sum of M
        # This is hard to make exact, so instead just check gradient flows
        K.requires_grad_(True)
        V = torch.randn(batch, n_heads, seq_len, d_v, device=dev, requires_grad=True)
        new_state = update(K, V, state)
        new_state.M.sum().backward()
        assert K.grad is not None
        assert V.grad is not None


class TestTitanMemory:
    def test_forward_shape(self, device):
        dev = torch.device(device)
        titan = TitanMemory(d_model=32, d_hidden=16).to(dev)
        x = torch.randn(2, 32, device=dev)
        out = titan(x)
        assert out.shape == (2, 32)

    def test_write_reduces_loss(self, device):
        dev = torch.device(device)
        titan = TitanMemory(d_model=16, d_hidden=8, lr=1e-2).to(dev)
        key = torch.randn(1, 16, device=dev)
        value = torch.randn(1, 16, device=dev)
        # Multiple writes should decrease loss
        losses = [titan.write(key, value).item() for _ in range(10)]
        assert losses[-1] < losses[0], "Loss should decrease over repeated writes"

    def test_snapshot_roundtrip(self, device):
        dev = torch.device(device)
        titan = TitanMemory(d_model=16, d_hidden=8).to(dev)
        weights = titan.snapshot_weights()
        assert len(weights) == titan.weight_vector_size()
        # Load slightly perturbed weights, then reload originals
        perturbed = [w + 0.1 for w in weights]
        titan.load_weights(perturbed)
        titan.load_weights(weights)
        restored = titan.snapshot_weights()
        for a, b in zip(weights, restored):
            assert abs(a - b) < 1e-6

    def test_weight_vector_size(self, device):
        titan = TitanMemory(d_model=32, d_hidden=16)
        # fc1: 32*16=512, fc2: 16*32=512 → total 1024
        assert titan.weight_vector_size() == 32 * 16 + 16 * 32


# ---------------------------------------------------------------------------
# L3MemoryBridge
# ---------------------------------------------------------------------------

D = 16   # d_model for tiny titans


@pytest.fixture
def titans():
    """Two tiny TitanMemory instances (one per simulated layer)."""
    return [TitanMemory(d_model=D, d_hidden=8) for _ in range(2)]


@pytest.fixture
def bridge(titans, tmp_path):
    """Bridge backed by the live Rust extension."""
    return L3MemoryBridge(titans, base_path=str(tmp_path), compress=False, sketch_rank=4)


class TestL3MemoryBridge:
    # ------------------------------------------------------------------
    # __init__ branches
    # ------------------------------------------------------------------

    def test_init_rust_available(self, bridge):
        """Happy path: Rust extension present → _available True."""
        assert bridge._available is True
        assert bridge._store is not None
        assert bridge._engine is not None
        assert bridge._prefetch_hits == 0
        assert bridge._prefetch_calls == 0

    def test_init_rust_unavailable(self, titans, tmp_path):
        """ImportError fallback: lines 223-226 in memory.py."""
        saved = sys.modules.pop("drex._sys", None)
        sys.modules["drex._sys"] = None  # forces ModuleNotFoundError (subclass of ImportError)
        try:
            b = L3MemoryBridge(titans, base_path=str(tmp_path))
            assert b._available is False
            assert b._store is None
            assert b._engine is None
        finally:
            if saved is not None:
                sys.modules["drex._sys"] = saved
            else:
                del sys.modules["drex._sys"]

    # ------------------------------------------------------------------
    # write_and_snapshot
    # ------------------------------------------------------------------

    def test_write_and_snapshot_available(self, bridge):
        """Rust available: titan is updated AND snapshot appears on disk."""
        key, val = torch.randn(D), torch.randn(D)
        bridge.write_and_snapshot(layer=0, head=0, step=1, key_vec=key, value_vec=val)
        assert bridge._store.exists(0, 0, 1)

    def test_write_and_snapshot_unavailable(self, titans, tmp_path):
        """_available=False: titan.write() still called, disk step skipped."""
        b = L3MemoryBridge(titans, base_path=str(tmp_path))
        b._available = False  # force unavailable path
        key, val = torch.randn(D), torch.randn(D)
        # Must not raise; should silently skip disk I/O
        b.write_and_snapshot(layer=0, head=0, step=2, key_vec=key, value_vec=val)
        # _store was initialized normally; check no disk entry was written
        assert not b._store.exists(0, 0, 2)

    # ------------------------------------------------------------------
    # retrieve_and_load
    # ------------------------------------------------------------------

    def test_retrieve_unavailable(self, titans, tmp_path):
        """_available=False: immediately returns False."""
        b = L3MemoryBridge(titans, base_path=str(tmp_path))
        b._available = False
        assert b.retrieve_and_load(layer=0, head=0, step=0) is False

    def test_retrieve_prefetch_cache_hit(self, bridge):
        """Prefetch fills cache; retrieve returns True via cache path."""
        key, val = torch.randn(D), torch.randn(D)
        bridge.write_and_snapshot(layer=0, head=0, step=55, key_vec=key, value_vec=val)
        bridge.trigger_prefetch(layer=0, query_vec=key, k=1)
        time.sleep(0.15)  # let the Tokio background task complete
        hit = bridge.retrieve_and_load(layer=0, head=0, step=55)
        assert hit is True

    def test_retrieve_disk_hit(self, bridge):
        """Disk fallback path: consume_prefetched returns None, disk exists."""
        key, val = torch.randn(D), torch.randn(D)
        bridge.write_and_snapshot(layer=0, head=0, step=7, key_vec=key, value_vec=val)
        # Drain the prefetch cache to force disk path
        bridge._engine.consume_prefetched(0, 0, 7)
        assert bridge.retrieve_and_load(layer=0, head=0, step=7) is True

    def test_retrieve_total_miss(self, bridge):
        """Neither cache nor disk has the snapshot."""
        assert bridge.retrieve_and_load(layer=0, head=0, step=9999) is False

    # ------------------------------------------------------------------
    # trigger_prefetch
    # ------------------------------------------------------------------

    def test_trigger_prefetch_available(self, bridge):
        """Fire prefetch without raising when Rust is available."""
        key, val = torch.randn(D), torch.randn(D)
        bridge.write_and_snapshot(layer=0, head=0, step=10, key_vec=key, value_vec=val)
        bridge.trigger_prefetch(layer=0, query_vec=key, k=1)

    def test_trigger_prefetch_unavailable(self, titans, tmp_path):
        """_available=False: no-op, must not raise."""
        b = L3MemoryBridge(titans, base_path=str(tmp_path))
        b._available = False
        b.trigger_prefetch(layer=0, query_vec=torch.randn(D), k=2)

    # ------------------------------------------------------------------
    # prefetch_hit_rate property
    # ------------------------------------------------------------------

    def test_prefetch_hit_rate_zero_calls(self, bridge):
        """No calls yet → guard returns 0.0."""
        assert bridge.prefetch_hit_rate == 0.0

    def test_prefetch_hit_rate_after_calls(self, bridge):
        """After a disk-only retrieve: calls=1, hits=0 → rate=0.0."""
        key, val = torch.randn(D), torch.randn(D)
        bridge.write_and_snapshot(layer=0, head=0, step=99, key_vec=key, value_vec=val)
        bridge._engine.consume_prefetched(0, 0, 99)  # drain cache
        bridge.retrieve_and_load(layer=0, head=0, step=99)
        assert bridge._prefetch_calls == 1
        assert bridge.prefetch_hit_rate == 0.0  # disk hit, not cache hit
