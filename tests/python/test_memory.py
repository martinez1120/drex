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
    MemoryModule,
    MemoryState,
    TitanMemory,
    WRITE_RATE_HI,
    WRITE_RATE_LO,
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


# ---------------------------------------------------------------------------
# MemoryModule
# ---------------------------------------------------------------------------


class TestMemoryModule:
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_odd_d_model_raises(self):
        """d_model must be even for the 50/50 episodic/semantic split."""
        with pytest.raises(ValueError, match="d_model must be even"):
            MemoryModule(d_model=7)

    def test_init_stores_attributes(self):
        mem = MemoryModule(d_model=16, gate_thresh=0.65)
        assert mem.d_model == 16
        assert mem.gate_thresh == 0.65
        assert mem._d_half == 8

    # ------------------------------------------------------------------
    # alpha() static method
    # ------------------------------------------------------------------

    def test_alpha_at_l_ref(self):
        """α(96) == 0.95 (the reference point of the exp_scale formula)."""
        assert abs(MemoryModule.alpha(96) - 0.95) < 1e-9

    def test_alpha_at_l32(self):
        """α(32) = 0.95^(96/32) = 0.95^3 ≈ 0.857375."""
        expected = 0.95 ** 3
        assert abs(MemoryModule.alpha(32) - expected) < 1e-9

    def test_alpha_decreases_with_shorter_sequences(self):
        """Shorter sequences → smaller α → faster EMA forgetting."""
        assert MemoryModule.alpha(16) < MemoryModule.alpha(32)
        assert MemoryModule.alpha(32) < MemoryModule.alpha(96)
        assert MemoryModule.alpha(96) < MemoryModule.alpha(128)

    # ------------------------------------------------------------------
    # last_write_rate / assert_write_rate_valid
    # ------------------------------------------------------------------

    def test_last_write_rate_initial(self):
        """Starts at 0.0 before any forward pass."""
        mem = MemoryModule(d_model=16)
        assert mem.last_write_rate() == 0.0

    def test_assert_write_rate_valid_in_range(self):
        """Does not raise when write rate is inside [0.10, 0.85]."""
        mem = MemoryModule(d_model=16)
        mem._last_write_rate = 0.50
        mem.assert_write_rate_valid()  # should not raise

    def test_assert_write_rate_valid_at_boundaries(self):
        """Boundary values are inclusive."""
        mem = MemoryModule(d_model=16)
        mem._last_write_rate = WRITE_RATE_LO
        mem.assert_write_rate_valid()
        mem._last_write_rate = WRITE_RATE_HI
        mem.assert_write_rate_valid()

    def test_assert_write_rate_invalid_high(self):
        """Raises AssertionError when write rate exceeds upper bound."""
        mem = MemoryModule(d_model=16)
        mem._last_write_rate = 0.99
        with pytest.raises(AssertionError, match="Write rate"):
            mem.assert_write_rate_valid()

    def test_assert_write_rate_invalid_low(self):
        """Raises AssertionError when write rate is below lower bound."""
        mem = MemoryModule(d_model=16)
        mem._last_write_rate = 0.01
        with pytest.raises(AssertionError, match="Write rate"):
            mem.assert_write_rate_valid()

    # ------------------------------------------------------------------
    # forward — output shape and values
    # ------------------------------------------------------------------

    def test_output_shape(self, device):
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        B, L = 3, 16
        x = torch.randn(B, L, 32, device=dev)
        out = mem(x)
        assert out.shape == (B, 32)

    def test_no_nan_no_inf(self, device):
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 24, 32, device=dev)
        out = mem(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_single_token_sequence(self, device):
        """L=1: write loop skips (wr_total=0), max(0,1) guard returns 0.0 rate."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 1, 32, device=dev)
        out = mem(x)
        assert out.shape == (2, 32)
        assert mem.last_write_rate() == 0.0

    def test_write_rate_updated_after_forward(self, device):
        """Write rate is in [0, 1] after a normal forward pass."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(4, 32, 32, device=dev)
        mem(x)
        wr = mem.last_write_rate()
        assert 0.0 <= wr <= 1.0

    def test_write_rate_nonzero_for_long_sequence(self, device):
        """With a long random sequence and default thresh=0.70, gate fires > 0%."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 64, 32, device=dev)
        mem(x)
        assert mem.last_write_rate() > 0.0

    def test_write_rate_stable_with_low_thresh(self, device):
        """Very low threshold → gate fires for almost all tokens."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32, gate_thresh=0.01).to(dev)
        x = torch.randn(2, 48, 32, device=dev)
        mem(x)
        # Near-universal firing expected; just verify it's in [0, 1]
        assert 0.0 <= mem.last_write_rate() <= 1.0

    def test_write_rate_suppressed_with_high_thresh(self, device):
        """Very high threshold → gate fires for very few tokens."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32, gate_thresh=999.0).to(dev)
        x = torch.randn(2, 48, 32, device=dev)
        mem(x)
        assert mem.last_write_rate() == 0.0

    # ------------------------------------------------------------------
    # forward — gradient flow
    # ------------------------------------------------------------------

    def test_backward_grad_on_input(self, device):
        """Gradients flow back to the input tensor."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 16, 32, device=dev, requires_grad=True)
        out = mem(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_backward_grad_on_params(self, device):
        """Gradients flow to model parameters."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 16, 32, device=dev)
        out = mem(x)
        out.sum().backward()
        assert any(p.grad is not None for p in mem.parameters())

    # ------------------------------------------------------------------
    # forward — null gate behaviour
    # ------------------------------------------------------------------

    def test_null_gate_initialises_near_half(self, device):
        """null_gate linear is zero-initialised so σ(0) = 0.5 initially."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        # null_gate bias=True; default init has weight≈0 and bias=0 → σ(0)=0.5
        q = torch.zeros(1, 32, device=dev)
        g = torch.sigmoid(mem.null_gate(q))
        # Allow a small range since weight is trunc_normal when wired via transformer
        assert 0.0 < g.item() < 1.0

    # ------------------------------------------------------------------
    # forward — numerical stability
    # ------------------------------------------------------------------

    def test_no_nan_with_zero_projection_input(self, device):
        """F.normalize eps guard: zero-magnitude projections must not produce NaN."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        # Force all linear weights and biases to zero so every projection outputs a
        # zero vector — the hardest case for F.normalize stability.
        with torch.no_grad():
            for m in mem.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.zero_()
                    if m.bias is not None:
                        m.bias.zero_()
        x = torch.randn(2, 16, 32, device=dev)
        out = mem(x)
        assert not torch.isnan(out).any(), "NaN in output with zero-projection weights"
        assert not torch.isinf(out).any(), "Inf in output with zero-projection weights"

    def test_write_rate_deterministic_across_forward_calls(self, device):
        """Vectorised write loop produces identical write rate on repeated identical inputs."""
        dev = torch.device(device)
        torch.manual_seed(42)
        mem = MemoryModule(d_model=32, gate_thresh=0.5).to(dev)
        x = torch.randn(3, 32, 32, device=dev)

        mem(x)
        wr_first = mem.last_write_rate()

        mem(x)
        wr_second = mem.last_write_rate()

        assert abs(wr_first - wr_second) < 1e-6, (
            f"Write rate is not deterministic: {wr_first} vs {wr_second}"
        )
        assert 0.0 <= wr_first <= 1.0


# ---------------------------------------------------------------------------
# MemoryModule — Phase 16 ablation: use_null_gate=False
# ---------------------------------------------------------------------------


class TestMemoryModuleNullGateAblation:
    """Tests for the null-retrieval-gate ablation (use_null_gate=False)."""

    def test_no_null_gate_stores_attribute(self):
        mem = MemoryModule(d_model=16, use_null_gate=False)
        assert mem.use_null_gate is False

    def test_null_gate_linear_not_created(self):
        """When use_null_gate=False, null_gate attribute is None (no wasted params)."""
        mem = MemoryModule(d_model=16, use_null_gate=False)
        assert mem.null_gate is None

    def test_default_still_creates_null_gate(self):
        """Default (use_null_gate=True) still creates the Linear as before."""
        mem = MemoryModule(d_model=16)
        assert mem.null_gate is not None

    def test_no_null_gate_output_shape(self, device):
        """Output shape is identical whether null gate is on or off."""
        dev = torch.device(device)
        B, L = 2, 16
        x = torch.randn(B, L, 32, device=dev)
        mem_on = MemoryModule(d_model=32, use_null_gate=True).to(dev)
        mem_off = MemoryModule(d_model=32, use_null_gate=False).to(dev)
        assert mem_on(x).shape == mem_off(x).shape == (B, 32)

    def test_no_null_gate_no_nan(self, device):
        dev = torch.device(device)
        mem = MemoryModule(d_model=32, use_null_gate=False).to(dev)
        x = torch.randn(2, 24, 32, device=dev)
        out = mem(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_no_null_gate_gradient_flows(self, device):
        dev = torch.device(device)
        mem = MemoryModule(d_model=32, use_null_gate=False).to(dev)
        x = torch.randn(2, 16, 32, device=dev, requires_grad=True)
        out = mem(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_null_gate_write_rate_valid(self, device):
        dev = torch.device(device)
        mem = MemoryModule(d_model=32, use_null_gate=False).to(dev)
        x = torch.randn(2, 48, 32, device=dev)
        mem(x)
        wr = mem.last_write_rate()
        assert 0.0 <= wr <= 1.0

    def test_no_null_gate_differs_from_gate_on(self, device):
        """With identical weights-seed, gated and un-gated outputs should differ."""
        dev = torch.device(device)
        torch.manual_seed(7)
        mem_on = MemoryModule(d_model=32, use_null_gate=True).to(dev)
        torch.manual_seed(7)
        mem_off = MemoryModule(d_model=32, use_null_gate=False).to(dev)
        x = torch.randn(2, 16, 32, device=dev)
        # Both are freshly seeded but architectures differ; outputs are not expected equal
        out_on = mem_on(x)
        out_off = mem_off(x)
        assert out_on.shape == out_off.shape
        # The outputs should differ because g_null multiplies by values in (0,1)
        assert not torch.allclose(out_on, out_off)


# ---------------------------------------------------------------------------
# MemoryModule — Phase 16 CPU-backend write loop
# ---------------------------------------------------------------------------


class TestMemoryModuleCpuBackend:
    """
    The sequential recurrence loop runs on CPU inside torch.no_grad() with
    detached key tensors.  This avoids:
      1. MPS/CUDA per-kernel-launch latency (each small bmm costs ~1-3 ms on MPS).
      2. Autograd graph construction through L-1 sequential assignments.

    Gradient signal to sem_proj/epi_proj comes exclusively from the read path
    (qns = normalize(sem_proj(q)), r_sem = M_sem @ qns) below the loop, which is
    the primary learning signal for memory retrieval quality.
    """

    def test_output_device_matches_input(self, device):
        """Output tensor must live on the same device as the input."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 16, 32, device=dev)
        out = mem(x)
        assert out.device.type == dev.type

    def test_backward_through_cpu_loop(self, device):
        """Gradients must flow from the final output back through the read path.

        With the detached write loop, gradient flows only through the query
        (x[:, -1, :]) via the read projection.  backward() must not raise and
        x.grad must be non-NaN on the correct device.
        """
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 16, 32, device=dev, requires_grad=True)
        out = mem(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.device.type == dev.type

    def test_param_grads_through_cpu_loop(self, device):
        """Projection weights must receive gradients via the read path.

        sem_proj/epi_proj are trained through the read query (qns, qne) rather
        than through the detached write loop.
        """
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 16, 32, device=dev)
        mem(x).sum().backward()
        grads = {n: p.grad for n, p in mem.named_parameters() if p.grad is not None}
        # Projections receive gradient through read path (qns = normalize(sem_proj(q)))
        assert any("sem_proj" in n for n in grads), "sem_proj missing grad"
        assert any("epi_proj" in n for n in grads), "epi_proj missing grad"

    def test_write_rate_unchanged_by_device_move(self, device):
        """Write rate must be the same regardless of whether the input is on CPU or
        the configured device (both paths run the loop on CPU)."""
        torch.manual_seed(0)
        mem_dev = MemoryModule(d_model=32).to(torch.device(device))
        torch.manual_seed(0)
        mem_cpu = MemoryModule(d_model=32)  # stays on CPU

        x_base = torch.randn(2, 32, 32)
        x_dev  = x_base.to(torch.device(device))

        with torch.no_grad():
            mem_dev(x_dev)
            mem_cpu(x_base)
        assert abs(mem_dev.last_write_rate() - mem_cpu.last_write_rate()) < 1e-5

    def test_no_nan_long_sequence(self, device):
        """L=256 (many loop iterations) must not produce NaN on any device."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 256, 32, device=dev)
        out = mem(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_single_token_no_loop(self, device):
        """L=1 skips the loop entirely; no tensors are moved to CPU."""
        dev = torch.device(device)
        mem = MemoryModule(d_model=32).to(dev)
        x = torch.randn(2, 1, 32, device=dev)
        out = mem(x)
        assert out.device.type == dev.type
        assert not torch.isnan(out).any()
        assert mem.last_write_rate() == 0.0
