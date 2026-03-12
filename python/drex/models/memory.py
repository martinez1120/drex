"""
drex.models.memory — associative memory components.

MemoryModule: Episodic/semantic split delta-rule memory with length-adaptive EMA and
  relative-norm write gate (validated Phases 11-12).

L2: Infini-Attention matrix memory (per-head M and z vectors).
L3: Titans-style MLP weight snapshots; L3MemoryBridge coordinates with the Rust layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# L2 state containers
# ---------------------------------------------------------------------------


@dataclass
class MemoryState:
    """
    Per-layer Infini-Attention matrix memory.

    Shapes (all float32):
        M  : (B, H, d_k, d_v)  — memory matrix
        z  : (B, H, d_k)       — normalisation accumulator
    """

    M: torch.Tensor  # (B, H, d_k, d_v)
    z: torch.Tensor  # (B, H, d_k)

    @staticmethod
    def zeros(batch: int, n_heads: int, d_k: int, d_v: int, device: torch.device) -> "MemoryState":
        return MemoryState(
            M=torch.zeros(batch, n_heads, d_k, d_v, device=device, dtype=torch.float32),
            z=torch.zeros(batch, n_heads, d_k, device=device, dtype=torch.float32),
        )

    def detach(self) -> "MemoryState":
        """Return a copy with gradients detached (for TBPTT segment boundaries)."""
        return MemoryState(M=self.M.detach(), z=self.z.detach())

    def to(self, device: torch.device) -> "MemoryState":
        return MemoryState(M=self.M.to(device), z=self.z.to(device))


@dataclass
class LayerState:
    """All recurrent state for one DrexLayer."""

    memory: MemoryState  # L2 Infini-Attention state
    step: int = 0        # global training step (used for L3 snapshot keys)

    @staticmethod
    def zeros(batch: int, n_heads: int, d_k: int, d_v: int, device: torch.device) -> "LayerState":
        return LayerState(
            memory=MemoryState.zeros(batch, n_heads, d_k, d_v, device),
            step=0,
        )

    def detach(self) -> "LayerState":
        return LayerState(memory=self.memory.detach(), step=self.step)

    def to(self, device: torch.device) -> "LayerState":
        return LayerState(memory=self.memory.to(device), step=self.step)


# ---------------------------------------------------------------------------
# L2 delta-rule write
# ---------------------------------------------------------------------------


def _elu1(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 feature map — positive-valued, used as φ in linear attention."""
    return F.elu(x) + 1.0


class DeltaRuleUpdate(nn.Module):
    """
    Stateless module implementing one Infini-Attention delta-rule write step.

    For each token position the update is:
        V_existing = φ(K) @ M          (B, H, S, d_v)
        M  +=  φ(K)ᵀ @ (V - V_existing)
        z  +=  sum(φ(K), dim=S)

    This is the "associative delta rule" from:
        Munkhdalai et al., "Leave No Context Behind", Google 2024.
    """

    def forward(
        self,
        K: torch.Tensor,      # (B, H, S, d_k)
        V: torch.Tensor,      # (B, H, S, d_v)
        state: MemoryState,
    ) -> MemoryState:
        phi_K = _elu1(K)      # (B, H, S, d_k)

        # V_existing = φ(K) M — what memory currently "says" for these keys
        # (B, H, S, d_k) @ (B, H, d_k, d_v) → (B, H, S, d_v)
        V_existing = torch.matmul(phi_K, state.M)

        # Associative delta update: ΔM = φ(K)ᵀ @ (V - V_existing)
        # (B, H, d_k, S) @ (B, H, S, d_v) → (B, H, d_k, d_v)
        delta_M = torch.matmul(phi_K.transpose(-2, -1), V - V_existing)

        new_M = state.M + delta_M
        new_z = state.z + phi_K.sum(dim=-2)  # (B, H, d_k)

        return MemoryState(M=new_M, z=new_z)


# ---------------------------------------------------------------------------
# L3 Titan MLP memory
# ---------------------------------------------------------------------------


class TitanMemory(nn.Module):
    """
    Titans-style memory MLP (Behrouz et al., Google Jan 2025).

    Memory is encoded as the weights of a small 2-layer MLP with no
    layer normalisation. Writing to memory = one gradient descent step
    on a surprise loss: ‖net(k) − v‖².

    The MLP has its own internal Adam optimiser so gradient steps are
    independent of the outer training optimiser.
    """

    def __init__(self, d_model: int, d_hidden: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Two-layer MLP, no bias, no normalisation
        self.fc1 = nn.Linear(d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)

        # Internal optimiser — updated only during write(), not outer backward
        self._optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read from memory: x → hidden → output."""
        return self.fc2(F.gelu(self.fc1(x)))

    def write(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        One gradient step on surprise loss: ‖net(key) − value‖².

        key, value: (B, d_model) — mean-pooled or representative token vecs.
        Returns the loss scalar (detached) for logging.
        """
        self._optim.zero_grad()
        pred = self(key)
        loss = F.mse_loss(pred, value.detach())
        loss.backward()
        self._optim.step()
        return loss.detach()

    def snapshot_weights(self) -> list[float]:
        """Return all parameters as a flat list of Python floats (f32)."""
        parts: list[torch.Tensor] = []
        for p in self.parameters():
            parts.append(p.detach().cpu().float().flatten())
        return torch.cat(parts).tolist()

    def load_weights(self, weights: list[float]) -> None:
        """Load weights from a flat list (must match weight_vector_size())."""
        buf = torch.tensor(weights, dtype=torch.float32)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(buf[offset : offset + n].view(p.shape))
                offset += n

    def weight_vector_size(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# L3 bridge — Python coordinator between TitanMemory and Rust _sys
# ---------------------------------------------------------------------------


class L3MemoryBridge:
    """
    Coordinates L3 (disk) memory I/O between Python TitanMemory and Rust _sys.

    Requires drex._RUST_AVAILABLE = True and a SnapshotStore + PrefetchEngine.
    When _RUST_AVAILABLE is False the bridge is a no-op, allowing the model to
    run with only L1+L2 memory.
    """

    def __init__(
        self,
        titan_memories: list[TitanMemory],  # one per layer
        base_path: str = "/tmp/drex_l3",
        compress: bool = False,
        max_prefetch_cache: int = 64,
        sketch_rank: int = 16,
    ) -> None:
        self._titans = titan_memories
        self._prefetch_hits = 0
        self._prefetch_calls = 0

        try:
            import drex._sys as _sys

            self._store = _sys.SnapshotStore(base_path, compress)
            self._engine = _sys.PrefetchEngine(
                self._store,
                d_model=titan_memories[0].d_model if titan_memories else 1,
                max_cache_entries=max_prefetch_cache,
                sketch_rank=sketch_rank,
            )
            self._available = True
        except ImportError:
            self._store = None
            self._engine = None
            self._available = False

    # ------------------------------------------------------------------

    def write_and_snapshot(
        self,
        layer: int,
        head: int,
        step: int,
        key_vec: torch.Tensor,  # (d_model,) representative key
        value_vec: torch.Tensor,  # (d_model,) representative value
    ) -> None:
        """Write key/value to TitanMemory then snapshot weights to Rust store."""
        titan = self._titans[layer]
        titan.write(key_vec.unsqueeze(0).float(), value_vec.unsqueeze(0).float())

        if not self._available:
            return

        weights = titan.snapshot_weights()
        self._store.write(layer, head, step, weights)
        # Register with prefetch engine for future similarity lookups
        self._engine.register_snapshot(layer, head, step, key_vec.float().tolist())

    def retrieve_and_load(self, layer: int, head: int, step: int) -> bool:
        """
        Try prefetch cache first, then disk. Loads weights into TitanMemory.
        Returns True on hit, False on miss.
        """
        if not self._available:
            return False

        self._prefetch_calls += 1

        # Check prefetch cache first
        weights = self._engine.consume_prefetched(layer, head, step)
        if weights is not None:
            self._prefetch_hits += 1
            self._titans[layer].load_weights(weights)
            return True

        # Fall back to synchronous disk read
        if self._store.exists(layer, head, step):
            weights = self._store.read(layer, head, step)
            self._titans[layer].load_weights(weights)
            return True

        return False

    def trigger_prefetch(self, layer: int, query_vec: torch.Tensor, k: int = 4) -> None:
        """Fire async prefetch using the sketch index. Non-blocking."""
        if not self._available:
            return
        self._engine.prefetch(layer, query_vec.float().tolist(), k)

    @property
    def prefetch_hit_rate(self) -> float:
        if self._prefetch_calls == 0:
            return 0.0
        return self._prefetch_hits / self._prefetch_calls


# ---------------------------------------------------------------------------
# Episodic/semantic associative MemoryModule (Phase 13 — validated Phases 11-12)
# ---------------------------------------------------------------------------

# EMA decay formula: α(L) = ALPHA_REF ^ (L_REF / L)
# Keeps τ/L ≈ 0.21 constant across L=32–128 (exp_47_3, Phase 11).
_ALPHA_REF: float = 0.95
_L_REF: int = 96

# Acceptable write-rate window during training (exp_45, Phase 9).
WRITE_RATE_LO: float = 0.10
WRITE_RATE_HI: float = 0.85


class MemoryModule(nn.Module):
    """
    Episodic/semantic split associative memory with length-adaptive EMA decay
    and a relative-vector-norm write gate.

    Validated architecture (Phases 11-12):
      - Two d_half×d_half matrices: M_sem (semantic) and M_epi (episodic,
        recency-weighted writes).
      - Delta-rule update: Δ = (k − vp) ⊗ k̂, EMA update M += (1−α)·Δ.
      - Length-adaptive decay: α(L) = 0.95^(96/L) — keeps τ/L ≈ 0.21.
      - Relative-norm write gate (OR over branches):
          fire iff ‖k_s − vp_s‖ ≥ thresh·‖k_s‖  OR
                   ‖k_e − vp_e‖ ≥ thresh·‖k_e‖
      - Production gate threshold: thresh=0.70 (exp_48_1, Phase 12).
      - Soft retrieval: r = concat(r_sem, r_epi) — no learned read gate
        (exp_38_3 ruled out learned read-gate combination).
      - Null retrieval gate: learned scalar g = σ(linear(q)) applied to r
        before the output projection, suppressing irrelevant retrievals.

    Forward contract:
        x : (B, L, d_model)  — full context with query token at position L-1.
        returns : (B, d_model)  — memory retrieval for the query at position L-1.

    Hard constraints (non-negotiable per research):
        - gate_thresh must not be set below 0.40 (exp_43_1).
        - Do not replace the fixed threshold with a randomly-initialised learnable
          parameter — this triggers the low-accuracy equilibrium (exp_43_1).
        - Use α(L) formula, never fixed α=0.95 alone (EMA bootstrap at L≤32).
        - Validate write rate ∈ [0.10, 0.85] after any write-mechanism change.
    """

    def __init__(
        self,
        d_model: int,
        gate_thresh: float = 0.70,
        use_null_gate: bool = True,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for episodic/semantic split, got {d_model}")
        self.d_model = d_model
        self.gate_thresh = gate_thresh
        self.use_null_gate = use_null_gate
        d_half = d_model // 2
        self._d_half = d_half

        # Separate key projections for each branch (no bias — scale-invariance)
        self.sem_proj = nn.Linear(d_model, d_half, bias=False)
        self.epi_proj = nn.Linear(d_model, d_half, bias=False)

        # Null retrieval gate: learned scalar σ(w·q) suppresses empty-memory reads.
        # Created only when use_null_gate=True so the ablation removes the parameters
        # entirely (avoids unused weights influencing parameter-count comparisons).
        self.null_gate: Optional[nn.Linear] = (
            nn.Linear(d_model, 1) if use_null_gate else None
        )

        # Output projection: concat(r_sem, r_epi) [d_model] → d_model
        self.out_proj = nn.Linear(d_model, d_model)

        # Write-rate from the most recent forward pass (float, updated in forward)
        self._last_write_rate: float = 0.0

    @staticmethod
    def alpha(L: int) -> float:
        """Length-adaptive EMA coefficient: α(L) = 0.95^(96/L)."""
        return _ALPHA_REF ** (_L_REF / L)

    def last_write_rate(self) -> float:
        """Write rate recorded during the most recent forward pass.

        Returns the fraction of (batch × step) positions where the OR gate
        fired.  Use to verify the rate stays in [WRITE_RATE_LO, WRITE_RATE_HI].
        """
        return self._last_write_rate

    def assert_write_rate_valid(self) -> None:
        """Raise AssertionError if the last write rate is outside [0.10, 0.85]."""
        wr = self._last_write_rate
        assert WRITE_RATE_LO <= wr <= WRITE_RATE_HI, (
            f"Write rate {wr:.3f} outside valid range "
            f"[{WRITE_RATE_LO}, {WRITE_RATE_HI}]. "
            "Check gate_thresh or sequence-length configuration."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the full context x and return the memory retrieval for the
        query token at position L-1.

        Args:
            x: (B, L, d_model) — token representations.  Position L-1 is
               the query; positions 0..L-2 are written into memory.

        Returns:
            (B, d_model) — memory retrieval after the null gate and output
            projection.
        """
        B, L, _ = x.shape
        a = self.alpha(L)
        d_half = self._d_half
        device = x.device

        # Zero-initialise associative matrices for this sequence
        M_sem = torch.zeros(B, d_half, d_half, device=device)
        M_epi = torch.zeros(B, d_half, d_half, device=device)

        if L > 1:
            # ── Vectorised pre-computation (GPU) ───────────────────────────
            # Batch all key projections and normalizations in two kernel launches
            # instead of 4*(L-1) individual calls.  Shapes: (B, L-1, d_half)
            ks_all  = self.sem_proj(x[:, :-1, :])
            ke_all  = self.epi_proj(x[:, :-1, :])
            kns_all = F.normalize(ks_all, dim=-1, eps=1e-6)   # unit semantic keys
            kne_all = F.normalize(ke_all, dim=-1, eps=1e-6)   # unit episodic keys

            # Gate reference norms pre-computed once: (B, L-1)
            ref_s_all = self.gate_thresh * ks_all.norm(dim=-1)
            ref_e_all = self.gate_thresh * ke_all.norm(dim=-1)

            # ── Sequential recurrence on CPU (detached) ─────────────────────
            # Detach key tensors and run the write loop inside torch.no_grad().
            # This eliminates two overhead sources:
            #   1. MPS/CUDA per-kernel-launch latency for tiny sequential bmm ops
            #      (each MPS kernel costs ~1-3 ms; L=512 → 511 launches per layer).
            #   2. Autograd graph construction for L-1 sequential tensor assignments
            #      (building O(L) grad-fn nodes on each forward is ~1.7 ms/iter on CPU).
            # Together these caused a 20× throughput regression at segment_len=512
            # (543 tok/s vs 11,700 for baseline).  With detached write the loop
            # cost is O(L × plain bmm) — no graph construction overhead.
            # Gradient signal to sem_proj/epi_proj reaches the parameters through
            # the read path (qns, qne at the query position below), which is
            # the primary learning signal for memory retrieval quality.
            cpu = torch.device("cpu")
            M_sem = M_sem.to(cpu)
            M_epi = M_epi.to(cpu)
            kns_c   = kns_all.detach().to(cpu)   # (B, L-1, d_half) — no grad
            kne_c   = kne_all.detach().to(cpu)
            ks_c    = ks_all.detach().to(cpu)
            ke_c    = ke_all.detach().to(cpu)
            ref_s_c = ref_s_all.detach().to(cpu)  # (B, L-1)
            ref_e_c = ref_e_all.detach().to(cpu)

            fires: list[torch.Tensor] = []

            with torch.no_grad():
                for t in range(L - 1):
                    kns_t = kns_c[:, t]   # (B, d_half)
                    kne_t = kne_c[:, t]
                    ks_t  = ks_c[:, t]
                    ke_t  = ke_c[:, t]

                    # Read from previous-step matrices
                    vps = torch.bmm(M_sem, kns_t.unsqueeze(-1)).squeeze(-1)   # (B, d_half)
                    vpe = torch.bmm(M_epi, kne_t.unsqueeze(-1)).squeeze(-1)

                    # OR-gate: fire if either branch error exceeds its reference
                    err_s = (ks_t - vps).norm(dim=-1)   # (B,)
                    err_e = (ke_t - vpe).norm(dim=-1)
                    fire = (
                        (err_s >= ref_s_c[:, t]) | (err_e >= ref_e_c[:, t])
                    ).float()
                    fires.append(fire)

                    # Outer-product delta-rule update
                    g3 = fire[:, None, None]   # (B, 1, 1)
                    Delta_s = torch.bmm((ks_t - vps).unsqueeze(-1), kns_t.unsqueeze(1))
                    Delta_e = torch.bmm((ke_t - vpe).unsqueeze(-1), kne_t.unsqueeze(1))

                    w_t = (t + 1) / L   # recency weight ∈ (0, 1]
                    M_sem = M_sem + (1.0 - a) * g3 * Delta_s
                    M_epi = M_epi + (1.0 - a) * w_t * g3 * Delta_e

            # Move accumulated matrices back to original device for the read phase
            M_sem = M_sem.to(device)
            M_epi = M_epi.to(device)

            # All fires are on CPU — stack+sum requires no GPU sync
            fires_t = torch.stack(fires, dim=1)   # (B, L-1) on CPU
            self._last_write_rate = fires_t.sum().item() / max(B * (L - 1), 1)
        else:
            self._last_write_rate = 0.0

        # Read at query position (last token in the sequence)
        q = x[:, -1, :]                                       # (B, d_model)
        qns = F.normalize(self.sem_proj(q), dim=-1, eps=1e-6)  # (B, d_half)
        qne = F.normalize(self.epi_proj(q), dim=-1, eps=1e-6)
        r_sem = torch.bmm(M_sem, qns.unsqueeze(-1)).squeeze(-1)   # (B, d_half)
        r_epi = torch.bmm(M_epi, qne.unsqueeze(-1)).squeeze(-1)
        r = torch.cat([r_sem, r_epi], dim=-1)                 # (B, d_model)

        # Null retrieval gate: suppress readout when memory is irrelevant.
        # Skipped (and the Linear not created) when use_null_gate=False.
        if self.use_null_gate and self.null_gate is not None:
            g_null = torch.sigmoid(self.null_gate(q))             # (B, 1)
            r = g_null * r

        return self.out_proj(r)                               # (B, d_model)
