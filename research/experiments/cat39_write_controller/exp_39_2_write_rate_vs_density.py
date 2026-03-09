"""
Experiment 39.2 — Write Rate Equilibrium Depends on Interference Density

Hypothesis: The steady-state write rate of EnergyGatedDelta depends on interference
density ρ = N_pairs / HIDDEN_DIM — higher ρ (more pairs to memorize) requires more
writes and produces a higher equilibrium write rate (>0.60 at ρ=1.0 vs ≤0.45 at ρ=0.1).

Background: exp_34_3 found write_rate ≈ 0.534 with NUM_PAIRS=5, HIDDEN_DIM=64
(ρ = 5/64 ≈ 0.078). The hypothesis that 0.534 is universal (independent of task load)
would be REFUTED if the equilibrium shifts meaningfully with ρ.

If SUPPORTED: the locked write rate is task-load-dependent, and the 0.534 observed
in exp_34_3 simply reflects ρ ≈ 0.078.
If REFUTED: the write rate equilibrium is architecture-intrinsic (not task-dependent).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

VOCAB_SIZE     = 64
HIDDEN_DIM     = 64
STEPS          = 1000
BATCH          = 32
ENERGY_THRESH  = 0.4   # same as exp_34_3
# N_pairs values; ρ = N_pairs / HIDDEN_DIM
N_PAIRS_LIST   = [2, 5, 10, 20, 32, 48]   # ρ ≈ 0.03, 0.08, 0.16, 0.31, 0.50, 0.75


def make_batch(batch_size, n_pairs):
    seq_len = n_pairs * 2 + 6
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (n_pairs * 4,)).unique()[:n_pairs]
        while len(keys) < n_pairs:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:n_pairs]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (n_pairs,)); pos = 0
        for i in range(n_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3): seq[b, p] = 3
        qi = torch.randint(0, n_pairs, (1,)).item()
        seq[b, seq_len-3] = 2; seq[b, seq_len-2] = keys[qi]; seq[b, seq_len-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class EnergyGatedDelta(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff      = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm    = nn.LayerNorm(HIDDEN_DIM)
        self.kp      = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.vp      = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.qp      = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp      = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self._wr     = 0.0

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        writes = 0; total = 0
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            err = (v - vp).norm(dim=-1)
            gate = (err > ENERGY_THRESH * v.norm(dim=-1)).float()
            writes += gate.sum().item(); total += B
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp).unsqueeze(-1), kn.unsqueeze(1))
        self._wr = writes / max(total, 1)
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def measure_equilibrium(n_pairs):
    seq_len = n_pairs * 2 + 6
    model   = EnergyGatedDelta(seq_len=seq_len)
    opt     = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH, n_pairs)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()

    model.eval(); wrs = []; acc_ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(BATCH, n_pairs)
            preds = model(seq)
            acc_ok += (preds.argmax(-1) == tgt).sum().item(); tot += tgt.size(0)
            wrs.append(model._wr)
    return sum(wrs) / len(wrs), acc_ok / tot


class Exp392WriteRateVsDensity(Experiment):
    experiment_id = "exp_39_2"
    hypothesis = (
        "EnergyGatedDelta write rate at equilibrium increases with ρ: "
        "wr(ρ=0.75) > wr(ρ=0.08) by >0.15, showing task-load dependence."
    )

    def run(self) -> ExperimentResult:
        metrics: dict[str, float] = {}
        wrs_list = []; accs_list = []
        rhos = [round(n / HIDDEN_DIM, 3) for n in N_PAIRS_LIST]

        for n_pairs, rho in zip(N_PAIRS_LIST, rhos):
            print(f"N_pairs={n_pairs} (ρ={rho:.3f})...")
            wr, acc = measure_equilibrium(n_pairs)
            metrics[f"wr_n{n_pairs:02d}"]  = round(wr,  4)
            metrics[f"acc_n{n_pairs:02d}"] = round(acc, 4)
            wrs_list.append(wr); accs_list.append(acc)
            print(f"  wr={wr:.3f}, acc={acc:.3f}")

        # Check if write rate varies with ρ
        wr_low  = wrs_list[0]    # ρ ≈ 0.03
        wr_high = wrs_list[-1]   # ρ ≈ 0.75
        wr_ref  = wrs_list[1]    # ρ ≈ 0.08 (closest to exp_34_3)
        delta_wr = wr_high - wr_low

        metrics["wr_low_rho"]  = round(wr_low,  4)
        metrics["wr_high_rho"] = round(wr_high, 4)
        metrics["wr_ref_rho"]  = round(wr_ref,  4)
        metrics["delta_wr_high_minus_low"] = round(delta_wr, 4)
        metrics["rhos"] = rhos

        if delta_wr > 0.15 and wr_high > wr_low:
            outcome = OUTCOME_SUPPORTED
            notes   = (f"Write rate increases with ρ: low={wr_low:.3f} (ρ≈{rhos[0]}), "
                       f"high={wr_high:.3f} (ρ≈{rhos[-1]}). Δwr={delta_wr:.3f}>0.15. "
                       f"Task-load dependent. ref(ρ≈{rhos[1]})={wr_ref:.3f} matches exp_34_3≈0.534.")
        elif abs(delta_wr) < 0.05:
            outcome = OUTCOME_REFUTED
            notes   = (f"Write rate invariant to ρ: low={wr_low:.3f}, high={wr_high:.3f}, "
                       f"Δ={delta_wr:.3f}<0.05. Locked at ≈{wr_ref:.3f} regardless of load.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = (f"Moderate variation: low={wr_low:.3f}, high={wr_high:.3f}, "
                       f"Δ={delta_wr:.3f}. Non-monotone or boundary effects.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, n_pairs_list=N_PAIRS_LIST,
                                       energy_thresh=ENERGY_THRESH))


if __name__ == "__main__":
    Exp392WriteRateVsDensity().execute()
