"""
Experiment 48.1 — Threshold Recalibration for Full OR-Gate Split System

Background: exp_47_2 showed that the full architecture (length-adaptive EMA +
episodic/semantic split + OR gate) at thresh=0.40 produces structurally elevated
write rates: wr_L32=0.774, wr_L96=0.653 (target: ≤0.70 / ≤0.50).

Root cause: With two branches each firing at p≈0.58, the OR gate fires at
Pr(A∪B) = 1−(1−p)² ≈ 0.82 for independent gates, observing 0.774 in practice.

Hypothesis: There exists thresh* ∈ (0.40, 0.80) such that the full OR-gate
split system with exp_scale (α(L)=0.95^(96/L)) achieves:
  (a) wr_L32 ∈ [0.20, 0.70]
  (b) wr_L96 ∈ [0.15, 0.50]
  (c) acc_ratio = acc_full / acc_base ≥ 0.97 at both L=32 and L=96
on ≥ 2/3 seeds.

Geometric estimate: We need each branch to fire at p≈0.35 so OR fires at ~0.58.
At thresh=0.40, p≈0.58. Threshold scales approximately linearly with p, so
thresh* ≈ 0.40 × (0.58/0.35) ≈ 0.66. Test range: {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}.

Per seed: 2 base models (no gate) × 2 lengths + 6 thresholds × 2 lengths = 14 runs.
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

VOCAB_SIZE   = 64
HIDDEN_DIM   = 64
NUM_PAIRS    = 5
STEPS        = 800
BATCH        = 32

SEQ_LENS     = [32, 96]
THRESH_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# Pass/fail criteria (same as exp_47_2)
MIN_ACC_RATIO = 0.97
WR_TARGET     = {32: (0.20, 0.70), 96: (0.15, 0.50)}

# Alpha formula: exp_scale (primary winner from exp_47_1)
def alpha_exp_scale(L: int) -> float:
    return 0.95 ** (96.0 / L)


def make_batch(batch_size: int = BATCH, seq_len: int = 32,
               num_pairs: int = NUM_PAIRS, vocab_size: int = VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 3, (num_pairs * 4,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class FullAdaptiveModel(nn.Module):
    """
    Full architecture: length-adaptive EMA (exp_scale) + episodic/semantic split
    + OR write gate at configurable threshold.
    use_gate=False gives the adaptive-EMA-split baseline (no gate writes).
    """

    def __init__(self, use_gate: bool = True,
                 gate_thresh: float = 0.40,
                 hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.use_gate    = use_gate
        self.gate_thresh = gate_thresh

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        half = hidden_dim // 2
        self.sem_p = nn.Linear(hidden_dim, half, bias=False)
        self.epi_p = nn.Linear(hidden_dim, half, bias=False)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)
        self._wr_count = 0
        self._wr_total = 0

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def reset_wr(self):
        self._wr_count = 0
        self._wr_total = 0

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        alpha = alpha_exp_scale(L)   # length-adaptive coefficient
        half  = H // 2
        M_s   = torch.zeros(B, half, half, device=h.device)
        M_e   = torch.zeros(B, half, half, device=h.device)

        for t in range(L - 1):
            ks  = self.sem_p(h[:, t, :])
            ke  = self.epi_p(h[:, t, :])
            kns = F.normalize(ks, dim=-1)
            kne = F.normalize(ke, dim=-1)
            vps = torch.bmm(M_s, kns.unsqueeze(-1)).squeeze(-1)
            vpe = torch.bmm(M_e, kne.unsqueeze(-1)).squeeze(-1)
            Delta_s = torch.bmm((ks - vps).unsqueeze(-1), kns.unsqueeze(1))
            Delta_e = torch.bmm((ke - vpe).unsqueeze(-1), kne.unsqueeze(1))

            if self.use_gate:
                err_s = (ks - vps).norm(dim=-1)
                err_e = (ke - vpe).norm(dim=-1)
                ref_s = self.gate_thresh * ks.norm(dim=-1)
                ref_e = self.gate_thresh * ke.norm(dim=-1)
                fire  = ((err_s >= ref_s) | (err_e >= ref_e)).float()
                self._wr_count += fire.sum().item()
                self._wr_total += B
                gate     = fire[:, None, None]
                Delta_s  = gate * Delta_s
                Delta_e  = gate * Delta_e

            w_t = (t + 1) / L   # episodic recency weight
            M_s = M_s + (1.0 - alpha) * Delta_s
            M_e = M_e + (1.0 - alpha) * w_t * Delta_e

        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(torch.cat([cs, ce], dim=-1)))


def train_eval(model: FullAdaptiveModel, seq_len: int,
               steps: int = STEPS, batch: int = BATCH) -> tuple[float, float]:
    opt = Adam(model.parameters(), lr=3e-4)
    model.reset_wr()
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len=seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step()
        opt.zero_grad()
    final_wr = model.write_rate()
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len=seq_len)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot, final_wr


class Exp481ThreshSweepAdaptive(Experiment):
    experiment_id = "exp_48_1"
    hypothesis = (
        "There exists thresh* ∈ (0.40, 0.80) such that the full OR-gate split "
        "system (exp_scale α(L)=0.95^(96/L)) achieves wr_L32 ∈ [0.20, 0.70], "
        "wr_L96 ∈ [0.15, 0.50], and acc_ratio ≥ 0.97 at both L=32 and L=96, "
        "on ≥ 2/3 seeds. Expected thresh* ≈ 0.60–0.65."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}

        # Train base (no-gate) models once per sequence length
        base_accs: dict[int, float] = {}
        for slen in SEQ_LENS:
            alpha_val = alpha_exp_scale(slen)
            print(f"\n  Base (no gate) L={slen}, α={alpha_val:.4f} ...")
            base = FullAdaptiveModel(use_gate=False)
            acc_base, _ = train_eval(base, seq_len=slen)
            base_accs[slen] = acc_base
            results[f"acc_base_L{slen}"] = round(acc_base, 4)
            print(f"    acc={acc_base:.4f}")

        # Sweep thresholds
        winning_thresholds: list[float] = []
        thresh_details: dict[float, dict] = {}

        for thresh in THRESH_SWEEP:
            print(f"\n  thresh={thresh:.2f}")
            thresh_pass = True
            failed: list[str] = []
            t_res: dict = {}

            for slen in SEQ_LENS:
                alpha_val = alpha_exp_scale(slen)
                print(f"    L={slen}, α={alpha_val:.4f} ...")
                gated = FullAdaptiveModel(use_gate=True, gate_thresh=thresh)
                acc_full, wr_full = train_eval(gated, seq_len=slen)
                acc_base = base_accs[slen]
                ratio = acc_full / max(acc_base, 1e-6)

                t_res[f"acc_full_L{slen}"] = round(acc_full, 4)
                t_res[f"wr_full_L{slen}"]  = round(wr_full, 4)
                t_res[f"ratio_L{slen}"]    = round(ratio, 4)

                wr_lo, wr_hi = WR_TARGET[slen]
                acc_ok = ratio >= MIN_ACC_RATIO
                wr_ok  = wr_lo <= wr_full <= wr_hi
                t_res[f"acc_ok_L{slen}"] = acc_ok
                t_res[f"wr_ok_L{slen}"]  = wr_ok
                print(f"      acc={acc_full:.4f}  wr={wr_full:.3f}  ratio={ratio:.3f}  "
                      f"acc_ok={acc_ok}  wr_ok={wr_ok}")

                if not acc_ok:
                    thresh_pass = False
                    failed.append(f"acc_ratio={ratio:.3f} < {MIN_ACC_RATIO} at L={slen}")
                if not wr_ok:
                    thresh_pass = False
                    failed.append(f"wr={wr_full:.3f} outside {(wr_lo, wr_hi)} at L={slen}")

            t_res["pass"] = thresh_pass
            thresh_details[thresh] = t_res

            tkey = f"t{int(thresh * 100)}"
            for k, v in t_res.items():
                results[f"{tkey}_{k}"] = v

            if thresh_pass:
                winning_thresholds.append(thresh)
                print(f"  *** PASS at thresh={thresh:.2f} ***")
            else:
                print(f"  FAIL at thresh={thresh:.2f} — {'; '.join(failed)}")

        results["winning_thresholds"] = winning_thresholds

        # Determine outcome
        if winning_thresholds:
            thresh_star = winning_thresholds[0]
            outcome = OUTCOME_SUPPORTED
            w = thresh_details[thresh_star]
            notes = (
                f"thresh* = {thresh_star:.2f} resolves OR-gate write-rate inflation. "
                f"wr(L=32)={w.get('wr_full_L32', '?')}, "
                f"wr(L=96)={w.get('wr_full_L96', '?')}. "
                f"acc_ratio(L=32)={w.get('ratio_L32', '?')}, "
                f"acc_ratio(L=96)={w.get('ratio_L96', '?')}. "
                f"Use α(L)=0.95^(96/L) + OR gate + thresh*={thresh_star:.2f} in "
                f"production MemoryModule. All blockers resolved."
            )
        else:
            # Check partial: wr in range but acc fails, vs wr never in range
            wr32_ever_ok = any(
                WR_TARGET[32][0] <= thresh_details[t].get("wr_full_L32", 0.0) <= WR_TARGET[32][1]
                for t in THRESH_SWEEP
            )
            wr96_ever_ok = any(
                WR_TARGET[96][0] <= thresh_details[t].get("wr_full_L96", 0.0) <= WR_TARGET[96][1]
                for t in THRESH_SWEEP
            )
            if wr32_ever_ok and wr96_ever_ok:
                outcome = OUTCOME_INCONCLUSIVE
                notes = (
                    "Write rates reached target at some threshold(s) but accuracy ratio "
                    "did not meet the 0.97 floor. Consider AND gate (exp_48_2) or "
                    "lowering MIN_ACC_RATIO to 0.95."
                )
            elif wr32_ever_ok:
                outcome = OUTCOME_INCONCLUSIVE
                notes = (
                    "L=32 wr entered target range but L=96 wr remained elevated. "
                    "OR gate may require different thresholds per length."
                )
            else:
                outcome = OUTCOME_REFUTED
                notes = (
                    "No threshold in sweep achieved both wr targets simultaneously. "
                    "OR gate structurally incompatible with joint wr criteria. "
                    "Recommend AND gate (exp_48_2): each branch gates its own writes "
                    "independently, avoiding OR inflation."
                )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp481ThreshSweepAdaptive().execute()
