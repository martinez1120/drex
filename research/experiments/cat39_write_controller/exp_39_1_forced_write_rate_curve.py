"""
Experiment 39.1 — Accuracy vs Forced Write Rate Curve

Hypothesis: There exists a write-rate sweet spot near 0.5 where accuracy peaks;
forcing the write rate to 0.10 or 0.90 (via threshold scaling) degrades accuracy
by >10% each, forming a concave accuracy curve with maximum around 0.50.

Background: exp_34_3 found EnergyGatedDelta locks at write_rate ≈ 0.534 regardless
of training stage. One interpretation: 0.54 is the optimal write rate for this task.
We test this by hard-forcing different write rates via threshold scaling and measuring
the accuracy penalty.

Method: Train EnergyGated models with threshold scales {0.1, 0.3, 0.5, 0.7, 0.9, 1.1}
where higher scale → more selective (lower write rate). Measure resulting write rate
and accuracy. Test if the default equilibrium (0.54) is a near-optimal value.
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

VOCAB_SIZE        = 64
HIDDEN_DIM        = 64
SEQ_LEN           = 24
NUM_PAIRS         = 5
STEPS             = 800
BATCH             = 32
THRESHOLD_SCALES  = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.4]


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,)); pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class FixedRateDelta(nn.Module):
    """EnergyGated delta rule with fixed threshold scale (not learnable)."""
    def __init__(self, thresh_scale=0.4):
        super().__init__()
        self.thresh_scale = thresh_scale
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        self.kp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.vp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.qp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self._write_rate = 0.0

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        writes = 0; total = 0
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            error = (v - vp).norm(dim=-1)
            gate  = (error > self.thresh_scale * v.norm(dim=-1)).float()
            writes += gate.sum().item(); total += B
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp).unsqueeze(-1), kn.unsqueeze(1))
        self._write_rate = writes / max(total, 1)
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def train_and_measure(thresh_scale):
    model = FixedRateDelta(thresh_scale=thresh_scale)
    opt   = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()

    model.eval(); ok = tot = 0; write_rates = []
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_batch(BATCH)
            pred = model(seq)
            ok  += (pred.argmax(-1) == tgt).sum().item(); tot += tgt.size(0)
            write_rates.append(model._write_rate)
    acc = ok / tot
    avg_wr = sum(write_rates) / len(write_rates)
    return acc, avg_wr


class Exp391ForcedWriteRateCurve(Experiment):
    experiment_id = "exp_39_1"
    hypothesis = (
        "Accuracy peaks near write_rate ≈ 0.50; forcing rate to 0.10 or 0.90 "
        "degrades accuracy by >10% each (concave accuracy vs write-rate curve)."
    )

    def run(self) -> ExperimentResult:
        metrics: dict[str, float] = {}
        accs = []; wrs = []
        for scale in THRESHOLD_SCALES:
            print(f"Training thresh_scale={scale:.1f}...")
            acc, wr = train_and_measure(scale)
            name = f"ts{int(scale*10):02d}"
            metrics[f"acc_{name}"]  = round(acc, 4)
            metrics[f"wr_{name}"]   = round(wr,  4)
            accs.append(acc); wrs.append(wr)
            print(f"  acc={acc:.3f}, write_rate={wr:.3f}")

        # Find peak accuracy and write rate at peak
        peak_idx  = accs.index(max(accs))
        peak_acc  = accs[peak_idx]
        peak_wr   = wrs[peak_idx]
        peak_scale = THRESHOLD_SCALES[peak_idx]

        # Find acc at extreme write rates (<0.20 and >0.80)
        low_wr_acc  = next((a for a, w in zip(accs, wrs) if w < 0.20), None)
        high_wr_acc = next((a for a, w in zip(accs, wrs) if w > 0.80), None)

        metrics["peak_acc"]   = round(peak_acc,   4)
        metrics["peak_wr"]    = round(peak_wr,     4)
        metrics["peak_scale"] = peak_scale
        metrics["drop_at_low_wr"]  = round(peak_acc - (low_wr_acc  or peak_acc), 4)
        metrics["drop_at_high_wr"] = round(peak_acc - (high_wr_acc or peak_acc), 4)

        low_drop  = metrics["drop_at_low_wr"]
        high_drop = metrics["drop_at_high_wr"]
        wr_near_optimal = 0.40 <= peak_wr <= 0.65

        if low_drop > 0.10 and high_drop > 0.10 and wr_near_optimal:
            outcome = OUTCOME_SUPPORTED
            notes   = (f"Concave curve confirmed: peak_wr={peak_wr:.3f} (scale={peak_scale}), "
                       f"low_drop={low_drop:.3f}, high_drop={high_drop:.3f}. "
                       f"Equilibrium ~0.54 appears near-optimal.")
        elif low_drop <= 0.10 and high_drop <= 0.10:
            outcome = OUTCOME_REFUTED
            notes   = (f"Flat accuracy curve: low_drop={low_drop:.3f}, "
                       f"high_drop={high_drop:.3f} both ≤ 0.10. Write rate doesn't matter.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = (f"Asymmetric: peak_wr={peak_wr:.3f}, low_drop={low_drop:.3f}, "
                       f"high_drop={high_drop:.3f}. One extreme hurts more than other.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, threshold_scales=THRESHOLD_SCALES))


if __name__ == "__main__":
    Exp391ForcedWriteRateCurve().execute()
