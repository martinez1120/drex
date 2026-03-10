"""
Experiment 42.7 — Episodic/Semantic Split Memory at Long Context

Hypothesis: The episodic/semantic split advantage (from exp_36_3 at SEQ_LEN=32)
persists at SEQ_LEN=96: split outperforms unified by >3% even at 3x longer contexts,
confirming scalability of the inductive bias.
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

VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
SEQ_LEN_32  = 32
SEQ_LEN_96  = 96
NUM_PAIRS_5 = 5
NUM_PAIRS_8 = 8
STEPS_800   = 800
STEPS_1000  = 1000
BATCH       = 32
HALF        = 32


def make_batch(batch_size, seq_len=32, num_pairs=5, vocab_size=64):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 3, (num_pairs * 4,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,)); pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3): seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len-3] = 2; seq[b, seq_len-2] = keys[qi]; seq[b, seq_len-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        e = self.embed(x); return self.norm(e + self.ff(e))


class SplitModel(nn.Module):
    """Fixed 50/50 episodic/semantic split with recency weighting on episodic."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc   = Encoder(vocab_size, hidden_dim)
        self.sem_p = nn.Linear(hidden_dim, half)
        self.epi_p = nn.Linear(hidden_dim, half)
        self.out   = nn.Linear(half * 2, vocab_size)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape; half = H // 2
        M_s = torch.zeros(B, half, half, device=h.device)
        M_e = torch.zeros(B, half, half, device=h.device)
        for t in range(L - 1):
            ks = self.sem_p(h[:, t, :]); ke = self.epi_p(h[:, t, :])
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_e = M_e + ((t + 1) / L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


class UnifiedModel(nn.Module):
    """Single unified memory matrix (no split)."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        self.enc = Encoder(vocab_size, hidden_dim)
        self.kp  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rp  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            M = M + torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
        q = self.kp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def train_eval(model, steps=800, batch=32, seq_len=32, num_pairs=5, vocab_size=64):
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
            ok += (model(seq).argmax(-1) == tgt).sum().item(); tot += tgt.size(0)
    return ok / tot


class Exp427SplitMemoryAtLongContext(Experiment):
    experiment_id = "exp_42_7"
    hypothesis = (
        "The episodic/semantic split advantage (from exp_36_3 at SEQ_LEN=32) "
        "persists at SEQ_LEN=96: split outperforms unified by >3% even at 3x "
        "longer contexts, confirming scalability of the inductive bias."
    )

    def run(self) -> ExperimentResult:
        print("Training SplitModel at SEQ_LEN=96 ...")
        acc_split_96 = train_eval(
            SplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS_1000, batch=BATCH, seq_len=SEQ_LEN_96,
            num_pairs=NUM_PAIRS_8, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_split_96={acc_split_96:.4f}")

        print("Training UnifiedModel at SEQ_LEN=96 ...")
        acc_unified_96 = train_eval(
            UnifiedModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS_1000, batch=BATCH, seq_len=SEQ_LEN_96,
            num_pairs=NUM_PAIRS_8, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_unified_96={acc_unified_96:.4f}")

        print("Training SplitModel at SEQ_LEN=32 ...")
        acc_split_32 = train_eval(
            SplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS_800, batch=BATCH, seq_len=SEQ_LEN_32,
            num_pairs=NUM_PAIRS_5, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_split_32={acc_split_32:.4f}")

        print("Training UnifiedModel at SEQ_LEN=32 ...")
        acc_unified_32 = train_eval(
            UnifiedModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS_800, batch=BATCH, seq_len=SEQ_LEN_32,
            num_pairs=NUM_PAIRS_5, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_unified_32={acc_unified_32:.4f}")

        gap_96 = round(float(acc_split_96) - float(acc_unified_96), 4)
        gap_32 = round(float(acc_split_32) - float(acc_unified_32), 4)

        metrics = dict(
            acc_split_96=round(float(acc_split_96), 4),
            acc_unified_96=round(float(acc_unified_96), 4),
            gap_96=gap_96,
            acc_split_32=round(float(acc_split_32), 4),
            acc_unified_32=round(float(acc_unified_32), 4),
            gap_32=gap_32,
        )

        if gap_96 > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Split advantage persists at long context: "
                f"split_96={acc_split_96:.3f}, unified_96={acc_unified_96:.3f}, "
                f"gap_96={gap_96:.3f}>0.03 (gap_32={gap_32:.3f} for reference)."
            )
        elif gap_96 < 0.0:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Split does not help at long context: "
                f"split_96={acc_split_96:.3f}, unified_96={acc_unified_96:.3f}, "
                f"gap_96={gap_96:.3f}<0.0."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Split advantage is positive but small at long context: "
                f"split_96={acc_split_96:.3f}, unified_96={acc_unified_96:.3f}, "
                f"gap_96={gap_96:.3f} (threshold 0.03)."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                seq_len_short=SEQ_LEN_32, seq_len_long=SEQ_LEN_96,
                num_pairs_short=NUM_PAIRS_5, num_pairs_long=NUM_PAIRS_8,
                steps_short=STEPS_800, steps_long=STEPS_1000, batch=BATCH,
            ),
        )


if __name__ == "__main__":
    Exp427SplitMemoryAtLongContext().execute()
