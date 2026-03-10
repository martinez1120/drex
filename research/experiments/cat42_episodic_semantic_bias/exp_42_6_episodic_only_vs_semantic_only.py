"""
Experiment 42.6 — Episodic-Only vs Semantic-Only vs Combined Split

Hypothesis: The split memory advantage comes primarily from the semantic matrix:
semantic-only (no episodic, just M_sem at full H/2×H/2) achieves within 3% of
the full split, while episodic-only is much worse (>10% gap vs split).
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

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN    = 32
NUM_PAIRS  = 5
STEPS      = 800
BATCH      = 32
HALF       = 32


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


class SemanticOnlyModel(nn.Module):
    """Single M_s (HALF x HALF), pure delta rule, read with sem_p."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc   = Encoder(vocab_size, hidden_dim)
        self.sem_p = nn.Linear(hidden_dim, half)
        self.out   = nn.Linear(half, vocab_size)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape; half = H // 2
        M_s = torch.zeros(B, half, half, device=h.device)
        for t in range(L - 1):
            ks = self.sem_p(h[:, t, :])
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(cs)


class EpisodicOnlyModel(nn.Module):
    """Single M_e (HALF x HALF), recency-weighted delta rule, read with epi_p."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc   = Encoder(vocab_size, hidden_dim)
        self.epi_p = nn.Linear(hidden_dim, half)
        self.out   = nn.Linear(half, vocab_size)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape; half = H // 2
        M_e = torch.zeros(B, half, half, device=h.device)
        for t in range(L - 1):
            ke = self.epi_p(h[:, t, :])
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_e = M_e + ((t + 1) / L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        q  = h[:, -1, :]
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(ce)


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


class Exp426EpisodicOnlyVsSemanticOnly(Experiment):
    experiment_id = "exp_42_6"
    hypothesis = (
        "The split memory advantage comes primarily from the semantic matrix: "
        "semantic-only achieves within 3% of the full split, while episodic-only "
        "is much worse (>10% gap vs split)."
    )

    def run(self) -> ExperimentResult:
        print("Training SemanticOnlyModel ...")
        acc_semantic_only = train_eval(
            SemanticOnlyModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_semantic_only={acc_semantic_only:.4f}")

        print("Training EpisodicOnlyModel ...")
        acc_episodic_only = train_eval(
            EpisodicOnlyModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_episodic_only={acc_episodic_only:.4f}")

        print("Training SplitModel ...")
        acc_split = train_eval(
            SplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_split={acc_split:.4f}")

        gap_sem_vs_split  = round(float(acc_split) - float(acc_semantic_only), 4)
        gap_epi_vs_sem    = round(float(acc_semantic_only) - float(acc_episodic_only), 4)

        metrics = dict(
            acc_split=round(float(acc_split), 4),
            acc_semantic_only=round(float(acc_semantic_only), 4),
            acc_episodic_only=round(float(acc_episodic_only), 4),
            gap_sem_vs_split=gap_sem_vs_split,
            gap_epi_vs_sem=gap_epi_vs_sem,
        )

        if gap_sem_vs_split < 0.03 and gap_epi_vs_sem > 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Semantic dominates: split={acc_split:.3f}, "
                f"sem_only={acc_semantic_only:.3f} (gap={gap_sem_vs_split:.3f}<0.03), "
                f"epi_only={acc_episodic_only:.3f} (gap_epi_vs_sem={gap_epi_vs_sem:.3f}>0.10)."
            )
        elif acc_episodic_only > acc_semantic_only + 0.05:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Episodic dominates unexpectedly: epi_only={acc_episodic_only:.3f}, "
                f"sem_only={acc_semantic_only:.3f}, split={acc_split:.3f}."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Mixed results: split={acc_split:.3f}, sem_only={acc_semantic_only:.3f}, "
                f"epi_only={acc_episodic_only:.3f}, "
                f"gap_sem_vs_split={gap_sem_vs_split:.3f}, gap_epi_vs_sem={gap_epi_vs_sem:.3f}."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                steps=STEPS, batch=BATCH,
            ),
        )


if __name__ == "__main__":
    Exp426EpisodicOnlyVsSemanticOnly().execute()
