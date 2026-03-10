"""
Experiment 42.8 — Multi-Scale Episodic Memory (Fast + Slow)

Hypothesis: Replacing the single episodic matrix with two episodic matrices at
different timescales (fast: linear recency, slow: sqrt recency) improves accuracy
over single-scale episodic by >5%, capturing both recent and distant temporal structure.
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
SEQ_LEN        = 32
NUM_PAIRS      = 5
STEPS          = 800
BATCH          = 32
HALF           = 32    # sem_dim for SingleScaleSplitModel
SEM_DIM        = 32    # semantic dim for MultiScaleEpisodicModel
EPI_FAST_DIM   = 16    # fast episodic dim
EPI_SLOW_DIM   = 16    # slow episodic dim


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


class SingleScaleSplitModel(nn.Module):
    """Standard split: sem_dim=32, epi_dim=32, concat -> Linear(64, VOCAB)."""
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


class MultiScaleEpisodicModel(nn.Module):
    """sem_dim=32, epi_fast_dim=16 (linear recency), epi_slow_dim=16 (sqrt recency).
    Read: concat [cs_sem(32), ce_fast(16), ce_slow(16)] = 64 -> Linear(64, VOCAB)."""
    def __init__(self, vocab_size=64, hidden_dim=64,
                 sem_dim=32, epi_fast_dim=16, epi_slow_dim=16):
        super().__init__()
        out_dim = sem_dim + epi_fast_dim + epi_slow_dim
        self.enc          = Encoder(vocab_size, hidden_dim)
        self.sem_p        = nn.Linear(hidden_dim, sem_dim)
        self.epi_fast_p   = nn.Linear(hidden_dim, epi_fast_dim)
        self.epi_slow_p   = nn.Linear(hidden_dim, epi_slow_dim)
        self.out          = nn.Linear(out_dim, vocab_size)
        self.sem_dim      = sem_dim
        self.epi_fast_dim = epi_fast_dim
        self.epi_slow_dim = epi_slow_dim

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape
        sd = self.sem_dim; fd = self.epi_fast_dim; sld = self.epi_slow_dim
        M_s  = torch.zeros(B, sd,  sd,  device=h.device)
        M_ef = torch.zeros(B, fd,  fd,  device=h.device)
        M_es = torch.zeros(B, sld, sld, device=h.device)
        for t in range(L - 1):
            ks  = self.sem_p(h[:, t, :])
            kef = self.epi_fast_p(h[:, t, :])
            kes = self.epi_slow_p(h[:, t, :])

            # semantic: pure delta rule
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))

            # fast episodic: linear recency weight
            w_fast = (t + 1) / L
            vpef = torch.bmm(M_ef, kef.unsqueeze(-1)).squeeze(-1)
            dvef = kef - vpef / (kef.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_ef = M_ef + w_fast * torch.bmm(dvef.unsqueeze(-1), kef.unsqueeze(1))

            # slow episodic: sqrt recency weight
            w_slow = ((t + 1) / L) ** 0.5
            vpes = torch.bmm(M_es, kes.unsqueeze(-1)).squeeze(-1)
            dves = kes - vpes / (kes.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_es = M_es + w_slow * torch.bmm(dves.unsqueeze(-1), kes.unsqueeze(1))

        q   = h[:, -1, :]
        cs  = torch.bmm(M_s,  self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        cef = torch.bmm(M_ef, self.epi_fast_p(q).unsqueeze(-1)).squeeze(-1)
        ces = torch.bmm(M_es, self.epi_slow_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, cef, ces], -1))


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


class Exp428MultiScaleEpisodic(Experiment):
    experiment_id = "exp_42_8"
    hypothesis = (
        "Replacing the single episodic matrix with two episodic matrices at "
        "different timescales (fast: linear recency, slow: sqrt recency) "
        "improves accuracy over single-scale episodic by >5%, capturing both "
        "recent and distant temporal structure."
    )

    def run(self) -> ExperimentResult:
        print("Training SingleScaleSplitModel ...")
        acc_single_scale = train_eval(
            SingleScaleSplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_single_scale={acc_single_scale:.4f}")

        print("Training MultiScaleEpisodicModel ...")
        acc_multiscale = train_eval(
            MultiScaleEpisodicModel(
                VOCAB_SIZE, HIDDEN_DIM,
                sem_dim=SEM_DIM, epi_fast_dim=EPI_FAST_DIM, epi_slow_dim=EPI_SLOW_DIM,
            ),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_multiscale={acc_multiscale:.4f}")

        gap = round(float(acc_multiscale) - float(acc_single_scale), 4)

        metrics = dict(
            acc_single_scale=round(float(acc_single_scale), 4),
            acc_multiscale=round(float(acc_multiscale), 4),
            gap=gap,
            sem_dim=SEM_DIM,
            epi_fast_dim=EPI_FAST_DIM,
            epi_slow_dim=EPI_SLOW_DIM,
        )

        if acc_multiscale > acc_single_scale + 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Multi-scale episodic outperforms single-scale: "
                f"multiscale={acc_multiscale:.3f}, single={acc_single_scale:.3f}, "
                f"gap={gap:.3f}>0.05."
            )
        elif acc_multiscale < acc_single_scale - 0.03:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Multi-scale episodic hurts performance: "
                f"multiscale={acc_multiscale:.3f}, single={acc_single_scale:.3f}, "
                f"gap={gap:.3f}<-0.03."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Multi-scale effect is below threshold: "
                f"multiscale={acc_multiscale:.3f}, single={acc_single_scale:.3f}, "
                f"gap={gap:.3f}."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                steps=STEPS, batch=BATCH,
                sem_dim=SEM_DIM, epi_fast_dim=EPI_FAST_DIM, epi_slow_dim=EPI_SLOW_DIM,
            ),
        )


if __name__ == "__main__":
    Exp428MultiScaleEpisodic().execute()
