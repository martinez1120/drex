"""
Experiment 42.2 — Shared vs Separate Key Projections for Episodic/Semantic

Hypothesis: Using separate key projections for episodic and semantic matrices
(two distinct Linear layers) is essential: sharing a single projection for both
drops accuracy by >5%, showing that the matrices need independent key spaces.
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


class SeparateKeysSplitModel(nn.Module):
    """Two distinct key projections: sem_p and epi_p (independent key spaces)."""
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


class SharedKeysSplitModel(nn.Module):
    """Single key projection kp: H -> H, split in half for sem/epi key spaces."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc = Encoder(vocab_size, hidden_dim)
        self.kp  = nn.Linear(hidden_dim, hidden_dim)   # shared projection, split at inference
        self.out = nn.Linear(half * 2, vocab_size)
        self.half = half

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape; half = self.half
        M_s = torch.zeros(B, half, half, device=h.device)
        M_e = torch.zeros(B, half, half, device=h.device)
        for t in range(L - 1):
            proj = self.kp(h[:, t, :])
            ks = proj[:, :half]; ke = proj[:, half:]
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_e = M_e + ((t + 1) / L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        proj_q = self.kp(h[:, -1, :])
        qs = proj_q[:, :half]; qe = proj_q[:, half:]
        cs = torch.bmm(M_s, qs.unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, qe.unsqueeze(-1)).squeeze(-1)
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


class Exp422SharedVsSeparateKeys(Experiment):
    experiment_id = "exp_42_2"
    hypothesis = (
        "Using separate key projections for episodic and semantic matrices "
        "(two distinct Linear layers) is essential: sharing a single projection "
        "for both drops accuracy by >5%, showing that the matrices need "
        "independent key spaces."
    )

    def run(self) -> ExperimentResult:
        print("Training SeparateKeysSplitModel ...")
        acc_separate = train_eval(
            SeparateKeysSplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_separate={acc_separate:.4f}")

        print("Training SharedKeysSplitModel ...")
        acc_shared = train_eval(
            SharedKeysSplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_shared={acc_shared:.4f}")

        gap = round(float(acc_separate) - float(acc_shared), 4)

        metrics = dict(
            acc_separate=round(float(acc_separate), 4),
            acc_shared=round(float(acc_shared), 4),
            gap=gap,
        )

        if acc_separate > acc_shared + 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Separate keys outperform shared keys: separate={acc_separate:.3f}, "
                f"shared={acc_shared:.3f}, gap={gap:.3f}>0.05."
            )
        elif acc_shared >= acc_separate - 0.02:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Shared keys are competitive: separate={acc_separate:.3f}, "
                f"shared={acc_shared:.3f}, gap={gap:.3f} (within 0.02)."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Some advantage for separate keys but below threshold: "
                f"separate={acc_separate:.3f}, shared={acc_shared:.3f}, gap={gap:.3f}."
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
    Exp422SharedVsSeparateKeys().execute()
