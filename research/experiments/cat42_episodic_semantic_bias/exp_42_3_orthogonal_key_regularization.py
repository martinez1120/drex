"""
Experiment 42.3 — Orthogonal Key Space Regularization

Hypothesis: Adding an orthogonality regularization loss (penalizing cosine similarity
between semantic and episodic key projections) improves accuracy by >3% by forcing
the two matrices to capture complementary information.
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
SEQ_LEN      = 32
NUM_PAIRS    = 5
STEPS        = 800
BATCH        = 32
HALF         = 32
ORTHO_WEIGHT = 0.01


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


class OrthoSplitModel(nn.Module):
    """Split model identical to SplitModel; orthogonality loss applied externally."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc   = Encoder(vocab_size, hidden_dim)
        self.sem_p = nn.Linear(hidden_dim, half)
        self.epi_p = nn.Linear(hidden_dim, half)
        self.out   = nn.Linear(half * 2, vocab_size)

    def ortho_loss(self):
        """Penalize cosine overlap between weight rows of sem_p and epi_p."""
        W_s = self.sem_p.weight   # (half, H)
        W_e = self.epi_p.weight   # (half, H)
        return (W_s @ W_e.T).norm() * ORTHO_WEIGHT

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


def train_eval(model, steps=800, batch=32, seq_len=32, num_pairs=5, vocab_size=64,
               use_ortho=False):
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
        loss = F.cross_entropy(model(seq), tgt)
        if use_ortho:
            loss = loss + model.ortho_loss()
        loss.backward(); opt.step(); opt.zero_grad()
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
            ok += (model(seq).argmax(-1) == tgt).sum().item(); tot += tgt.size(0)
    return ok / tot


class Exp423OrthogonalKeyRegularization(Experiment):
    experiment_id = "exp_42_3"
    hypothesis = (
        "Adding an orthogonality regularization loss (penalizing cosine similarity "
        "between semantic and episodic key projections) improves accuracy by >3% "
        "by forcing the two matrices to capture complementary information."
    )

    def run(self) -> ExperimentResult:
        print("Training SplitModel (baseline, no ortho loss) ...")
        acc_baseline = train_eval(
            SplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
            use_ortho=False,
        )
        print(f"  acc_baseline={acc_baseline:.4f}")

        print("Training OrthoSplitModel (with orthogonality loss) ...")
        acc_ortho = train_eval(
            OrthoSplitModel(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
            use_ortho=True,
        )
        print(f"  acc_ortho={acc_ortho:.4f}")

        gap = round(float(acc_ortho) - float(acc_baseline), 4)

        metrics = dict(
            acc_baseline=round(float(acc_baseline), 4),
            acc_ortho=round(float(acc_ortho), 4),
            gap=gap,
            ortho_weight=ORTHO_WEIGHT,
        )

        if acc_ortho > acc_baseline + 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Orthogonal regularization helps: baseline={acc_baseline:.3f}, "
                f"ortho={acc_ortho:.3f}, gap={gap:.3f}>0.03."
            )
        elif acc_ortho < acc_baseline - 0.02:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Orthogonal regularization hurts: baseline={acc_baseline:.3f}, "
                f"ortho={acc_ortho:.3f}, gap={gap:.3f}<-0.02."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Orthogonality effect is small: baseline={acc_baseline:.3f}, "
                f"ortho={acc_ortho:.3f}, gap={gap:.3f}."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                steps=STEPS, batch=BATCH, ortho_weight=ORTHO_WEIGHT,
            ),
        )


if __name__ == "__main__":
    Exp423OrthogonalKeyRegularization().execute()
