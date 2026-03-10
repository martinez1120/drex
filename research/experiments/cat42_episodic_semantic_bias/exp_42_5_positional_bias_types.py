"""
Experiment 42.5 — Positional Bias Types for Episodic Write

Hypothesis: A learned positional weight function (a small MLP mapping position
to a scalar weight) outperforms linear recency (t/L) and uniform (1.0) on
episodic writes by >3%, showing that the optimal temporal discount is not linear.
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


class LinearRecencyModel(nn.Module):
    """Episodic write weight = (t+1)/L  (standard linear recency)."""
    def __init__(self, vocab_size=64, hidden_dim=64, seq_len=32):
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
            w = (t + 1) / L
            M_e = M_e + w * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


class SqrtRecencyModel(nn.Module):
    """Episodic write weight = sqrt((t+1)/L)."""
    def __init__(self, vocab_size=64, hidden_dim=64, seq_len=32):
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
            w = ((t + 1) / L) ** 0.5
            M_e = M_e + w * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


class LearnedRecencyModel(nn.Module):
    """Episodic write weight = sigmoid(pos_mlp(pos_embed[t])) — learned per position."""
    def __init__(self, vocab_size=64, hidden_dim=64, seq_len=32):
        super().__init__()
        half = hidden_dim // 2
        self.enc      = Encoder(vocab_size, hidden_dim)
        self.sem_p    = nn.Linear(hidden_dim, half)
        self.epi_p    = nn.Linear(hidden_dim, half)
        self.out      = nn.Linear(half * 2, vocab_size)
        self.pos_emb  = nn.Embedding(seq_len, 16)
        self.pos_mlp  = nn.Linear(16, 1)
        self.seq_len  = seq_len

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape; half = H // 2
        M_s = torch.zeros(B, half, half, device=h.device)
        M_e = torch.zeros(B, half, half, device=h.device)
        pos_idx = torch.arange(L, device=h.device)
        pos_weights = torch.sigmoid(self.pos_mlp(self.pos_emb(pos_idx))).squeeze(-1)  # (L,)
        for t in range(L - 1):
            ks = self.sem_p(h[:, t, :]); ke = self.epi_p(h[:, t, :])
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
            w = pos_weights[t]   # scalar
            M_e = M_e + w * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
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


class Exp425PositionalBiasTypes(Experiment):
    experiment_id = "exp_42_5"
    hypothesis = (
        "A learned positional weight function (small MLP mapping position to scalar) "
        "outperforms linear recency (t/L) and uniform (1.0) on episodic writes by >3%, "
        "showing that the optimal temporal discount is not linear."
    )

    def run(self) -> ExperimentResult:
        print("Training LinearRecencyModel ...")
        acc_linear = train_eval(
            LinearRecencyModel(VOCAB_SIZE, HIDDEN_DIM, SEQ_LEN),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_linear={acc_linear:.4f}")

        print("Training SqrtRecencyModel ...")
        acc_sqrt = train_eval(
            SqrtRecencyModel(VOCAB_SIZE, HIDDEN_DIM, SEQ_LEN),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_sqrt={acc_sqrt:.4f}")

        print("Training LearnedRecencyModel ...")
        acc_learned = train_eval(
            LearnedRecencyModel(VOCAB_SIZE, HIDDEN_DIM, SEQ_LEN),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_learned={acc_learned:.4f}")

        gap_learned_over_linear = round(float(acc_learned) - float(acc_linear), 4)

        metrics = dict(
            acc_linear=round(float(acc_linear), 4),
            acc_sqrt=round(float(acc_sqrt), 4),
            acc_learned=round(float(acc_learned), 4),
            gap_learned_over_linear=gap_learned_over_linear,
        )

        if acc_learned > acc_linear + 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Learned recency beats linear: learned={acc_learned:.3f}, "
                f"linear={acc_linear:.3f}, gap={gap_learned_over_linear:.3f}>0.03."
            )
        elif acc_linear >= acc_learned - 0.01:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Linear recency is sufficient: linear={acc_linear:.3f}, "
                f"learned={acc_learned:.3f}, gap={gap_learned_over_linear:.3f} (within 0.01)."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Learned recency has modest advantage: learned={acc_learned:.3f}, "
                f"linear={acc_linear:.3f}, gap={gap_learned_over_linear:.3f}."
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
    Exp425PositionalBiasTypes().execute()
