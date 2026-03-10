"""
Experiment 42.4 — Read Combination Methods for Episodic/Semantic

Hypothesis: A learned attention gate over [sem_out, epi_out] (softmax weighting)
outperforms simple concatenation by >5%, showing that dynamic read combination
extracts more information than fixed 50/50 concatenation.
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


def _write_matrices(enc, sem_p, epi_p, seq):
    """Shared write logic: return M_s, M_e, final query h."""
    h = enc(seq); B, L, H = h.shape; half = sem_p.out_features
    M_s = torch.zeros(B, half, half, device=h.device)
    M_e = torch.zeros(B, half, half, device=h.device)
    for t in range(L - 1):
        ks = sem_p(h[:, t, :]); ke = epi_p(h[:, t, :])
        vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
        dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
        M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
        vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
        dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
        M_e = M_e + ((t + 1) / L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
    return M_s, M_e, h[:, -1, :]


class ConcatRead(nn.Module):
    """cat([cs, ce], dim=-1) -> Linear(H, VOCAB)."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc   = Encoder(vocab_size, hidden_dim)
        self.sem_p = nn.Linear(hidden_dim, half)
        self.epi_p = nn.Linear(hidden_dim, half)
        self.out   = nn.Linear(half * 2, vocab_size)

    def forward(self, seq):
        M_s, M_e, q = _write_matrices(self.enc, self.sem_p, self.epi_p, seq)
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


class GatedRead(nn.Module):
    """Softmax gate over [sem_out, epi_out] weighted by query."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc        = Encoder(vocab_size, hidden_dim)
        self.sem_p      = nn.Linear(hidden_dim, half)
        self.epi_p      = nn.Linear(hidden_dim, half)
        self.gate_proj  = nn.Linear(hidden_dim, 2)
        self.proj_s     = nn.Linear(half, hidden_dim)
        self.proj_e     = nn.Linear(half, hidden_dim)
        self.out        = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        M_s, M_e, q = _write_matrices(self.enc, self.sem_p, self.epi_p, seq)
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        gate = F.softmax(self.gate_proj(q), -1)   # (B, 2)
        ctx  = gate[:, 0:1] * self.proj_s(cs) + gate[:, 1:2] * self.proj_e(ce)
        return self.out(ctx)


class SumRead(nn.Module):
    """Project cs and ce to same dim, add, then output."""
    def __init__(self, vocab_size=64, hidden_dim=64):
        super().__init__()
        half = hidden_dim // 2
        self.enc   = Encoder(vocab_size, hidden_dim)
        self.sem_p = nn.Linear(hidden_dim, half)
        self.epi_p = nn.Linear(hidden_dim, half)
        self.proj_s = nn.Linear(half, hidden_dim)
        self.proj_e = nn.Linear(half, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        M_s, M_e, q = _write_matrices(self.enc, self.sem_p, self.epi_p, seq)
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        ctx = self.proj_s(cs) + self.proj_e(ce)
        return self.out(ctx)


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


class Exp424ReadCombinationMethods(Experiment):
    experiment_id = "exp_42_4"
    hypothesis = (
        "A learned attention gate over [sem_out, epi_out] (softmax weighting) "
        "outperforms simple concatenation by >5%, showing that dynamic read "
        "combination extracts more information than fixed 50/50 concatenation."
    )

    def run(self) -> ExperimentResult:
        print("Training ConcatRead ...")
        acc_concat = train_eval(
            ConcatRead(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_concat={acc_concat:.4f}")

        print("Training GatedRead ...")
        acc_gated = train_eval(
            GatedRead(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_gated={acc_gated:.4f}")

        print("Training SumRead ...")
        acc_sum = train_eval(
            SumRead(VOCAB_SIZE, HIDDEN_DIM),
            steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE,
        )
        print(f"  acc_sum={acc_sum:.4f}")

        gap_gated_over_concat = round(float(acc_gated) - float(acc_concat), 4)

        metrics = dict(
            acc_concat=round(float(acc_concat), 4),
            acc_gated=round(float(acc_gated), 4),
            acc_sum=round(float(acc_sum), 4),
            gap_gated_over_concat=gap_gated_over_concat,
        )

        if acc_gated > acc_concat + 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Gated read outperforms concat: gated={acc_gated:.3f}, "
                f"concat={acc_concat:.3f}, gap={gap_gated_over_concat:.3f}>0.05."
            )
        elif acc_concat >= acc_gated - 0.02:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Concat is competitive with gated: concat={acc_concat:.3f}, "
                f"gated={acc_gated:.3f}, gap={gap_gated_over_concat:.3f} (within 0.02)."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Gated has some advantage but below threshold: gated={acc_gated:.3f}, "
                f"concat={acc_concat:.3f}, gap={gap_gated_over_concat:.3f}."
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
    Exp424ReadCombinationMethods().execute()
