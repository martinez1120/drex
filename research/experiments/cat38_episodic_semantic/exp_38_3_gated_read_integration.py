"""
Experiment 38.3 — Gated Cross-Module Integration vs Concatenation

Hypothesis: Using a learned attention gate to combine episodic and semantic read-outs
(per-query weighting via softmax([w_sem, w_epi])) outperforms simple concatenation
by >5%, showing the model can selectively attend to the more relevant memory module.

Background: exp_36_3 concatenates M_sem and M_epi read-outs. This fixes the blend at
50/50 per dimension at query time. A gated combination lets the query decide per-item
which memory module is more informative, potentially better for mixed-type queries.
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
HALF       = HIDDEN_DIM // 2


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
        for i in range(NUM_PAIRS):
            pos = i * 4
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]
        for p in range(NUM_PAIRS * 4, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


def _write_memories(M_s, M_e, h, t, L, sem_p, epi_p):
    ks = sem_p(h[:, t, :]); ke = epi_p(h[:, t, :])
    vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
    dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
    M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
    vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
    dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
    M_e = M_e + ((t+1)/L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
    return M_s, M_e


class ConcatReadModel(nn.Module):
    """Baseline: concatenate sem and epi read-outs (replicates exp_36_3)."""
    def __init__(self):
        super().__init__()
        self.enc   = Encoder()
        self.sem_p = nn.Linear(HIDDEN_DIM, HALF)
        self.epi_p = nn.Linear(HIDDEN_DIM, HALF)
        self.out   = nn.Linear(HALF * 2, VOCAB_SIZE)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape
        M_s = torch.zeros(B, HALF, HALF, device=h.device)
        M_e = torch.zeros(B, HALF, HALF, device=h.device)
        for t in range(L - 1):
            M_s, M_e = _write_memories(M_s, M_e, h, t, L, self.sem_p, self.epi_p)
        q = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


class GatedReadModel(nn.Module):
    """Gated read: softmax gate over [sem_out, epi_out] normalised to HIDDEN_DIM."""
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.sem_p  = nn.Linear(HIDDEN_DIM, HALF)
        self.epi_p  = nn.Linear(HIDDEN_DIM, HALF)
        self.gate   = nn.Linear(HIDDEN_DIM, 2)   # query → (w_sem, w_epi)
        self.proj_s = nn.Linear(HALF, HIDDEN_DIM)
        self.proj_e = nn.Linear(HALF, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape
        M_s = torch.zeros(B, HALF, HALF, device=h.device)
        M_e = torch.zeros(B, HALF, HALF, device=h.device)
        for t in range(L - 1):
            M_s, M_e = _write_memories(M_s, M_e, h, t, L, self.sem_p, self.epi_p)
        q  = h[:, -1, :]
        cs = self.proj_s(torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1))
        ce = self.proj_e(torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1))
        weights = F.softmax(self.gate(q), -1)        # (B, 2)
        ctx = weights[:, 0:1] * cs + weights[:, 1:2] * ce   # (B, H)
        return self.out(ctx)


def train_eval(model_class):
    model = model_class(); opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(BATCH)
            ok  += (model(seq).argmax(-1) == tgt).sum().item(); tot += tgt.size(0)
    return ok / tot


class Exp383GatedReadIntegration(Experiment):
    experiment_id = "exp_38_3"
    hypothesis = (
        "Gated read (learned softmax over [M_sem, M_epi] outputs) outperforms "
        "simple concatenation by >5%."
    )

    def run(self) -> ExperimentResult:
        print("Training ConcatReadModel...")
        acc_concat = train_eval(ConcatReadModel)
        print(f"  acc_concat={acc_concat:.4f}")
        print("Training GatedReadModel...")
        acc_gated  = train_eval(GatedReadModel)
        print(f"  acc_gated={acc_gated:.4f}")

        gap = acc_gated - acc_concat
        metrics = dict(
            acc_concat=round(acc_concat, 4),
            acc_gated=round(acc_gated, 4),
            gap_gated_minus_concat=round(gap, 4),
        )

        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Gated read wins: gated={acc_gated:.3f}, "
                     f"concat={acc_concat:.3f}, gap={gap:.3f}>0.05.")
        elif gap < -0.03:
            outcome = OUTCOME_REFUTED
            notes = (f"Concat wins: concat={acc_concat:.3f}, "
                     f"gated={acc_gated:.3f}, gap={gap:.3f}<-0.03.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Similar: concat={acc_concat:.3f}, gated={acc_gated:.3f}, "
                     f"gap={gap:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS))


if __name__ == "__main__":
    Exp383GatedReadIntegration().execute()
