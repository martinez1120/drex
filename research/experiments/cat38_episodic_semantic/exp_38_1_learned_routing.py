"""
Experiment 38.1 — Learned Routing Improves Episodic/Semantic Split

Hypothesis: A soft-routing mechanism — a trainable logistic gate g_t ∈ (0,1) per
timestep that blends episodic and semantic write contributions — outperforms the
fixed 50/50 split of exp_36_3 by >5% accuracy.

Background: exp_36_3 found that FIXED split memory outperforms unified memory by
+5.9%. The next question is: can a LEARNED router decide per-item which memory
(episodic or semantic) should store each association?

FixedSplit (baseline): sum_proj → 50% semantic, 50% episodic dims
LearnedSplit: router MLP predicts g_t; semantic gets g_t content, episodic gets (1-g_t)
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


class FixedSplitModel(nn.Module):
    """Replicates exp_36_3 split memory (fixed 50/50 semantic/episodic)."""
    def __init__(self):
        super().__init__()
        self.enc  = Encoder()
        self.sem_p = nn.Linear(HIDDEN_DIM, HALF)
        self.epi_p = nn.Linear(HIDDEN_DIM, HALF)
        self.out   = nn.Linear(HALF * 2, VOCAB_SIZE)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape
        M_s = torch.zeros(B, HALF, HALF, device=h.device)
        M_e = torch.zeros(B, HALF, HALF, device=h.device)
        for t in range(L - 1):
            ks = self.sem_p(h[:, t, :]); ke = self.epi_p(h[:, t, :])
            # Semantic: pure delta rule
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
            # Episodic: recency-weighted
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_e = M_e + ((t+1)/L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


class LearnedRouterModel(nn.Module):
    """Soft router: g_t = σ(router(h_t)) blends semantic/episodic writes."""
    def __init__(self):
        super().__init__()
        self.enc     = Encoder()
        self.sem_p   = nn.Linear(HIDDEN_DIM, HALF)
        self.epi_p   = nn.Linear(HIDDEN_DIM, HALF)
        self.router  = nn.Linear(HIDDEN_DIM, 1)   # outputs g ∈ (0,1)
        self.out     = nn.Linear(HALF * 2, VOCAB_SIZE)

    def forward(self, seq):
        h = self.enc(seq); B, L, H = h.shape
        M_s = torch.zeros(B, HALF, HALF, device=h.device)
        M_e = torch.zeros(B, HALF, HALF, device=h.device)
        for t in range(L - 1):
            ht  = h[:, t, :]
            g   = torch.sigmoid(self.router(ht))     # (B,1): 1=semantic, 0=episodic
            ks  = self.sem_p(ht); ke = self.epi_p(ht)

            # Semantic write (weighted by g)
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + g.unsqueeze(-1) * torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))

            # Episodic write (weighted by 1-g, recency)
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
            recency = (t + 1) / L
            M_e = M_e + (1-g).unsqueeze(-1) * recency * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))

        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


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


class Exp381LearnedRouting(Experiment):
    experiment_id = "exp_38_1"
    hypothesis = (
        "Learned soft-routing on episodic/semantic split outperforms fixed 50/50 split "
        "by >5% accuracy; router learns a non-trivial allocation (not always 0.5)."
    )

    def run(self) -> ExperimentResult:
        print("Training FixedSplitModel...")
        acc_fixed  = train_eval(FixedSplitModel)
        print(f"  acc_fixed={acc_fixed:.4f}")
        print("Training LearnedRouterModel...")
        acc_router = train_eval(LearnedRouterModel)
        print(f"  acc_router={acc_router:.4f}")

        gap = acc_router - acc_fixed
        metrics = dict(
            acc_fixed=round(acc_fixed, 4),
            acc_router=round(acc_router, 4),
            gap_router_minus_fixed=round(gap, 4),
        )

        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Learned routing wins: router={acc_router:.3f}, "
                     f"fixed={acc_fixed:.3f}, gap={gap:.3f}>0.05.")
        elif gap < -0.03:
            outcome = OUTCOME_REFUTED
            notes = (f"Fixed split wins: fixed={acc_fixed:.3f}, "
                     f"router={acc_router:.3f}, gap={gap:.3f}<-0.03.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Near-equal: fixed={acc_fixed:.3f}, router={acc_router:.3f}, "
                     f"gap={gap:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS))


if __name__ == "__main__":
    Exp381LearnedRouting().execute()
