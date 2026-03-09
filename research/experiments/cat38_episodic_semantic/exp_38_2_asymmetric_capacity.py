"""
Experiment 38.2 — Asymmetric Episodic/Semantic Capacity Allocation

Hypothesis: Allocating different hidden-dim proportions to episodic vs semantic memory
reveals that the optimal split is 25% episodic / 75% semantic (EPI_FRAC=0.25),
exceeding 50/50 by >3%. The task is primarily semantic (content-based KV recall),
so semantic capacity should dominate.

Rationale: exp_36_3's 50/50 split was arbitrary. Semantic recall is the primary task
(which value is associated with this key?) while episodic only provides auxiliary
temporal signal. Asymmetric allocation should match the task structure.
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
# Fractions of HIDDEN_DIM allocated to episodic; semantic gets the rest.
EPI_FRACS  = [0.25, 0.50, 0.75]   # 25/75, 50/50, 75/25 splits


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


class AsymmetricSplitModel(nn.Module):
    def __init__(self, epi_frac=0.50):
        super().__init__()
        self.epi_dim = max(4, round(HIDDEN_DIM * epi_frac))
        self.sem_dim = HIDDEN_DIM - self.epi_dim
        self.embed   = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff      = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm    = nn.LayerNorm(HIDDEN_DIM)
        self.sem_p   = nn.Linear(HIDDEN_DIM, self.sem_dim)
        self.epi_p   = nn.Linear(HIDDEN_DIM, self.epi_dim)
        self.out     = nn.Linear(self.sem_dim + self.epi_dim, VOCAB_SIZE)

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M_s = torch.zeros(B, self.sem_dim, self.sem_dim, device=h.device)
        M_e = torch.zeros(B, self.epi_dim, self.epi_dim, device=h.device)
        for t in range(L - 1):
            ks = self.sem_p(h[:, t, :]); ke = self.epi_p(h[:, t, :])
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            dvs = ks - vps / (ks.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dve = ke - vpe / (ke.pow(2).sum(-1, keepdim=True) + 1e-6)
            M_e = M_e + ((t+1)/L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(torch.cat([cs, ce], -1))


def train_eval(epi_frac):
    model = AsymmetricSplitModel(epi_frac=epi_frac)
    opt   = Adam(model.parameters(), lr=3e-4)
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


class Exp382AsymmetricCapacity(Experiment):
    experiment_id = "exp_38_2"
    hypothesis = (
        "EPI_FRAC=0.25 (25% episodic, 75% semantic) outperforms 50/50 by >3%, "
        "indicating semantic capacity dominates for content-association tasks."
    )

    def run(self) -> ExperimentResult:
        accs: dict[str, float] = {}
        for frac in EPI_FRACS:
            name = f"epi{int(frac*100):03d}"
            print(f"Training {name} (epi_frac={frac})...")
            a = train_eval(frac)
            accs[name] = round(a, 4)
            print(f"  acc={a:.4f}")

        acc_50  = accs["epi050"]
        best_k  = max(accs, key=accs.__getitem__)
        best_a  = accs[best_k]
        gap_vs_50 = best_a - acc_50

        metrics = dict(**accs,
                       best_split=best_k,
                       best_acc=round(best_a, 4),
                       gap_best_vs_50=round(gap_vs_50, 4),
                       acc_25_epi=accs.get("epi025", 0),
                       acc_75_epi=accs.get("epi075", 0))

        if accs.get("epi025", 0) > acc_50 + 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (f"25% episodic wins: epi025={accs['epi025']:.3f}, "
                     f"epi050={acc_50:.3f}, gap={accs['epi025']-acc_50:.3f}>0.03.")
        elif accs.get("epi075", 0) > acc_50 + 0.03:
            outcome = OUTCOME_REFUTED
            notes = (f"Refuted opposite: 75% episodic wins: epi075={accs['epi075']:.3f}. "
                     f"Episodic capacity dominates unexpectedly.")
        elif gap_vs_50 > 0.03:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Some asymmetry helps: best={best_k}({best_a:.3f}) "
                     f"vs 50/50({acc_50:.3f}), gap={gap_vs_50:.3f}. "
                     f"But predicted direction wrong.")
        else:
            outcome = OUTCOME_REFUTED
            notes = (f"50/50 optimal: best={best_k}({best_a:.3f}), "
                     f"50/50={acc_50:.3f}, gap={gap_vs_50:.3f} ≤ 0.03.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, epi_fracs=EPI_FRACS))


if __name__ == "__main__":
    Exp382AsymmetricCapacity().execute()
