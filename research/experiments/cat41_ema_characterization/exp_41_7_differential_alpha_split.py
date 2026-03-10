"""
Experiment 41.7 — Differential Alpha for Episodic vs Semantic Matrices

Hypothesis: The optimal EMA alpha differs between episodic and semantic matrices:
the best (alpha_sem, alpha_epi) pair outperforms any shared alpha by >3%,
with alpha_epi > alpha_sem (episodic needs more smoothing due to recency weighting).
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
SEQ_LEN = 24
NUM_PAIRS = 5
STEPS = 600
BATCH = 32
HALF = 32
ALPHA_GRID = [0.70, 0.85, 0.95, 1.0]


def make_batch(batch_size, seq_len=24, num_pairs=5, vocab_size=64):
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


class DiffAlphaSplitModel(nn.Module):
    def __init__(self, alpha_sem=1.0, alpha_epi=1.0,
                 hidden_dim=HIDDEN_DIM, half=HALF, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.alpha_sem = alpha_sem
        self.alpha_epi = alpha_epi
        self.half = half
        self.embed  = nn.Embedding(vocab_size, hidden_dim)
        self.ff     = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                    nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm   = nn.LayerNorm(hidden_dim)
        self.proj_s = nn.Linear(hidden_dim, half)
        self.proj_e = nn.Linear(hidden_dim, half)
        self.rp     = nn.Linear(half * 2, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M_s = torch.zeros(B, self.half, self.half, device=h.device)
        M_e = torch.zeros(B, self.half, self.half, device=h.device)
        for t in range(L - 1):
            hs = self.proj_s(h[:, t, :]); he = self.proj_e(h[:, t, :])
            ks = F.normalize(hs, dim=-1); ke = F.normalize(he, dim=-1)
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dvs = hs - vps; dve = he - vpe
            delta_s = torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
            delta_e = ((t + 1) / L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
            if self.alpha_sem < 1.0:
                M_s = M_s + (1.0 - self.alpha_sem) * delta_s
            else:
                M_s = M_s + delta_s
            if self.alpha_epi < 1.0:
                M_e = M_e + (1.0 - self.alpha_epi) * delta_e
            else:
                M_e = M_e + delta_e
        q = h[:, -1, :]
        qs = self.proj_s(q); qe = self.proj_e(q)
        rs = torch.bmm(M_s, F.normalize(qs, dim=-1).unsqueeze(-1)).squeeze(-1)
        re = torch.bmm(M_e, F.normalize(qe, dim=-1).unsqueeze(-1)).squeeze(-1)
        r = torch.cat([rs, re], dim=-1)
        return self.out(self.rp(r))


def train_and_eval(model):
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
            preds = model(seq).argmax(dim=-1)
            correct += (preds == tgt).sum().item(); total += BATCH
    return correct / total


class Exp417DifferentialAlphaSplit(Experiment):
    experiment_id = "exp_41_7"
    hypothesis = (
        "The optimal EMA alpha differs between episodic and semantic matrices: "
        "the best (alpha_sem, alpha_epi) pair outperforms any shared alpha by >3%, "
        "with alpha_epi > alpha_sem (episodic needs more smoothing due to recency weighting)."
    )

    def run(self) -> ExperimentResult:
        acc_by_combo = {}
        for alpha_sem in ALPHA_GRID:
            for alpha_epi in ALPHA_GRID:
                model = DiffAlphaSplitModel(alpha_sem=alpha_sem, alpha_epi=alpha_epi)
                acc = train_and_eval(model)
                acc_by_combo[(alpha_sem, alpha_epi)] = acc

        # Train shared alpha=0.95 baseline
        shared_model = DiffAlphaSplitModel(alpha_sem=0.95, alpha_epi=0.95)
        acc_shared_095 = train_and_eval(shared_model)

        best_combo = max(acc_by_combo, key=lambda c: acc_by_combo[c])
        best_acc = acc_by_combo[best_combo]
        best_alpha_sem, best_alpha_epi = best_combo
        gap_diff_over_shared = best_acc - acc_shared_095

        metrics = {}
        for (a_sem, a_epi), acc in acc_by_combo.items():
            key = f"acc_as{int(a_sem * 100):03d}_ae{int(a_epi * 100):03d}"
            metrics[key] = round(acc, 4)
        metrics["best_alpha_sem"]       = round(best_alpha_sem, 4)
        metrics["best_alpha_epi"]       = round(best_alpha_epi, 4)
        metrics["gap_diff_over_shared"] = round(gap_diff_over_shared, 4)
        metrics["acc_shared_095"]       = round(acc_shared_095, 4)

        config = {
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "half": HALF,
            "steps": STEPS,
            "alpha_grid": ALPHA_GRID,
        }

        if gap_diff_over_shared > 0.03 and best_alpha_epi > best_alpha_sem:
            outcome = OUTCOME_SUPPORTED
        elif gap_diff_over_shared < 0.0:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"best_combo=({best_alpha_sem}, {best_alpha_epi}), best_acc={round(best_acc, 4)}, "
            f"acc_shared_095={round(acc_shared_095, 4)}, gap={round(gap_diff_over_shared, 4)}"
        )
        return self.result(outcome, metrics, notes, config=config)


if __name__ == "__main__":
    Exp417DifferentialAlphaSplit().execute()
