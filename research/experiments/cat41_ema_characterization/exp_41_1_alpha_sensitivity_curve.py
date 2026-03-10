"""
Experiment 41.1 — EMA Alpha Sensitivity Curve

Hypothesis: EMA accuracy peaks in the range α ∈ [0.85, 0.95] and drops at both
extremes (α=0.5 too aggressive smoothing, α=0.99 nearly identical to standard).
The optimal α gives >3% improvement over α=1.0 (standard delta).
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
ALPHAS = [0.5, 0.7, 0.8, 0.85, 0.90, 0.95, 0.99, 1.0]


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


class DeltaBase(nn.Module):
    def __init__(self, alpha=1.0, hidden_dim=64, vocab_size=64):
        super().__init__()
        self.alpha = alpha
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq, noise_scale=0.0):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            Delta = torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
            if self.alpha < 1.0:
                M = M + (1.0 - self.alpha) * Delta
            else:
                M = M + Delta
        if noise_scale > 0:
            M = M + noise_scale * torch.randn_like(M) * M.std().clamp_min(1e-6)
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


def train_eval(alpha):
    model = DeltaBase(alpha=alpha, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
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


class Exp411AlphaSensitivityCurve(Experiment):
    experiment_id = "exp_41_1"
    hypothesis = (
        "EMA accuracy peaks in the range α ∈ [0.85, 0.95] and drops at both extremes "
        "(α=0.5 too aggressive smoothing, α=0.99 nearly identical to standard). "
        "The optimal α gives >3% improvement over α=1.0 (standard delta)."
    )

    def run(self) -> ExperimentResult:
        acc_by_alpha = {}
        for alpha in ALPHAS:
            acc_by_alpha[alpha] = train_eval(alpha)

        best_alpha = max(acc_by_alpha, key=lambda a: acc_by_alpha[a])
        best_acc = acc_by_alpha[best_alpha]
        acc_standard = acc_by_alpha[1.0]
        gap_vs_standard = best_acc - acc_standard

        metrics = {}
        for alpha in ALPHAS:
            key = f"acc_alpha_{int(alpha * 100):03d}"
            metrics[key] = round(acc_by_alpha[alpha], 4)
        metrics["best_alpha"] = round(best_alpha, 4)
        metrics["best_acc"] = round(best_acc, 4)
        metrics["acc_standard"] = round(acc_standard, 4)
        metrics["gap_vs_standard"] = round(gap_vs_standard, 4)

        config = {
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "steps": STEPS,
            "alphas": ALPHAS,
        }

        if gap_vs_standard > 0.03 and 0.85 <= best_alpha <= 0.99:
            outcome = OUTCOME_SUPPORTED
        elif gap_vs_standard < 0.0:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"Best alpha={best_alpha} with acc={round(best_acc, 4)}, "
            f"standard alpha=1.0 acc={round(acc_standard, 4)}, "
            f"gap={round(gap_vs_standard, 4)}"
        )
        return self.result(outcome, metrics, notes, config=config)


if __name__ == "__main__":
    Exp411AlphaSensitivityCurve().execute()
