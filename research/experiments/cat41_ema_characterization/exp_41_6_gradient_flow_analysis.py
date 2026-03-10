"""
Experiment 41.6 — Gradient Flow Through EMA Updates

Hypothesis: EMA smoothing (α=0.95) reduces gradient variance at the embedding layer
by >30% compared to standard delta, providing more stable training
(lower embedding gradient std over the final 200 training steps).
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


def train_collect_grads(alpha):
    model = DeltaBase(alpha=alpha, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    opt = Adam(model.parameters(), lr=3e-4)
    grad_norms = []
    model.train()
    for step in range(STEPS):
        seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward()
        if model.embed.weight.grad is not None:
            gn = model.embed.weight.grad.norm().item()
        else:
            gn = 0.0
        if step >= STEPS - 200:
            grad_norms.append(gn)
        opt.step()
    return model, grad_norms


def eval_model(model):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
            preds = model(seq).argmax(dim=-1)
            correct += (preds == tgt).sum().item(); total += BATCH
    return correct / total


class Exp416GradientFlowAnalysis(Experiment):
    experiment_id = "exp_41_6"
    hypothesis = (
        "EMA smoothing (α=0.95) reduces gradient variance at the embedding layer by >30% "
        "compared to standard delta, providing more stable training "
        "(lower embedding gradient std over the final 200 training steps)."
    )

    def run(self) -> ExperimentResult:
        std_model, grad_norms_std = train_collect_grads(alpha=1.0)
        ema_model, grad_norms_ema = train_collect_grads(alpha=0.95)

        acc_std = eval_model(std_model)
        acc_ema = eval_model(ema_model)

        import statistics
        mean_grad_std = statistics.mean(grad_norms_std) if grad_norms_std else 0.0
        mean_grad_ema = statistics.mean(grad_norms_ema) if grad_norms_ema else 0.0
        var_grad_std  = statistics.variance(grad_norms_std) if len(grad_norms_std) > 1 else 0.0
        var_grad_ema  = statistics.variance(grad_norms_ema) if len(grad_norms_ema) > 1 else 0.0
        std_norms_std = statistics.stdev(grad_norms_std) if len(grad_norms_std) > 1 else 0.0
        std_norms_ema = statistics.stdev(grad_norms_ema) if len(grad_norms_ema) > 1 else 0.0

        std_ratio = std_norms_ema / max(std_norms_std, 1e-8)

        metrics = {
            "mean_grad_std": round(mean_grad_std, 4),
            "mean_grad_ema": round(mean_grad_ema, 4),
            "var_grad_std":  round(var_grad_std, 4),
            "var_grad_ema":  round(var_grad_ema, 4),
            "std_ratio":     round(std_ratio, 4),
            "acc_std":       round(acc_std, 4),
            "acc_ema":       round(acc_ema, 4),
        }

        config = {
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "steps": STEPS,
            "grad_window": 200,
        }

        if std_ratio < 0.70:
            outcome = OUTCOME_SUPPORTED
        elif std_ratio > 1.0:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"std_ratio={round(std_ratio, 4)}, std_norms_std={round(std_norms_std, 4)}, "
            f"std_norms_ema={round(std_norms_ema, 4)}, acc_std={round(acc_std, 4)}, "
            f"acc_ema={round(acc_ema, 4)}"
        )
        return self.result(outcome, metrics, notes, config=config)


if __name__ == "__main__":
    Exp416GradientFlowAnalysis().execute()
