"""
Experiment 41.5 — Per-Position Learned Alpha vs Global Alpha

Hypothesis: A per-position learned alpha (one scalar per sequence position,
initialized at 0.95 and optimized via gradient) provides no significant improvement
over global alpha (< 2% gap), confirming that global alpha is sufficient and
position-specific tuning is unnecessary.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import math
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
GLOBAL_ALPHA = 0.95


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


class GlobalEMADelta(nn.Module):
    def __init__(self, alpha=GLOBAL_ALPHA, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.alpha = alpha
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            Delta = torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
            M = M + (1.0 - self.alpha) * Delta
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


class PerPositionEMADelta(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)
        # Initialize logits so that sigmoid(logit) * 0.49 + 0.50 = 0.95
        # sigmoid(logit) = (0.95 - 0.50) / 0.49 = 0.45/0.49 ~ 0.9184
        # logit = log(0.9184 / (1 - 0.9184)) ~ log(11.27) ~ 2.422
        init_val = math.log(0.9184 / (1.0 - 0.9184))
        self.alpha_logits = nn.Parameter(torch.full((seq_len,), init_val))

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        alphas = torch.sigmoid(self.alpha_logits) * 0.49 + 0.50  # shape (seq_len,)
        for t in range(L - 1):
            alpha_t = alphas[t]
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            Delta = torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
            M = M + (1.0 - alpha_t) * Delta
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


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


class Exp415PerPositionAlpha(Experiment):
    experiment_id = "exp_41_5"
    hypothesis = (
        "A per-position learned alpha (one scalar per sequence position, initialized at 0.95 "
        "and optimized via gradient) provides no significant improvement over global alpha "
        "(< 2% gap), confirming that global alpha is sufficient and position-specific tuning "
        "is unnecessary."
    )

    def run(self) -> ExperimentResult:
        acc_global  = train_and_eval(GlobalEMADelta())
        acc_per_pos = train_and_eval(PerPositionEMADelta(seq_len=SEQ_LEN))

        gap = acc_per_pos - acc_global

        metrics = {
            "acc_global":               round(acc_global, 4),
            "acc_per_pos":              round(acc_per_pos, 4),
            "gap_per_pos_minus_global": round(gap, 4),
        }

        config = {
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "steps": STEPS,
            "global_alpha": GLOBAL_ALPHA,
            "seq_len": SEQ_LEN,
        }

        if abs(gap) < 0.02:
            outcome = OUTCOME_SUPPORTED   # null result confirmed
        elif gap > 0.05:
            outcome = OUTCOME_REFUTED     # per-position is significantly better
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"acc_global={round(acc_global, 4)}, acc_per_pos={round(acc_per_pos, 4)}, "
            f"gap={round(gap, 4)}"
        )
        return self.result(outcome, metrics, notes, config=config)


if __name__ == "__main__":
    Exp415PerPositionAlpha().execute()
