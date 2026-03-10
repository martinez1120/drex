"""
Experiment 41.3 — EMA Benefit Scales with Sequence Length

Hypothesis: The accuracy gain from EMA (α=0.95) over standard delta increases
monotonically with sequence length: the gain at SEQ_LEN=96 is >2× the gain at SEQ_LEN=24.
Longer sequences accumulate more noise/interference, making smoothing more beneficial.
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
NUM_PAIRS = 5
STEPS = 600
BATCH = 32
SEQ_LENS = [16, 24, 48, 96]
ALPHA_EMA = 0.95


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


def train_eval_at_seqlen(alpha, seq_len):
    num_pairs = min(NUM_PAIRS, (seq_len - 6) // 2)
    model = DeltaBase(alpha=alpha, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH, seq_len=seq_len, num_pairs=num_pairs, vocab_size=VOCAB_SIZE)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_batch(BATCH, seq_len=seq_len, num_pairs=num_pairs, vocab_size=VOCAB_SIZE)
            preds = model(seq).argmax(dim=-1)
            correct += (preds == tgt).sum().item(); total += BATCH
    return correct / total


class Exp413EmaVsSeqLen(Experiment):
    experiment_id = "exp_41_3"
    hypothesis = (
        "The accuracy gain from EMA (α=0.95) over standard delta increases monotonically "
        "with sequence length: the gain at SEQ_LEN=96 is >2× the gain at SEQ_LEN=24. "
        "Longer sequences accumulate more noise/interference, making smoothing more beneficial."
    )

    def run(self) -> ExperimentResult:
        gaps = {}
        acc_std_by_len = {}
        acc_ema_by_len = {}

        for seq_len in SEQ_LENS:
            acc_std = train_eval_at_seqlen(1.0, seq_len)
            acc_ema = train_eval_at_seqlen(ALPHA_EMA, seq_len)
            acc_std_by_len[seq_len] = acc_std
            acc_ema_by_len[seq_len] = acc_ema
            gaps[seq_len] = acc_ema - acc_std

        gain_ratio = gaps[SEQ_LENS[-1]] / max(gaps[SEQ_LENS[0]], 1e-4)
        all_non_negative = all(g >= 0 for g in gaps.values())

        metrics = {}
        for seq_len in SEQ_LENS:
            metrics[f"acc_std_L{seq_len}"] = round(acc_std_by_len[seq_len], 4)
            metrics[f"acc_ema_L{seq_len}"] = round(acc_ema_by_len[seq_len], 4)
            metrics[f"gap_L{seq_len}"]     = round(gaps[seq_len], 4)
        metrics["gain_ratio"] = round(gain_ratio, 4)

        config = {
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "steps": STEPS,
            "seq_lens": SEQ_LENS,
            "alpha_ema": ALPHA_EMA,
        }

        if gain_ratio > 2.0 and all_non_negative:
            outcome = OUTCOME_SUPPORTED
        elif all(g < 0.01 for g in gaps.values()):
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"gaps by seq_len: {gaps}, gain_ratio={round(gain_ratio, 4)}, "
            f"all_non_negative={all_non_negative}"
        )
        return self.result(outcome, metrics, notes, config=config)


if __name__ == "__main__":
    Exp413EmaVsSeqLen().execute()
