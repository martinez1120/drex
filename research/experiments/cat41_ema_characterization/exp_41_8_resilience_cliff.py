"""
Experiment 41.8 — EMA Pushes the Resilience Cliff to Higher Noise

Hypothesis: Standard delta has an accuracy cliff at σ≈0.10 (rapid drop to near-chance)
while EMA (α=0.95) pushes the cliff to σ≥0.30. Cliff defined as first σ where
accuracy falls below 50% of clean accuracy.
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
NOISE_SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


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


def find_cliff(accs_by_sigma, clean_acc):
    for sigma, acc in sorted(accs_by_sigma.items()):
        if sigma > 0 and acc < 0.5 * clean_acc:
            return sigma
    return float('inf')


def train_model(alpha):
    model = DeltaBase(alpha=alpha, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_model_at_noise(model, noise_scale):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
            preds = model(seq, noise_scale=noise_scale).argmax(dim=-1)
            correct += (preds == tgt).sum().item(); total += BATCH
    return correct / total


class Exp418ResilienceCliff(Experiment):
    experiment_id = "exp_41_8"
    hypothesis = (
        "Standard delta has an accuracy cliff at σ≈0.10 (rapid drop to near-chance) "
        "while EMA (α=0.95) pushes the cliff to σ≥0.30. Cliff defined as first σ where "
        "accuracy falls below 50% of clean accuracy."
    )

    def run(self) -> ExperimentResult:
        std_model = train_model(alpha=1.0)
        ema_model = train_model(alpha=0.95)

        acc_std_by_sigma = {}
        acc_ema_by_sigma = {}
        for sigma in NOISE_SIGMAS:
            acc_std_by_sigma[sigma] = eval_model_at_noise(std_model, sigma)
            acc_ema_by_sigma[sigma] = eval_model_at_noise(ema_model, sigma)

        cliff_std = find_cliff(acc_std_by_sigma, acc_std_by_sigma[0.0])
        cliff_ema = find_cliff(acc_ema_by_sigma, acc_ema_by_sigma[0.0])
        cliff_ratio = cliff_ema / max(cliff_std, 1e-6)

        metrics = {}
        for sigma in NOISE_SIGMAS:
            key_std = f"acc_std_s{int(sigma * 100):03d}"
            key_ema = f"acc_ema_s{int(sigma * 100):03d}"
            metrics[key_std] = round(acc_std_by_sigma[sigma], 4)
            metrics[key_ema] = round(acc_ema_by_sigma[sigma], 4)
        metrics["cliff_std"]   = round(cliff_std, 4) if cliff_std != float('inf') else -1.0
        metrics["cliff_ema"]   = round(cliff_ema, 4) if cliff_ema != float('inf') else -1.0
        metrics["cliff_ratio"] = round(cliff_ratio, 4)

        config = {
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "steps": STEPS,
            "noise_sigmas": NOISE_SIGMAS,
        }

        if cliff_ema >= 0.30 and cliff_std <= 0.15:
            outcome = OUTCOME_SUPPORTED
        elif cliff_ema <= cliff_std:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"cliff_std={cliff_std}, cliff_ema={cliff_ema}, cliff_ratio={round(cliff_ratio, 4)}, "
            f"clean acc: std={round(acc_std_by_sigma[0.0], 4)}, ema={round(acc_ema_by_sigma[0.0], 4)}"
        )
        return self.result(outcome, metrics, notes, config=config)


if __name__ == "__main__":
    Exp418ResilienceCliff().execute()
