"""
Experiment 37.2 — Row-Normalized Memory Matrix Resists Noise Collapse

Hypothesis: Normalizing the rows of M to unit L2 norm after each delta-rule write
bounds the maximum M entry magnitude; at σ_test=0.10 the accuracy-retention ratio
is ≥ 0.50, vs 0.08 for the unnormalized baseline.

Rationale: Unnormalized M grows in magnitude as it accumulates associations.
Large-magnitude M amplifies additive noise (M + σ*randn*M.std()), causing
catastrophic collapse. Row-normalisation caps M.std(), making noise proportionally
smaller and retrieval more stable.
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
SEQ_LEN    = 24
NUM_PAIRS  = 5
STEPS      = 600
BATCH      = 32
EVAL_SIGMAS = [0.0, 0.03, 0.05, 0.10, 0.20]


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,)); pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


def _encoder(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM):
    embed = nn.Embedding(vocab, hidden)
    ff    = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(),
                           nn.Linear(hidden * 2, hidden))
    norm  = nn.LayerNorm(hidden)
    return embed, ff, norm


class StandardDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed, self.ff, self.norm = _encoder()
        self.rp  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq, noise_scale=0.0):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            M = M + torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
        if noise_scale > 0:
            M = M + noise_scale * torch.randn_like(M) * M.std().clamp_min(1e-6)
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


class RowNormDelta(nn.Module):
    """After each delta-rule write, row-normalize M so ||M[i,:]||_2 ≤ 1."""
    def __init__(self):
        super().__init__()
        self.embed, self.ff, self.norm = _encoder()
        self.rp  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq, noise_scale=0.0):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            M = M + torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
            # Row-normalize: clip each row to unit norm
            row_norms = M.norm(dim=-1, keepdim=True).clamp_min(1.0)
            M = M / row_norms
        if noise_scale > 0:
            M = M + noise_scale * torch.randn_like(M) * M.std().clamp_min(1e-6)
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


def train_model(model_class):
    model = model_class()
    opt   = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    return model


def eval_sigma(model, sigma, n=50):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(n):
            seq, tgt = make_batch(BATCH)
            ok  += (model(seq, noise_scale=sigma).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    model.train(); return ok / tot


class Exp372RowNormRobustness(Experiment):
    experiment_id = "exp_37_2"
    hypothesis = (
        "Row-normalizing M after each write bounds M magnitude; "
        "acc_ratio at σ=0.10 ≥ 0.50 (vs 0.08 for standard delta)."
    )

    def run(self) -> ExperimentResult:
        print("Training StandardDelta...")
        m_std  = train_model(StandardDelta)
        print("Training RowNormDelta...")
        m_norm = train_model(RowNormDelta)

        metrics: dict[str, float] = {}
        for σ in EVAL_SIGMAS:
            a_std  = eval_sigma(m_std,  σ)
            a_norm = eval_sigma(m_norm, σ)
            key = f"s{int(σ*100):03d}"
            metrics[f"std_{key}"]  = round(a_std,  4)
            metrics[f"norm_{key}"] = round(a_norm, 4)
            print(f"  σ={σ:.2f}: std={a_std:.3f}, norm={a_norm:.3f}")

        base_std  = metrics["std_s000"]
        base_norm = metrics["norm_s000"]
        ratio_std  = metrics["std_s010"]  / max(base_std,  1e-6)
        ratio_norm = metrics["norm_s010"] / max(base_norm, 1e-6)

        metrics.update(
            baseline_std=base_std, baseline_norm=base_norm,
            ratio_std_at10=round(ratio_std, 4),
            ratio_norm_at10=round(ratio_norm, 4),
            improvement=round(ratio_norm - ratio_std, 4),
        )

        if ratio_norm >= 0.50 and ratio_norm > ratio_std + 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Row-norm resilient: ratio_norm={ratio_norm:.3f} ≥ 0.50 "
                     f"(ratio_std={ratio_std:.3f}). Clean: norm={base_norm:.3f}, std={base_std:.3f}.")
        elif ratio_norm < 0.30:
            outcome = OUTCOME_REFUTED
            notes = (f"Row-norm still brittle: ratio_norm={ratio_norm:.3f} < 0.30. "
                     f"Normalisation does not prevent noise collapse.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Partial: ratio_norm={ratio_norm:.3f}, ratio_std={ratio_std:.3f}, "
                     f"improvement={ratio_norm-ratio_std:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, eval_sigmas=EVAL_SIGMAS))


if __name__ == "__main__":
    Exp372RowNormRobustness().execute()
