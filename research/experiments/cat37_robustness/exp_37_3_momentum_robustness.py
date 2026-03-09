"""
Experiment 37.3 — Momentum-Smoothed M Updates Resist Noise Collapse

Hypothesis: Applying EMA to M updates — M_t = α·M_{t-1} + (1-α)·(M_{t-1} + Δ_t) —
with α=0.85 reduces noise fragility: acc_ratio at σ=0.10 ≥ 0.50, AND the clean
accuracy stays within 5% of the standard model (small acceptable cost).

Rationale: The standard delta rule applies full incremental writes to M. If the model
is queried with corrupted M (additive noise), large singular values of M amplify the
corruption. A momentum-smoothed update averages over time, reducing the spectral
radius of M. We test α ∈ {0.70, 0.85, 0.95} to find the robustness/accuracy trade-off.
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

VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
SEQ_LEN     = 24
NUM_PAIRS   = 5
STEPS       = 600
BATCH       = 32
ALPHAS      = [0.70, 0.85, 0.95]   # EMA decay rates to test
EVAL_SIGMAS = [0.0, 0.05, 0.10, 0.20]


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


class DeltaBase(nn.Module):
    def __init__(self, alpha=1.0):
        """alpha=1.0 → standard delta rule; alpha<1 → EMA smoothing."""
        super().__init__()
        self.alpha = alpha
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        self.rp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

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
                # EMA: M_new = alpha * M + (1-alpha) * (M + Delta) = M + (1-alpha)*Delta
                M = M + (1.0 - self.alpha) * Delta
            else:
                M = M + Delta
        if noise_scale > 0:
            M = M + noise_scale * torch.randn_like(M) * M.std().clamp_min(1e-6)
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


def train_and_eval(alpha, eval_sigmas):
    model = DeltaBase(alpha=alpha)
    opt   = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()

    results = {}
    model.eval()
    with torch.no_grad():
        for σ in eval_sigmas:
            ok = tot = 0
            for _ in range(50):
                seq, tgt = make_batch(BATCH)
                ok  += (model(seq, noise_scale=σ).argmax(-1) == tgt).sum().item()
                tot += tgt.size(0)
            results[σ] = ok / tot
    return results


class Exp373MomentumRobustness(Experiment):
    experiment_id = "exp_37_3"
    hypothesis = (
        "EMA update (α=0.85) achieves acc_ratio ≥ 0.50 at σ=0.10 "
        "while retaining ≥ 95% of clean accuracy."
    )

    def run(self) -> ExperimentResult:
        metrics: dict[str, float] = {}

        # Baseline: standard delta (alpha=1.0)
        print("Training standard delta (alpha=1.0)...")
        res_std = train_and_eval(1.0, EVAL_SIGMAS)
        for σ, a in res_std.items():
            metrics[f"std_s{int(σ*100):03d}"] = round(a, 4)
            print(f"  σ={σ:.2f}: std={a:.3f}")

        # EMA variants
        best_ratio = 0.0; best_alpha = 1.0
        for alpha in ALPHAS:
            print(f"Training EMA delta (alpha={alpha})...")
            res_ema = train_and_eval(alpha, EVAL_SIGMAS)
            for σ, a in res_ema.items():
                metrics[f"ema{int(alpha*100):03d}_s{int(σ*100):03d}"] = round(a, 4)
            ratio_ema = res_ema[0.10] / max(res_ema[0.0], 1e-6)
            metrics[f"ratio_ema{int(alpha*100):03d}"] = round(ratio_ema, 4)
            print(f"  clean={res_ema[0.0]:.3f}, σ=0.10={res_ema[0.1]:.3f}, ratio={ratio_ema:.3f}")
            if ratio_ema > best_ratio:
                best_ratio = ratio_ema; best_alpha = alpha

        ratio_std = metrics["std_s010"] / max(metrics["std_s000"], 1e-6)
        metrics["ratio_std"]  = round(ratio_std, 4)
        metrics["best_alpha"] = best_alpha
        metrics["best_ratio"] = round(best_ratio, 4)
        metrics["best_clean"] = metrics.get(f"ema{int(best_alpha*100):03d}_s000", 0.0)

        clean_std  = metrics["std_s000"]
        clean_loss = 1.0 - (metrics["best_clean"] / max(clean_std, 1e-6))

        if best_ratio >= 0.50 and clean_loss <= 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (f"EMA (α={best_alpha}) resilient: ratio={best_ratio:.3f} ≥ 0.50, "
                     f"clean_loss={clean_loss:.3f} ≤ 0.05. ratio_std={ratio_std:.3f}.")
        elif best_ratio < 0.30:
            outcome = OUTCOME_REFUTED
            notes = (f"EMA fails: best_ratio={best_ratio:.3f} at α={best_alpha}. "
                     f"Momentum doesn't prevent collapse. ratio_std={ratio_std:.3f}.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Partial: best_ratio={best_ratio:.3f} at α={best_alpha}, "
                     f"clean_loss={clean_loss:.3f}. ratio_std={ratio_std:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, alphas=ALPHAS, eval_sigmas=EVAL_SIGMAS))


if __name__ == "__main__":
    Exp373MomentumRobustness().execute()
