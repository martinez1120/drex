"""
Experiment 37.1 — Noise-Augmented Training Improves Memory Robustness

Hypothesis: Training DeltaModel with Gaussian noise on M at read-time (σ_train=0.05)
reduces catastrophic collapse: accuracy at σ_test=0.10 stays ≥50% of clean baseline
(vs the 8% ratio observed in exp_35_1).

Logic: If the model learns under noisy M, it develops representations that are more
tolerant of corrupted memory — a form of dropout/denoising regularization on M.
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
SIGMA_TRAIN = 0.05        # noise scale during training
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


class DeltaModel(nn.Module):
    """Standard DeltaModel — adds noise to M at read-time when noise_scale > 0."""
    def __init__(self):
        super().__init__()
        self.embed    = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff       = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                      nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm     = nn.LayerNorm(HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq, noise_scale=0.0):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp; denom = kn.pow(2).sum(-1, keepdim=True) + 1e-6
            M = M + torch.bmm(dv.unsqueeze(-1) / denom.unsqueeze(-1), kn.unsqueeze(1))
        if noise_scale > 0:
            M = M + noise_scale * torch.randn_like(M) * M.std().clamp_min(1e-6)
        q = h[:, -1, :]
        return self.output(self.read_proj(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def eval_at_sigma(model, sigma, n_batches=50):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_batch(BATCH)
            pred = model(seq, noise_scale=sigma).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    model.train()
    return correct / total


class Exp371NoiseAugmentedTraining(Experiment):
    experiment_id = "exp_37_1"
    hypothesis = (
        "Training DeltaModel with σ_train=0.05 noise on M keeps acc_ratio(σ=0.10) ≥ 0.50 "
        "(vs 0.08 ratio found in exp_35_1 with no augmentation)."
    )

    def run(self) -> ExperimentResult:
        # Condition A: standard training (no noise augmentation)
        print("Training standard DeltaModel (no noise augmentation)...")
        model_std = DeltaModel()
        opt_std = Adam(model_std.parameters(), lr=3e-4)
        model_std.train()
        for _ in range(STEPS):
            seq, tgt = make_batch(BATCH)
            F.cross_entropy(model_std(seq, noise_scale=0.0), tgt).backward()
            opt_std.step(); opt_std.zero_grad()

        # Condition B: noise-augmented training
        print("Training DeltaModel with noise augmentation (σ_train=0.05)...")
        model_aug = DeltaModel()
        opt_aug = Adam(model_aug.parameters(), lr=3e-4)
        model_aug.train()
        for _ in range(STEPS):
            seq, tgt = make_batch(BATCH)
            F.cross_entropy(model_aug(seq, noise_scale=SIGMA_TRAIN), tgt).backward()
            opt_aug.step(); opt_aug.zero_grad()

        # Evaluate both at all noise levels
        accs_std: dict[str, float] = {}
        accs_aug: dict[str, float] = {}
        for σ in EVAL_SIGMAS:
            a_std = eval_at_sigma(model_std, σ)
            a_aug = eval_at_sigma(model_aug, σ)
            accs_std[f"std_s{int(σ*100):03d}"] = round(a_std, 4)
            accs_aug[f"aug_s{int(σ*100):03d}"] = round(a_aug, 4)
            print(f"  σ={σ:.2f}: std={a_std:.3f}, aug={a_aug:.3f}")

        baseline_std = accs_std["std_s000"]
        baseline_aug = accs_aug["aug_s000"]
        ratio_std = accs_std["std_s010"] / max(baseline_std, 1e-6)
        ratio_aug = accs_aug["aug_s010"] / max(baseline_aug, 1e-6)

        metrics = dict(
            baseline_std=baseline_std,
            baseline_aug=baseline_aug,
            ratio_std_at10=round(ratio_std, 4),
            ratio_aug_at10=round(ratio_aug, 4),
            aug_improvement=round(ratio_aug - ratio_std, 4),
            **accs_std,
            **accs_aug,
        )

        # SUPPORTED: noise augmentation achieves ratio ≥ 0.50 AND improves over baseline
        if ratio_aug >= 0.50 and ratio_aug > ratio_std + 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Noise augmentation effective: ratio_aug={ratio_aug:.3f} ≥ 0.50 "
                     f"(vs ratio_std={ratio_std:.3f}). Baseline aug={baseline_aug:.3f}.")
        elif ratio_aug < 0.30:
            outcome = OUTCOME_REFUTED
            notes = (f"Noise augmentation fails: ratio_aug={ratio_aug:.3f} < 0.30. "
                     f"Still catastrophically brittle. ratio_std={ratio_std:.3f}.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Partial improvement: ratio_aug={ratio_aug:.3f}, "
                     f"ratio_std={ratio_std:.3f}. aug_improvement={ratio_aug-ratio_std:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM, steps=STEPS,
                                       sigma_train=SIGMA_TRAIN, eval_sigmas=EVAL_SIGMAS))


if __name__ == "__main__":
    Exp371NoiseAugmentedTraining().execute()
