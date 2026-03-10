"""
Experiment 44.2 — Scaling Positive Findings to Larger Model

Hypothesis: The EMA advantage (α=0.95) over standard delta persists at larger
hidden dimension (HIDDEN_DIM=128): the accuracy gap is >2% at H=128, confirming
that EMA is not merely compensating for small-model overfitting.
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

VOCAB_SIZE  = 128
SEQ_LEN     = 32
NUM_PAIRS   = 8
STEPS       = 800
BATCH       = 32
HIDDEN_DIMS = [64, 128]


def make_batch(batch_size, seq_len=32, num_pairs=8, vocab_size=128):
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
    """Standard or EMA delta rule with unified memory matrix."""
    def __init__(self, hidden_dim=64, vocab_size=128, alpha=1.0):
        super().__init__()
        self.alpha      = alpha
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.kp   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rp   = nn.Linear(hidden_dim, hidden_dim)
        self.out  = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k  = self.kp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
            if self.alpha < 1.0:
                M = M + (1.0 - self.alpha) * Delta
            else:
                M = M + Delta
        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


class SplitModel(nn.Module):
    """Episodic/semantic split memory, no EMA."""
    def __init__(self, hidden_dim=64, vocab_size=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        half = hidden_dim // 2
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.sem_p = nn.Linear(hidden_dim, half, bias=False)
        self.epi_p = nn.Linear(hidden_dim, half, bias=False)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        half = H // 2
        M_s = torch.zeros(B, half, half, device=h.device)
        M_e = torch.zeros(B, half, half, device=h.device)
        for t in range(L - 1):
            ks  = self.sem_p(h[:, t, :])
            ke  = self.epi_p(h[:, t, :])
            kns = F.normalize(ks, dim=-1)
            kne = F.normalize(ke, dim=-1)
            vps = torch.bmm(M_s, kns.unsqueeze(-1)).squeeze(-1)
            vpe = torch.bmm(M_e, kne.unsqueeze(-1)).squeeze(-1)
            M_s = M_s + torch.bmm((ks - vps).unsqueeze(-1), kns.unsqueeze(1))
            M_e = M_e + torch.bmm((ke - vpe).unsqueeze(-1), kne.unsqueeze(1))
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(torch.cat([cs, ce], dim=-1)))


class EMASplitModel(nn.Module):
    """EMA smoothing (α=0.95) combined with episodic/semantic split memory."""
    def __init__(self, hidden_dim=64, vocab_size=128, alpha=0.95):
        super().__init__()
        self.alpha      = alpha
        self.hidden_dim = hidden_dim
        half = hidden_dim // 2
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.sem_p = nn.Linear(hidden_dim, half, bias=False)
        self.epi_p = nn.Linear(hidden_dim, half, bias=False)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        half = H // 2
        M_s = torch.zeros(B, half, half, device=h.device)
        M_e = torch.zeros(B, half, half, device=h.device)
        for t in range(L - 1):
            ks  = self.sem_p(h[:, t, :])
            ke  = self.epi_p(h[:, t, :])
            kns = F.normalize(ks, dim=-1)
            kne = F.normalize(ke, dim=-1)
            vps = torch.bmm(M_s, kns.unsqueeze(-1)).squeeze(-1)
            vpe = torch.bmm(M_e, kne.unsqueeze(-1)).squeeze(-1)
            Delta_s = torch.bmm((ks - vps).unsqueeze(-1), kns.unsqueeze(1))
            Delta_e = torch.bmm((ke - vpe).unsqueeze(-1), kne.unsqueeze(1))
            if self.alpha < 1.0:
                M_s = M_s + (1.0 - self.alpha) * Delta_s
                M_e = M_e + (1.0 - self.alpha) * Delta_e
            else:
                M_s = M_s + Delta_s
                M_e = M_e + Delta_e
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(torch.cat([cs, ce], dim=-1)))


def train_eval(model, steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
               num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE):
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot


class Exp442ScaleToLargerModel(Experiment):
    experiment_id = "exp_44_2"
    hypothesis = (
        "The EMA advantage (α=0.95) over standard delta persists at larger hidden dimension "
        "(HIDDEN_DIM=128): the accuracy gap is >2% at H=128, confirming that EMA is not "
        "merely compensating for small-model overfitting."
    )

    def run(self) -> ExperimentResult:
        results_by_dim: dict[int, dict[str, float]] = {}

        for hidden_dim in HIDDEN_DIMS:
            print(f"\n--- hidden_dim={hidden_dim} ---")

            print(f"  Training StandardDelta (alpha=1.0, H={hidden_dim}) ...")
            acc_std = train_eval(
                DeltaBase(hidden_dim=hidden_dim, vocab_size=VOCAB_SIZE, alpha=1.0)
            )
            print(f"  acc_std={acc_std:.4f}")

            print(f"  Training EMADelta (alpha=0.95, H={hidden_dim}) ...")
            acc_ema = train_eval(
                DeltaBase(hidden_dim=hidden_dim, vocab_size=VOCAB_SIZE, alpha=0.95)
            )
            print(f"  acc_ema={acc_ema:.4f}")

            print(f"  Training SplitModel (H={hidden_dim}) ...")
            acc_split = train_eval(
                SplitModel(hidden_dim=hidden_dim, vocab_size=VOCAB_SIZE)
            )
            print(f"  acc_split={acc_split:.4f}")

            print(f"  Training EMASplitModel (alpha=0.95, H={hidden_dim}) ...")
            acc_ema_split = train_eval(
                EMASplitModel(hidden_dim=hidden_dim, vocab_size=VOCAB_SIZE, alpha=0.95)
            )
            print(f"  acc_ema_split={acc_ema_split:.4f}")

            results_by_dim[hidden_dim] = {
                "std":       acc_std,
                "ema":       acc_ema,
                "split":     acc_split,
                "ema_split": acc_ema_split,
            }

        r64  = results_by_dim[64]
        r128 = results_by_dim[128]

        gap_ema_h64  = round(float(r64["ema"])  - float(r64["std"]),  4)
        gap_ema_h128 = round(float(r128["ema"]) - float(r128["std"]), 4)
        gain_ratio   = round(
            float(gap_ema_h64) / max(abs(float(gap_ema_h128)), 1e-4), 4
        ) if gap_ema_h128 != 0.0 else float("inf")

        metrics = {
            "acc_std_h64":       round(float(r64["std"]),       4),
            "acc_ema_h64":       round(float(r64["ema"]),       4),
            "acc_split_h64":     round(float(r64["split"]),     4),
            "acc_ema_split_h64": round(float(r64["ema_split"]), 4),
            "acc_std_h128":      round(float(r128["std"]),      4),
            "acc_ema_h128":      round(float(r128["ema"]),      4),
            "acc_split_h128":    round(float(r128["split"]),    4),
            "acc_ema_split_h128":round(float(r128["ema_split"]),4),
            "gap_ema_h64":       gap_ema_h64,
            "gap_ema_h128":      gap_ema_h128,
            "gain_ratio":        gain_ratio,
        }

        if gap_ema_h128 > 0.02 and (float(r128["split"]) - float(r128["std"])) > 0.02:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"EMA advantage persists at H=128: gap_ema_h128={gap_ema_h128:.3f}>0.02 "
                f"and split advantage {float(r128['split'])-float(r128['std']):.3f}>0.02. "
                f"gain_ratio={gain_ratio:.3f} (ratio of H64 to H128 EMA gap)."
            )
        elif gap_ema_h128 < 0.0:
            outcome = OUTCOME_REFUTED
            notes = (
                f"EMA hurts at H=128: gap_ema_h128={gap_ema_h128:.3f}<0.0. "
                "EMA benefit disappears or reverses at larger scale."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"EMA gap at H=128 is {gap_ema_h128:.3f}, positive but ≤0.02 threshold. "
                f"Split advantage at H=128: {float(r128['split'])-float(r128['std']):.3f}. "
                "Cannot confirm scale-invariance of EMA benefit."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dims=HIDDEN_DIMS,
                seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                steps=STEPS, batch=BATCH, alpha=0.95,
            ),
        )


if __name__ == "__main__":
    Exp442ScaleToLargerModel().execute()
