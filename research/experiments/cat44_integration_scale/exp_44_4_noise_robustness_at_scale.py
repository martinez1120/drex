"""
Experiment 44.4 — Full System Noise Robustness at Scale

Hypothesis: The EMA+Split combination achieves a noise cliff (50% clean accuracy drop)
at σ≥0.20 (post-training M noise), while standard delta has a cliff at σ≤0.10,
confirming the combined system's robustness advantage at standard test conditions.
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

VOCAB_SIZE   = 64
HIDDEN_DIM   = 64
SEQ_LEN      = 32
NUM_PAIRS    = 5
STEPS        = 800
BATCH        = 32
HALF         = HIDDEN_DIM // 2
EVAL_SIGMAS  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def make_batch(batch_size, seq_len=32, num_pairs=5, vocab_size=64):
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


class StandardDelta(nn.Module):
    """Plain delta rule with unified memory (alpha=1.0). Supports M noise at eval."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
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

    def forward(self, seq, noise_scale=0.0):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k  = self.kp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
        if noise_scale > 0.0:
            M = M + noise_scale * torch.randn_like(M) * M.std().clamp_min(1e-6)
        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


class EMASplitDelta(nn.Module):
    """EMA smoothing (α=0.95) + episodic/semantic split. Supports M noise at eval."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, alpha=0.95):
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

    def forward(self, seq, noise_scale=0.0):
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
        if noise_scale > 0.0:
            M_s = M_s + noise_scale * torch.randn_like(M_s) * M_s.std().clamp_min(1e-6)
            M_e = M_e + noise_scale * torch.randn_like(M_e) * M_e.std().clamp_min(1e-6)
        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(torch.cat([cs, ce], dim=-1)))


def train_model(model, steps=STEPS, batch=BATCH, seq_len=SEQ_LEN,
                num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE):
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()


def eval_at_sigma(model, sigma, batch=BATCH, seq_len=SEQ_LEN,
                  num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE, n_batches=60):
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
            ok  += (model(seq, noise_scale=sigma).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot


def find_cliff(sigma_to_acc: dict, clean_acc: float) -> float:
    """Return the first sigma where accuracy drops below 50% of clean accuracy."""
    for sigma in sorted(sigma_to_acc.keys()):
        if sigma > 0 and sigma_to_acc[sigma] < 0.5 * clean_acc:
            return sigma
    return float("inf")


class Exp444NoiseRobustnessAtScale(Experiment):
    experiment_id = "exp_44_4"
    hypothesis = (
        "The EMA+Split combination achieves a noise cliff (50% clean accuracy drop) "
        "at σ≥0.20 (post-training M noise), while standard delta has a cliff at σ≤0.10, "
        "confirming the combined system's robustness advantage at standard test conditions."
    )

    def run(self) -> ExperimentResult:
        print("Training StandardDelta (alpha=1.0, unified) ...")
        model_std = StandardDelta()
        train_model(model_std)

        print("Training EMASplitDelta (alpha=0.95 + split) ...")
        model_ema_split = EMASplitDelta(alpha=0.95)
        train_model(model_ema_split)

        print("Evaluating noise robustness across sigmas ...")
        std_accs: dict[float, float]       = {}
        ema_split_accs: dict[float, float] = {}

        for sigma in EVAL_SIGMAS:
            acc_std       = eval_at_sigma(model_std,       sigma)
            acc_ema_split = eval_at_sigma(model_ema_split, sigma)
            std_accs[sigma]       = acc_std
            ema_split_accs[sigma] = acc_ema_split
            print(f"  sigma={sigma:.2f}  std={acc_std:.4f}  ema_split={acc_ema_split:.4f}")

        clean_std       = std_accs[0.0]
        clean_ema_split = ema_split_accs[0.0]

        cliff_std       = find_cliff(std_accs,       clean_std)
        cliff_ema_split = find_cliff(ema_split_accs, clean_ema_split)

        metrics: dict = {}
        for sigma in EVAL_SIGMAS:
            key = int(round(sigma * 100))
            metrics[f"acc_std_s{key:03d}"]       = round(float(std_accs[sigma]),       4)
            metrics[f"acc_ema_split_s{key:03d}"]  = round(float(ema_split_accs[sigma]), 4)

        metrics["cliff_std"]       = round(float(cliff_std),       4) if cliff_std != float("inf") else 9999.0
        metrics["cliff_ema_split"] = round(float(cliff_ema_split), 4) if cliff_ema_split != float("inf") else 9999.0

        cliff_std_val       = cliff_std       if cliff_std       != float("inf") else 9.99
        cliff_ema_split_val = cliff_ema_split if cliff_ema_split != float("inf") else 9.99

        if cliff_ema_split_val >= 0.20 and cliff_std_val <= 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Noise cliff confirmed: ema_split cliff={cliff_ema_split_val:.2f}>=0.20, "
                f"std cliff={cliff_std_val:.2f}<=0.10. "
                f"Clean accs: std={clean_std:.3f}, ema_split={clean_ema_split:.3f}."
            )
        elif cliff_ema_split_val <= cliff_std_val:
            outcome = OUTCOME_REFUTED
            notes = (
                f"EMA+Split is not more robust: cliff_ema_split={cliff_ema_split_val:.2f} "
                f"<= cliff_std={cliff_std_val:.2f}. No robustness advantage."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Partial robustness advantage: ema_split cliff={cliff_ema_split_val:.2f}, "
                f"std cliff={cliff_std_val:.2f}. "
                f"EMA+Split is more robust but thresholds (0.20 / 0.10) not both met."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                steps=STEPS, batch=BATCH, alpha=0.95,
                eval_sigmas=EVAL_SIGMAS,
            ),
        )


if __name__ == "__main__":
    Exp444NoiseRobustnessAtScale().execute()
