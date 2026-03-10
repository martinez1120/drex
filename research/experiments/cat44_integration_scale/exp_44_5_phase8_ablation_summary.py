"""
Experiment 44.5 — Phase 7+8 Component Ablation Summary

Hypothesis: The best single mechanism from Phase 7-8 (EMA smoothing) already captures
>60% of the combined improvement, and each additional mechanism (split memory, stable
gate init) contributes diminishing but positive marginal gains (>1% each).
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
SEQ_LEN    = 32
NUM_PAIRS  = 5
STEPS      = 800
BATCH      = 32
HALF       = HIDDEN_DIM // 2


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


class BaselineModel(nn.Module):
    """Config 1: plain delta rule, no EMA, no split, no gate."""
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

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k  = self.kp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


class EMAModel(nn.Module):
    """Config 2: +EMA alpha=0.95, unified memory."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, alpha=0.95):
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
            k     = self.kp(h[:, t, :])
            kn    = F.normalize(k, dim=-1)
            vp    = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
            if self.alpha < 1.0:
                M = M + (1.0 - self.alpha) * Delta
            else:
                M = M + Delta
        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


class SplitModel(nn.Module):
    """Config 3: +Split (episodic/semantic), no EMA, no gate."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
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
    """Config 4: +EMA+Split (alpha=0.95 + episodic/semantic split)."""
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


class GateModel(nn.Module):
    """Config 5: +Gate (energy gate thresh=0.4, unified memory, no EMA, no split)."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, gate_thresh=0.4):
        super().__init__()
        self.gate_thresh = gate_thresh
        self.hidden_dim  = hidden_dim
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
            k     = self.kp(h[:, t, :])
            kn    = F.normalize(k, dim=-1)
            vp    = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
            energy = Delta.pow(2).mean([1, 2])
            gate   = (energy >= self.gate_thresh).float()[:, None, None]
            M      = M + gate * Delta
        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


class FullModel(nn.Module):
    """Config 6: +EMA+Split+Gate — all three mechanisms combined."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE,
                 alpha=0.95, gate_thresh=0.4):
        super().__init__()
        self.alpha       = alpha
        self.gate_thresh = gate_thresh
        self.hidden_dim  = hidden_dim
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
            energy  = (Delta_s.pow(2).mean([1, 2]) + Delta_e.pow(2).mean([1, 2])) * 0.5
            gate    = (energy >= self.gate_thresh).float()[:, None, None]
            Delta_s = gate * Delta_s
            Delta_e = gate * Delta_e
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


# 6 configurations ordered by complexity
CONFIGS = [
    ("baseline",   lambda: BaselineModel()),
    ("ema",        lambda: EMAModel(alpha=0.95)),
    ("split",      lambda: SplitModel()),
    ("ema_split",  lambda: EMASplitModel(alpha=0.95)),
    ("gate",       lambda: GateModel(gate_thresh=0.4)),
    ("full",       lambda: FullModel(alpha=0.95, gate_thresh=0.4)),
]


class Exp445Phase8AblationSummary(Experiment):
    experiment_id = "exp_44_5"
    hypothesis = (
        "The best single mechanism from Phase 7-8 (EMA smoothing) already captures "
        ">60% of the combined improvement, and each additional mechanism (split memory, "
        "stable gate init) contributes diminishing but positive marginal gains (>1% each)."
    )

    def run(self) -> ExperimentResult:
        accs: dict[str, float] = {}

        for name, build_fn in CONFIGS:
            print(f"Training config={name} ...")
            acc = train_eval(build_fn())
            accs[name] = acc
            print(f"  acc_{name}={acc:.4f}")

        acc_baseline  = accs["baseline"]
        acc_ema       = accs["ema"]
        acc_split     = accs["split"]
        acc_ema_split = accs["ema_split"]
        acc_gate      = accs["gate"]
        acc_full      = accs["full"]

        marginal_ema       = round(float(acc_ema)       - float(acc_baseline),  4)
        marginal_split     = round(float(acc_split)     - float(acc_baseline),  4)
        marginal_ema_split = round(float(acc_ema_split) - float(acc_ema),       4)
        combined_gain      = round(float(acc_full)      - float(acc_baseline),  4)
        fraction_from_ema  = round(
            float(marginal_ema) / max(float(combined_gain), 1e-4), 4
        )

        metrics = {
            "acc_baseline":      round(float(acc_baseline),  4),
            "acc_ema":           round(float(acc_ema),       4),
            "acc_split":         round(float(acc_split),     4),
            "acc_ema_split":     round(float(acc_ema_split), 4),
            "acc_gate":          round(float(acc_gate),      4),
            "acc_full":          round(float(acc_full),      4),
            "marginal_ema":      marginal_ema,
            "marginal_split":    marginal_split,
            "marginal_ema_split":marginal_ema_split,
            "combined_gain":     combined_gain,
            "fraction_from_ema": fraction_from_ema,
        }

        gate_contrib = round(float(acc_full) - float(acc_ema_split), 4)

        if fraction_from_ema > 0.60 and marginal_ema_split > 0.01 and gate_contrib > 0.01:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"EMA dominates with fraction_from_ema={fraction_from_ema:.3f}>0.60. "
                f"marginal_ema_split={marginal_ema_split:.3f}>0.01. "
                f"gate contribution={gate_contrib:.3f}>0.01. "
                "Diminishing but positive marginal gains confirmed."
            )
        elif fraction_from_ema < 0.40:
            outcome = OUTCOME_REFUTED
            notes = (
                f"EMA is NOT the dominant mechanism: fraction_from_ema={fraction_from_ema:.3f}<0.40. "
                f"Split or gate is the primary contributor. combined_gain={combined_gain:.3f}."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"fraction_from_ema={fraction_from_ema:.3f} (threshold >0.60). "
                f"marginal_ema_split={marginal_ema_split:.3f} (threshold >0.01). "
                f"gate_contrib={gate_contrib:.3f} (threshold >0.01). "
                "Not all three conditions met for SUPPORTED."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                steps=STEPS, batch=BATCH, alpha=0.95, gate_thresh=0.4,
            ),
        )


if __name__ == "__main__":
    Exp445Phase8AblationSummary().execute()
