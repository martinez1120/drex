"""
Experiment 44.1 — Full System Integration: EMA + Episodic/Semantic + Stable Gate Init

Hypothesis: Combining the three Phase 7-8 positive findings — EMA smoothing (α=0.95),
episodic/semantic split memory, and a well-initialized write gate (thresh=0.4) —
outperforms all partial combinations by >3%, showing the mechanisms are orthogonal.
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


class IntegratedModel(nn.Module):
    """
    Unified model that can enable/disable EMA smoothing, episodic/semantic split,
    and hard energy-gate write gating independently.
    """
    def __init__(self, use_ema=False, use_split=False, use_gate=False,
                 alpha=0.95, gate_thresh=0.4,
                 hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_ema    = use_ema
        self.use_split  = use_split
        self.use_gate   = use_gate
        self.alpha      = alpha
        self.gate_thresh = gate_thresh
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        if use_split:
            half = hidden_dim // 2
            self.sem_p = nn.Linear(hidden_dim, half, bias=False)
            self.epi_p = nn.Linear(hidden_dim, half, bias=False)
            self.rp    = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.kp = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.rp = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        half = H // 2

        if self.use_split:
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
                if self.use_gate:
                    energy = (Delta_s.pow(2).mean([1, 2]) + Delta_e.pow(2).mean([1, 2])) * 0.5
                    gate   = (energy >= self.gate_thresh).float()[:, None, None]
                    Delta_s = gate * Delta_s
                    Delta_e = gate * Delta_e
                if self.use_ema and self.alpha < 1.0:
                    M_s = M_s + (1.0 - self.alpha) * Delta_s
                    M_e = M_e + (1.0 - self.alpha) * Delta_e
                else:
                    M_s = M_s + Delta_s
                    M_e = M_e + Delta_e
            q  = h[:, -1, :]
            cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
            ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
            read = torch.cat([cs, ce], dim=-1)
        else:
            M = torch.zeros(B, H, H, device=h.device)
            for t in range(L - 1):
                k  = self.kp(h[:, t, :])
                kn = F.normalize(k, dim=-1)
                vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
                Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
                if self.use_gate:
                    energy = Delta.pow(2).mean([1, 2])
                    gate   = (energy >= self.gate_thresh).float()[:, None, None]
                    Delta  = gate * Delta
                if self.use_ema and self.alpha < 1.0:
                    M = M + (1.0 - self.alpha) * Delta
                else:
                    M = M + Delta
            q    = self.kp(h[:, -1, :])
            read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)

        return self.out(self.rp(read))


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


# 8 ablation conditions: all 2^3 combos of (EMA, split, gate)
CONFIGS = [
    ("baseline",   False, False, False),
    ("ema",        True,  False, False),
    ("split",      False, True,  False),
    ("gate",       False, False, True),
    ("ema_split",  True,  True,  False),
    ("ema_gate",   True,  False, True),
    ("split_gate", False, True,  True),
    ("full",       True,  True,  True),
]


class Exp441FullSystemIntegration(Experiment):
    experiment_id = "exp_44_1"
    hypothesis = (
        "Combining the three Phase 7-8 positive findings — EMA smoothing (α=0.95), "
        "episodic/semantic split memory, and a well-initialized write gate (thresh=0.4) — "
        "outperforms all partial combinations by >3%, showing the mechanisms are orthogonal."
    )

    def run(self) -> ExperimentResult:
        accs: dict[str, float] = {}

        for name, use_ema, use_split, use_gate in CONFIGS:
            print(f"Training config={name} (use_ema={use_ema}, use_split={use_split}, use_gate={use_gate}) ...")
            model = IntegratedModel(
                use_ema=use_ema, use_split=use_split, use_gate=use_gate,
                alpha=0.95, gate_thresh=0.4,
            )
            acc = train_eval(model)
            accs[name] = acc
            print(f"  acc_{name}={acc:.4f}")

        acc_full         = accs["full"]
        partial_names    = ["ema", "split", "gate", "ema_split", "ema_gate", "split_gate"]
        best_partial     = max(accs[n] for n in partial_names)
        gap_full_vs_best = round(float(acc_full) - float(best_partial), 4)

        metrics = {
            "acc_baseline":              round(float(accs["baseline"]),   4),
            "acc_ema":                   round(float(accs["ema"]),        4),
            "acc_split":                 round(float(accs["split"]),      4),
            "acc_gate":                  round(float(accs["gate"]),       4),
            "acc_ema_split":             round(float(accs["ema_split"]),  4),
            "acc_ema_gate":              round(float(accs["ema_gate"]),   4),
            "acc_split_gate":            round(float(accs["split_gate"]), 4),
            "acc_full":                  round(float(acc_full),           4),
            "gap_full_vs_best_partial":  gap_full_vs_best,
        }

        if gap_full_vs_best > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Full system exceeds best partial combo by {gap_full_vs_best:.3f}>0.03. "
                f"acc_full={acc_full:.3f}, best_partial={best_partial:.3f}. "
                "EMA+Split+Gate mechanisms combine orthogonally."
            )
        elif gap_full_vs_best < -0.03:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Full combination hurts: gap={gap_full_vs_best:.3f}<-0.03. "
                f"acc_full={acc_full:.3f}, best_partial={best_partial:.3f}. "
                "Mechanisms have negative interactions when combined."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Full vs best-partial gap={gap_full_vs_best:.3f} within ±0.03. "
                f"acc_full={acc_full:.3f}, best_partial={best_partial:.3f}. "
                "Cannot confirm orthogonality or interference."
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
    Exp441FullSystemIntegration().execute()
