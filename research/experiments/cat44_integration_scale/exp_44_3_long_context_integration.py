"""
Experiment 44.3 — Full System at Long Context (SEQ_LEN=128)

Hypothesis: The EMA+Split combination maintains >70% accuracy at SEQ_LEN=128
with NUM_PAIRS=10, while the standard delta rule drops below 50%.
Long contexts benefit most from both EMA (smoothing accumulated interference)
and episodic/semantic split (temporal structure increasingly important).
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
SEQ_LEN    = 128
NUM_PAIRS  = 10
STEPS      = 1000
BATCH      = 32
HALF       = HIDDEN_DIM // 2


def make_batch(batch_size, seq_len=128, num_pairs=10, vocab_size=64):
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
    """Plain delta rule with unified memory, no EMA (alpha=1.0)."""
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


class EMADelta(nn.Module):
    """EMA-smoothed delta rule with unified memory (alpha=0.95)."""
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


class SplitDelta(nn.Module):
    """Episodic/semantic split memory, no EMA."""
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


class EMASplitDelta(nn.Module):
    """EMA smoothing (α=0.95) combined with episodic/semantic split memory."""
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


class Exp443LongContextIntegration(Experiment):
    experiment_id = "exp_44_3"
    hypothesis = (
        "The EMA+Split combination maintains >70% accuracy at SEQ_LEN=128 with NUM_PAIRS=10, "
        "while the standard delta rule drops below 50%. Long contexts benefit most from both "
        "EMA (smoothing accumulated interference) and episodic/semantic split "
        "(temporal structure increasingly important)."
    )

    def run(self) -> ExperimentResult:
        print("Training StandardDelta (alpha=1.0, unified) at SEQ_LEN=128 ...")
        acc_std = train_eval(StandardDelta())
        print(f"  acc_std={acc_std:.4f}")

        print("Training EMADelta (alpha=0.95, unified) at SEQ_LEN=128 ...")
        acc_ema = train_eval(EMADelta(alpha=0.95))
        print(f"  acc_ema={acc_ema:.4f}")

        print("Training SplitDelta (no EMA) at SEQ_LEN=128 ...")
        acc_split = train_eval(SplitDelta())
        print(f"  acc_split={acc_split:.4f}")

        print("Training EMASplitDelta (alpha=0.95 + split) at SEQ_LEN=128 ...")
        acc_ema_split = train_eval(EMASplitDelta(alpha=0.95))
        print(f"  acc_ema_split={acc_ema_split:.4f}")

        gap_ema_split_vs_std = round(float(acc_ema_split) - float(acc_std), 4)

        metrics = {
            "acc_std":               round(float(acc_std),       4),
            "acc_ema":               round(float(acc_ema),       4),
            "acc_split":             round(float(acc_split),     4),
            "acc_ema_split":         round(float(acc_ema_split), 4),
            "gap_ema_split_vs_std":  gap_ema_split_vs_std,
        }

        if acc_ema_split > 0.70 and acc_std < 0.50:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"EMA+Split achieves acc={acc_ema_split:.3f}>0.70 while std delta "
                f"acc={acc_std:.3f}<0.50 at SEQ_LEN=128. Long-context advantage confirmed. "
                f"gap_ema_split_vs_std={gap_ema_split_vs_std:.3f}."
            )
        elif acc_std >= 0.70:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Standard delta achieves acc={acc_std:.3f}>=0.70 even at SEQ_LEN=128. "
                "EMA+Split combination offers no unique long-context advantage."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"acc_ema_split={acc_ema_split:.3f} (threshold 0.70), "
                f"acc_std={acc_std:.3f} (threshold 0.50). "
                f"gap={gap_ema_split_vs_std:.3f}. Neither condition fully met."
            )

        return self.result(
            outcome, metrics, notes,
            config=dict(
                vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                steps=STEPS, batch=BATCH, alpha=0.95,
            ),
        )


if __name__ == "__main__":
    Exp443LongContextIntegration().execute()
