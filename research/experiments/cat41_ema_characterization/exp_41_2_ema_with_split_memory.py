"""
Experiment 41.2 — EMA Applied to Episodic/Semantic Split Memory

Hypothesis: Combining EMA smoothing (α=0.95) with the episodic/semantic split memory
outperforms both EMA-alone and split-alone by >3%, showing the mechanisms are
orthogonal and composable.
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
HALF = 32
ALPHA = 0.95


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


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, seq):
        h = self.embed(seq)
        return self.norm(h + self.ff(h))


class StandardDelta(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.rp  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            Delta = torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
            M = M + Delta
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


class EMaDelta(nn.Module):
    def __init__(self, alpha=0.95, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.alpha = alpha
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.rp  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]; kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            dv = k - vp
            Delta = torch.bmm(dv.unsqueeze(-1), kn.unsqueeze(1))
            M = M + (1.0 - self.alpha) * Delta
        return self.out(self.rp(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1)))


class SplitDelta(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, half=HALF, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.half = half
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.proj_s = nn.Linear(hidden_dim, half)
        self.proj_e = nn.Linear(hidden_dim, half)
        self.rp  = nn.Linear(half * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def _write_memories(self, M_s, M_e, h, t, L):
        hs = self.proj_s(h[:, t, :]); he = self.proj_e(h[:, t, :])
        ks = F.normalize(hs, dim=-1); ke = F.normalize(he, dim=-1)
        vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
        vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
        dvs = hs - vps; dve = he - vpe
        M_s = M_s + torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
        M_e = M_e + ((t + 1) / L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        return M_s, M_e

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        M_s = torch.zeros(B, self.half, self.half, device=h.device)
        M_e = torch.zeros(B, self.half, self.half, device=h.device)
        for t in range(L - 1):
            M_s, M_e = self._write_memories(M_s, M_e, h, t, L)
        q = h[:, -1, :]
        qs = self.proj_s(q); qe = self.proj_e(q)
        rs = torch.bmm(M_s, F.normalize(qs, dim=-1).unsqueeze(-1)).squeeze(-1)
        re = torch.bmm(M_e, F.normalize(qe, dim=-1).unsqueeze(-1)).squeeze(-1)
        r = torch.cat([rs, re], dim=-1)
        return self.out(self.rp(r))


class EMASpitDelta(nn.Module):
    def __init__(self, alpha=0.95, hidden_dim=HIDDEN_DIM, half=HALF, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.alpha = alpha
        self.half = half
        self.encoder = Encoder(vocab_size, hidden_dim)
        self.proj_s = nn.Linear(hidden_dim, half)
        self.proj_e = nn.Linear(hidden_dim, half)
        self.rp  = nn.Linear(half * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        M_s = torch.zeros(B, self.half, self.half, device=h.device)
        M_e = torch.zeros(B, self.half, self.half, device=h.device)
        for t in range(L - 1):
            hs = self.proj_s(h[:, t, :]); he = self.proj_e(h[:, t, :])
            ks = F.normalize(hs, dim=-1); ke = F.normalize(he, dim=-1)
            vps = torch.bmm(M_s, ks.unsqueeze(-1)).squeeze(-1)
            vpe = torch.bmm(M_e, ke.unsqueeze(-1)).squeeze(-1)
            dvs = hs - vps; dve = he - vpe
            M_s = M_s + (1.0 - self.alpha) * torch.bmm(dvs.unsqueeze(-1), ks.unsqueeze(1))
            M_e = M_e + (1.0 - self.alpha) * ((t + 1) / L) * torch.bmm(dve.unsqueeze(-1), ke.unsqueeze(1))
        q = h[:, -1, :]
        qs = self.proj_s(q); qe = self.proj_e(q)
        rs = torch.bmm(M_s, F.normalize(qs, dim=-1).unsqueeze(-1)).squeeze(-1)
        re = torch.bmm(M_e, F.normalize(qe, dim=-1).unsqueeze(-1)).squeeze(-1)
        r = torch.cat([rs, re], dim=-1)
        return self.out(self.rp(r))


def train_and_eval(model):
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_batch(BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS, vocab_size=VOCAB_SIZE)
            preds = model(seq).argmax(dim=-1)
            correct += (preds == tgt).sum().item(); total += BATCH
    return correct / total


class Exp412EmaWithSplitMemory(Experiment):
    experiment_id = "exp_41_2"
    hypothesis = (
        "Combining EMA smoothing (α=0.95) with the episodic/semantic split memory "
        "outperforms both EMA-alone and split-alone by >3%, showing the mechanisms are "
        "orthogonal and composable."
    )

    def run(self) -> ExperimentResult:
        acc_standard  = train_and_eval(StandardDelta())
        acc_ema       = train_and_eval(EMaDelta(alpha=ALPHA))
        acc_split     = train_and_eval(SplitDelta())
        acc_ema_split = train_and_eval(EMASpitDelta(alpha=ALPHA))

        gap_ema_over_split = acc_ema_split - acc_split
        gap_ema_over_ema   = acc_ema_split - acc_ema

        metrics = {
            "acc_standard":       round(acc_standard, 4),
            "acc_ema":            round(acc_ema, 4),
            "acc_split":          round(acc_split, 4),
            "acc_ema_split":      round(acc_ema_split, 4),
            "gap_ema_over_split": round(gap_ema_over_split, 4),
            "gap_ema_over_ema":   round(gap_ema_over_ema, 4),
        }

        config = {
            "vocab_size": VOCAB_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "half": HALF,
            "steps": STEPS,
            "alpha": ALPHA,
        }

        if gap_ema_over_split > 0.03 and gap_ema_over_ema > 0.03:
            outcome = OUTCOME_SUPPORTED
        elif acc_ema_split < max(acc_split, acc_ema) - 0.03:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (
            f"acc_standard={round(acc_standard, 4)}, acc_ema={round(acc_ema, 4)}, "
            f"acc_split={round(acc_split, 4)}, acc_ema_split={round(acc_ema_split, 4)}, "
            f"gap_over_split={round(gap_ema_over_split, 4)}, gap_over_ema={round(gap_ema_over_ema, 4)}"
        )
        return self.result(outcome, metrics, notes, config=config)


if __name__ == "__main__":
    Exp412EmaWithSplitMemory().execute()
