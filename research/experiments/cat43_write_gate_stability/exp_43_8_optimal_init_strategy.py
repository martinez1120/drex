"""
Experiment 43.8 — Optimal Write Gate Initialization

Hypothesis: Initializing the learnable threshold at 0.4 (the known accuracy-maximizing
region from exp_39_1) reliably achieves >90% of the maximum possible write-gate accuracy,
showing that good initialization is sufficient to solve the multi-stability problem
without architectural changes.
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
BATCH = 32
STEPS = 800
INIT_THRESHOLDS = [0.05, 0.20, 0.40, 0.80, 1.20]


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


class LearnableThresholdDelta(nn.Module):
    def __init__(self, init_threshold=0.4):
        super().__init__()
        log_val = float(torch.tensor(float(init_threshold)).log().clamp(-4, 2).item())
        self.log_thresh = nn.Parameter(torch.tensor(log_val))
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        self.kp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.vp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.qp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self._wr   = 0.0

    @property
    def threshold(self):
        return self.log_thresh.exp()

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        writes = 0.0; total = 0
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp_val = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            err = (v - vp_val).norm(dim=-1)
            thresh = self.threshold
            gate = torch.sigmoid((err - thresh * v.norm(dim=-1)) * 10.0)
            writes = writes + gate.sum().item(); total += B
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp_val).unsqueeze(-1), kn.unsqueeze(1))
        self._wr = writes / max(total, 1)
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def train_and_eval(init_threshold, steps):
    model = LearnableThresholdDelta(init_threshold=init_threshold)
    opt = Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = 0; total = 0; wr_total = 0.0
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += BATCH
            wr_total += model._wr
    acc = correct / total
    avg_wr = wr_total / 20
    return round(acc, 4), round(avg_wr, 4)


class Exp438OptimalInitStrategy(Experiment):
    experiment_id = "exp_43_8"
    hypothesis = (
        "Initializing the learnable threshold at 0.4 (the known accuracy-maximizing "
        "region from exp_39_1) reliably achieves >90% of the maximum possible write-gate accuracy, "
        "showing that good initialization is sufficient to solve the multi-stability problem "
        "without architectural changes."
    )

    def run(self) -> ExperimentResult:
        accs = {}
        wrs = {}
        metrics = {}

        for t in INIT_THRESHOLDS:
            acc, wr = train_and_eval(t, STEPS)
            key = int(t * 100)
            accs[key] = acc
            wrs[key] = wr
            metrics[f"acc_init{key:03d}"] = acc
            metrics[f"wr_init{key:03d}"] = wr

        max_acc = max(accs.values())
        best_init_key = max(accs, key=lambda k: accs[k])
        best_init = best_init_key / 100.0

        acc_040 = accs[40]   # threshold 0.40
        acc_120 = accs[120]  # threshold 1.20
        acc_005 = accs[5]    # threshold 0.05

        acc_ratio_optimal = round(acc_040 / max_acc, 4) if max_acc > 1e-9 else 0.0

        metrics["max_acc"] = round(max_acc, 4)
        metrics["acc_ratio_optimal"] = acc_ratio_optimal
        metrics["best_init"] = round(best_init, 4)

        if (acc_ratio_optimal >= 0.90
                and acc_040 > acc_120 + 0.05
                and acc_040 > acc_005 + 0.05):
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Init at 0.4 achieves {acc_ratio_optimal:.3f} of max accuracy ({max_acc:.3f}), "
                f"which is >=0.90, and outperforms init=0.05 by {acc_040-acc_005:.3f} "
                f"and init=1.20 by {acc_040-acc_120:.3f} (both >0.05)."
            )
        elif acc_040 < max_acc - 0.05:
            outcome = OUTCOME_REFUTED
            justification = (
                f"Init at 0.4 does not reliably find good solution: "
                f"acc_040={acc_040:.3f} < max_acc-0.05={max_acc-0.05:.3f}. "
                f"Best init was {best_init:.2f} with acc={max_acc:.3f}."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Init at 0.4 achieved acc_ratio_optimal={acc_ratio_optimal:.3f} "
                f"(need >=0.90). acc_040={acc_040:.3f}, max_acc={max_acc:.3f}. "
                f"Margin over init=0.05: {acc_040-acc_005:.3f}, over init=1.20: {acc_040-acc_120:.3f}."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp438OptimalInitStrategy().execute()
