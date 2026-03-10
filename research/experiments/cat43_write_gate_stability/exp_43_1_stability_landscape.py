"""
Experiment 43.1 — Write Gate Full Stability Landscape

Hypothesis: The EnergyGated threshold has multiple stable equilibria: models initialized
at different thresholds converge to distinct stable values (multi-stability confirmed),
with the accuracy-maximizing equilibrium near threshold=0.3-0.5.
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
INIT_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0, 1.5]


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


def train_model(init_threshold, steps):
    model = LearnableThresholdDelta(init_threshold=init_threshold)
    opt = Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    # Eval accuracy
    correct = 0; total = 0
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += BATCH
    acc = correct / total
    return model.threshold.item(), model._wr, acc


class Exp431StabilityLandscape(Experiment):
    experiment_id = "exp_43_1"
    hypothesis = (
        "The EnergyGated threshold has multiple stable equilibria: models initialized "
        "at different thresholds converge to distinct stable values (multi-stability confirmed), "
        "with the accuracy-maximizing equilibrium near threshold=0.3-0.5."
    )

    def run(self) -> ExperimentResult:
        final_thresholds = []
        final_wrs = []
        final_accs = []
        metrics = {}

        for t in INIT_THRESHOLDS:
            final_thresh, final_wr, acc = train_model(t, STEPS)
            final_thresholds.append(final_thresh)
            final_wrs.append(final_wr)
            final_accs.append(acc)
            key = f"final_thresh_init{int(t * 100):03d}"
            metrics[key] = round(final_thresh, 4)

        spread = max(final_thresholds) - min(final_thresholds)

        # Cluster thresholds within 0.1 of each other
        sorted_thresholds = sorted(final_thresholds)
        clusters = []
        current_cluster = [sorted_thresholds[0]]
        for i in range(1, len(sorted_thresholds)):
            if sorted_thresholds[i] - sorted_thresholds[i - 1] <= 0.1:
                current_cluster.append(sorted_thresholds[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_thresholds[i]]
        clusters.append(current_cluster)
        n_clusters = len(clusters)

        best_idx = int(max(range(len(final_accs)), key=lambda i: final_accs[i]))
        best_thresh = final_thresholds[best_idx]
        best_acc = final_accs[best_idx]

        metrics["spread"] = round(spread, 4)
        metrics["n_clusters"] = n_clusters
        metrics["best_thresh"] = round(best_thresh, 4)
        metrics["best_acc"] = round(best_acc, 4)

        if n_clusters >= 3 and spread > 0.5:
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Multi-stability confirmed: {n_clusters} distinct clusters found "
                f"with spread={spread:.3f}>0.5 across {len(INIT_THRESHOLDS)} initializations."
            )
        elif n_clusters == 1 and spread < 0.2:
            outcome = OUTCOME_REFUTED
            justification = (
                f"Single attractor found: all thresholds converged to same cluster "
                f"with spread={spread:.3f}<0.2."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Partial multi-stability: {n_clusters} clusters, spread={spread:.3f}. "
                f"Neither clearly multi-stable nor single-attractor."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp431StabilityLandscape().execute()
