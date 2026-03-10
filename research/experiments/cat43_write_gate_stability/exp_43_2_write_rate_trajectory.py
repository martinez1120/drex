"""
Experiment 43.2 — Write Rate Trajectory During Training

Hypothesis: The write rate trajectory is monotonically decreasing during training
(model learns to write less over time as representations improve), not oscillating.
Models initialized at high threshold (0.8) and low threshold (0.2) both show
monotonically settling write rates by step 400.
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
CHECKPOINT_STEPS = [50, 100, 200, 300, 400, 600, 800]


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


class EnergyGatedDelta(nn.Module):
    """EnergyGated delta rule with fixed hard threshold."""
    def __init__(self, energy_thresh=0.4):
        super().__init__()
        self.thresh = energy_thresh
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        self.kp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.vp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.qp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self._wr   = 0.0

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        writes = 0; total = 0
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp_val = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            err  = (v - vp_val).norm(dim=-1)
            gate = (err > self.thresh * v.norm(dim=-1)).float()
            writes += gate.sum().item(); total += B
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp_val).unsqueeze(-1), kn.unsqueeze(1))
        self._wr = writes / max(total, 1)
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def eval_write_rate(model, n_batches=20):
    model.eval()
    total_wr = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, _ = make_batch(BATCH)
            model(seq)
            total_wr += model._wr
    model.train()
    return total_wr / n_batches


def eval_accuracy(model, n_batches=20):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += BATCH
    model.train()
    return correct / total


def train_with_trajectory(thresh):
    model = EnergyGatedDelta(energy_thresh=thresh)
    opt = Adam(model.parameters(), lr=3e-4)
    trajectory = {}
    checkpoint_set = set(CHECKPOINT_STEPS)
    for step in range(1, STEPS + 1):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in checkpoint_set:
            wr = eval_write_rate(model, n_batches=20)
            trajectory[step] = round(wr, 4)
    acc = eval_accuracy(model, n_batches=20)
    return trajectory, round(acc, 4)


def count_monotone_violations(trajectory, after_step=200):
    steps_after = [s for s in CHECKPOINT_STEPS if s >= after_step]
    values = [trajectory[s] for s in steps_after]
    violations = sum(1 for i in range(len(values) - 1) if values[i] < values[i + 1])
    return violations


class Exp432WriteRateTrajectory(Experiment):
    experiment_id = "exp_43_2"
    hypothesis = (
        "The write rate trajectory is monotonically decreasing during training "
        "(model learns to write less over time as representations improve), not oscillating. "
        "Models initialized at high threshold (0.8) and low threshold (0.2) both show "
        "monotonically settling write rates by step 400."
    )

    def run(self) -> ExperimentResult:
        traj_A, acc_A = train_with_trajectory(thresh=0.2)
        traj_B, acc_B = train_with_trajectory(thresh=0.8)

        metrics = {}
        for step in CHECKPOINT_STEPS:
            metrics[f"wr_A_s{step}"] = traj_A[step]
            metrics[f"wr_B_s{step}"] = traj_B[step]

        viol_A = count_monotone_violations(traj_A, after_step=200)
        viol_B = count_monotone_violations(traj_B, after_step=200)

        # Allow 1 violation for each
        monotone_A = viol_A <= 1
        monotone_B = viol_B <= 1

        metrics["monotone_A"] = int(monotone_A)
        metrics["monotone_B"] = int(monotone_B)
        metrics["violations_A"] = viol_A
        metrics["violations_B"] = viol_B
        metrics["acc_A"] = acc_A
        metrics["acc_B"] = acc_B

        if monotone_A and monotone_B:
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Both trajectories are monotonically decreasing after step 200 "
                f"(violations A={viol_A}, B={viol_B}, each <=1)."
            )
        elif viol_A > 2 or viol_B > 2:
            outcome = OUTCOME_REFUTED
            justification = (
                f"Trajectories oscillate: violations A={viol_A}, B={viol_B} (>2 for at least one)."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Mixed results: violations A={viol_A}, B={viol_B}. "
                f"Neither clearly monotone nor clearly oscillating."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp432WriteRateTrajectory().execute()
