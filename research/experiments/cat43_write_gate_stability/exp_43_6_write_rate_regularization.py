"""
Experiment 43.6 — Write Rate Regularization

Hypothesis: Adding a soft write-rate regularization loss (lambda=0.1 x |wr - 0.5|^2)
during training converges all initializations to write rate ~0.5 +/- 0.05
regardless of initial threshold, reducing equilibrium spread to <0.15.
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
INIT_THRESHOLDS = [0.1, 0.3, 0.5, 0.8, 1.2]
LAMBDA = 0.1
TARGET_WR = 0.5


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


class LearnableThresholdDeltaReg(nn.Module):
    """LearnableThresholdDelta with differentiable gate mean for regularization."""
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
        self._gate_mean = None  # differentiable gate mean for regularization

    @property
    def threshold(self):
        return self.log_thresh.exp()

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        gate_accum = []
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp_val = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            err = (v - vp_val).norm(dim=-1)
            thresh = self.threshold
            gate = torch.sigmoid((err - thresh * v.norm(dim=-1)) * 10.0)
            gate_accum.append(gate)
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp_val).unsqueeze(-1), kn.unsqueeze(1))
        # gate_accum: list of (B,) tensors, shape (L-1, B)
        all_gates = torch.stack(gate_accum, dim=0)  # (L-1, B)
        self._gate_mean = all_gates.mean()  # scalar, differentiable
        self._wr = self._gate_mean.item()
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def train_model(init_threshold, use_regularization, steps):
    model = LearnableThresholdDeltaReg(init_threshold=init_threshold)
    opt = Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        ce_loss = F.cross_entropy(logits, tgt)
        if use_regularization and model._gate_mean is not None:
            reg_loss = LAMBDA * (model._gate_mean - TARGET_WR) ** 2
            loss = ce_loss + reg_loss
        else:
            loss = ce_loss
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    wr_total = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += BATCH
            wr_total += model._wr
    avg_wr = wr_total / 20
    acc = correct / total
    return round(avg_wr, 4), round(acc, 4)


class Exp436WriteRateRegularization(Experiment):
    experiment_id = "exp_43_6"
    hypothesis = (
        "Adding a soft write-rate regularization loss (lambda=0.1 x |wr - 0.5|^2) "
        "during training converges all initializations to write rate ~0.5 +/- 0.05 "
        "regardless of initial threshold, reducing equilibrium spread to <0.15."
    )

    def run(self) -> ExperimentResult:
        wr_reg = []
        wr_unreg = []
        acc_reg_list = []
        acc_unreg_list = []

        for t in INIT_THRESHOLDS:
            wr_r, acc_r = train_model(t, use_regularization=True, steps=STEPS)
            wr_u, acc_u = train_model(t, use_regularization=False, steps=STEPS)
            wr_reg.append(wr_r)
            wr_unreg.append(wr_u)
            acc_reg_list.append(acc_r)
            acc_unreg_list.append(acc_u)

        spread_reg = max(wr_reg) - min(wr_reg)
        spread_unreg = max(wr_unreg) - min(wr_unreg)
        mean_wr_reg = sum(wr_reg) / len(wr_reg)
        acc_reg_mean = sum(acc_reg_list) / len(acc_reg_list)
        acc_unreg_mean = sum(acc_unreg_list) / len(acc_unreg_list)

        metrics = {
            "wr_reg": [round(v, 4) for v in wr_reg],
            "wr_unreg": [round(v, 4) for v in wr_unreg],
            "spread_reg": round(spread_reg, 4),
            "spread_unreg": round(spread_unreg, 4),
            "mean_wr_reg": round(mean_wr_reg, 4),
            "acc_reg": round(acc_reg_mean, 4),
            "acc_unreg": round(acc_unreg_mean, 4),
        }

        if spread_reg < 0.15 and spread_unreg > 0.5:
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Regularization works: spread_reg={spread_reg:.3f}<0.15 "
                f"and spread_unreg={spread_unreg:.3f}>0.5."
            )
        elif spread_reg >= 0.5:
            outcome = OUTCOME_REFUTED
            justification = (
                f"Regularization does not help: spread_reg={spread_reg:.3f}>=0.5."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Partial effect: spread_reg={spread_reg:.3f} (need <0.15), "
                f"spread_unreg={spread_unreg:.3f} (need >0.5). "
                f"mean_wr_reg={mean_wr_reg:.3f}."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp436WriteRateRegularization().execute()
