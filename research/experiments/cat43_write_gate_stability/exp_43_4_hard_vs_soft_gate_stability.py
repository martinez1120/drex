"""
Experiment 43.4 — Hard vs Soft Write Gate Convergence Stability

Hypothesis: A hard threshold gate (binary 0/1) shows lower equilibrium write rate
variance across random seeds than a soft sigmoid gate, because the discrete nature
prevents smooth gradient-driven drift toward extreme values.
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
SEEDS = [42, 7, 13, 99, 0]
THRESH = 0.4


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


class HardGateDelta(nn.Module):
    """Delta rule with fixed hard threshold gate (binary 0/1)."""
    def __init__(self, thresh=0.4):
        super().__init__()
        self.thresh = thresh
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


class SoftGateDelta(nn.Module):
    """Delta rule with soft sigmoid gate (same threshold, differentiable)."""
    def __init__(self, thresh=0.4):
        super().__init__()
        self.thresh = thresh
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
        writes = 0.0; total = 0
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp_val = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            err  = (v - vp_val).norm(dim=-1)
            gate = torch.sigmoid((err - self.thresh * v.norm(dim=-1)) * 10.0)
            writes = writes + gate.sum().item(); total += B
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp_val).unsqueeze(-1), kn.unsqueeze(1))
        self._wr = writes / max(total, 1)
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def train_and_get_wr(model_cls, seed, steps, thresh=0.4):
    torch.manual_seed(seed)
    model = model_cls(thresh=thresh)
    opt = Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
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


def compute_variance(values):
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


class Exp434HardVsSoftGateStability(Experiment):
    experiment_id = "exp_43_4"
    hypothesis = (
        "A hard threshold gate (binary 0/1) shows lower equilibrium write rate variance "
        "across random seeds than a soft sigmoid gate, because the discrete nature "
        "prevents smooth gradient-driven drift toward extreme values."
    )

    def run(self) -> ExperimentResult:
        wr_hard_seeds = []
        wr_soft_seeds = []
        acc_hard_seeds = []
        acc_soft_seeds = []

        for seed in SEEDS:
            wr_h, acc_h = train_and_get_wr(HardGateDelta, seed=seed, steps=STEPS, thresh=THRESH)
            wr_s, acc_s = train_and_get_wr(SoftGateDelta, seed=seed, steps=STEPS, thresh=THRESH)
            wr_hard_seeds.append(wr_h)
            wr_soft_seeds.append(wr_s)
            acc_hard_seeds.append(acc_h)
            acc_soft_seeds.append(acc_s)

        var_hard = compute_variance(wr_hard_seeds)
        var_soft = compute_variance(wr_soft_seeds)
        variance_ratio = round(var_hard / var_soft, 4) if var_soft > 1e-9 else 0.0

        mean_acc_hard = round(sum(acc_hard_seeds) / len(acc_hard_seeds), 4)
        mean_acc_soft = round(sum(acc_soft_seeds) / len(acc_soft_seeds), 4)

        metrics = {
            "wr_hard_seeds": [round(v, 4) for v in wr_hard_seeds],
            "wr_soft_seeds": [round(v, 4) for v in wr_soft_seeds],
            "var_hard": round(var_hard, 6),
            "var_soft": round(var_soft, 6),
            "variance_ratio": variance_ratio,
            "mean_acc_hard": mean_acc_hard,
            "mean_acc_soft": mean_acc_soft,
        }

        if var_soft < 1e-9:
            # Soft gate has essentially zero variance — hard gate can't be 2x lower
            outcome = OUTCOME_INCONCLUSIVE
            justification = "Soft gate variance is near zero; cannot compute meaningful ratio."
        elif var_hard < var_soft * 0.5:
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Hard gate has 2x lower variance: var_hard={var_hard:.6f}, "
                f"var_soft={var_soft:.6f}, ratio={variance_ratio:.3f}<0.5."
            )
        elif var_hard >= var_soft:
            outcome = OUTCOME_REFUTED
            justification = (
                f"Soft gate is not more unstable: var_hard={var_hard:.6f} >= "
                f"var_soft={var_soft:.6f}, ratio={variance_ratio:.3f}>=1.0."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Hard gate has lower variance but not 2x: var_hard={var_hard:.6f}, "
                f"var_soft={var_soft:.6f}, ratio={variance_ratio:.3f} (need <0.5)."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp434HardVsSoftGateStability().execute()
