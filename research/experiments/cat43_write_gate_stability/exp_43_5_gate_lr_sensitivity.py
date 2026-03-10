"""
Experiment 43.5 — Gate Learning Rate Sensitivity

Hypothesis: Training the gate threshold with a 10x lower learning rate than model
weights reduces the equilibrium spread (across initializations) from >1.0 (exp_39_3)
to <0.30, stabilizing convergence by preventing rapid gate adaptation.
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
LR_MODEL = 3e-4
LR_GATE_RATIOS = [1.0, 0.1]


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


def get_non_thresh_params(model):
    return [p for name, p in model.named_parameters() if name != "log_thresh"]


def train_model(init_threshold, lr_gate_ratio, steps):
    model = LearnableThresholdDelta(init_threshold=init_threshold)
    opt = Adam(
        [
            {"params": [model.log_thresh], "lr": LR_MODEL * lr_gate_ratio},
            {"params": get_non_thresh_params(model), "lr": LR_MODEL},
        ]
    )
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    acc_correct = 0; acc_total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            acc_correct += (preds == tgt).sum().item()
            acc_total += BATCH
    acc = acc_correct / acc_total
    return model.threshold.item(), round(acc, 4)


class Exp435GateLrSensitivity(Experiment):
    experiment_id = "exp_43_5"
    hypothesis = (
        "Training the gate threshold with a 10x lower learning rate than model weights "
        "reduces the equilibrium spread (across initializations) from >1.0 (exp_39_3) "
        "to <0.30, stabilizing convergence by preventing rapid gate adaptation."
    )

    def run(self) -> ExperimentResult:
        results = {}
        for ratio in LR_GATE_RATIOS:
            final_thresholds = []
            final_accs = []
            for t in INIT_THRESHOLDS:
                final_thresh, acc = train_model(t, lr_gate_ratio=ratio, steps=STEPS)
                final_thresholds.append(final_thresh)
                final_accs.append(acc)
            spread = max(final_thresholds) - min(final_thresholds)
            results[ratio] = {
                "thresholds": final_thresholds,
                "spread": spread,
                "accs": final_accs,
            }

        spread_full_lr = results[1.0]["spread"]
        spread_low_lr = results[0.1]["spread"]
        reduction_factor = round(spread_full_lr / spread_low_lr, 4) if spread_low_lr > 1e-6 else float("inf")

        # Format ratio key as "1_0" and "0_1"
        metrics = {
            "spread_ratio_1_0": round(spread_full_lr, 4),
            "spread_ratio_0_1": round(spread_low_lr, 4),
            "reduction_factor": round(reduction_factor, 4),
            "final_thresholds_ratio_0_1": [round(t, 4) for t in results[0.1]["thresholds"]],
        }

        if spread_low_lr < 0.30 and spread_low_lr < spread_full_lr * 0.5:
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Low LR gate reduces spread: spread_low_lr={spread_low_lr:.3f}<0.30 "
                f"and is {reduction_factor:.2f}x smaller than spread_full_lr={spread_full_lr:.3f}."
            )
        elif spread_low_lr >= spread_full_lr * 0.8:
            outcome = OUTCOME_REFUTED
            justification = (
                f"Low LR does not help: spread_low_lr={spread_low_lr:.3f} >= "
                f"spread_full_lr*0.8={spread_full_lr*0.8:.3f}."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Partial reduction: spread_low_lr={spread_low_lr:.3f}, "
                f"spread_full_lr={spread_full_lr:.3f}, reduction_factor={reduction_factor:.2f}. "
                f"Did not meet both criteria (spread<0.30 AND <50% of full LR spread)."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp435GateLrSensitivity().execute()
