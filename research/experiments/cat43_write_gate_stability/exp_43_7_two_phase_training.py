"""
Experiment 43.7 — Two-Phase Training for Write Gate Stability

Hypothesis: First freeze the threshold (train only model weights), then unfreeze
(fine-tune threshold for 200 steps) reduces the equilibrium spread to <0.30 compared
to joint training from the start (spread~1.022 from exp_39_3).
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
STEPS_PHASE1 = 600
STEPS_PHASE2 = 200
INIT_THRESHOLDS = [0.1, 0.3, 0.5, 0.8, 1.2]


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


def train_joint(init_threshold):
    """Train everything jointly for STEPS_PHASE1 + STEPS_PHASE2 steps."""
    total_steps = STEPS_PHASE1 + STEPS_PHASE2
    model = LearnableThresholdDelta(init_threshold=init_threshold)
    opt = Adam(model.parameters(), lr=3e-4)
    for _ in range(total_steps):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += BATCH
    acc = correct / total
    return model.threshold.item(), round(acc, 4)


def train_two_phase(init_threshold):
    """Phase 1: freeze threshold. Phase 2: unfreeze threshold."""
    model = LearnableThresholdDelta(init_threshold=init_threshold)

    # Phase 1: train everything EXCEPT log_thresh
    model.log_thresh.requires_grad_(False)
    opt1 = Adam(get_non_thresh_params(model), lr=3e-4)
    for _ in range(STEPS_PHASE1):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt1.zero_grad(); loss.backward(); opt1.step()

    # Phase 2: unfreeze and train all parameters
    model.log_thresh.requires_grad_(True)
    opt2 = Adam(model.parameters(), lr=3e-4)
    for _ in range(STEPS_PHASE2):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt2.zero_grad(); loss.backward(); opt2.step()

    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(20):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += BATCH
    acc = correct / total
    return model.threshold.item(), round(acc, 4)


class Exp437TwoPhaseTraining(Experiment):
    experiment_id = "exp_43_7"
    hypothesis = (
        "First freeze the threshold (train only model weights), then unfreeze "
        "(fine-tune threshold for 200 steps) reduces the equilibrium spread to <0.30 "
        "compared to joint training from the start (spread~1.022 from exp_39_3)."
    )

    def run(self) -> ExperimentResult:
        joint_thresholds = []
        joint_accs = []
        two_phase_thresholds = []
        two_phase_accs = []

        for t in INIT_THRESHOLDS:
            jt, ja = train_joint(t)
            tp_t, tp_a = train_two_phase(t)
            joint_thresholds.append(jt)
            joint_accs.append(ja)
            two_phase_thresholds.append(tp_t)
            two_phase_accs.append(tp_a)

        spread_joint = max(joint_thresholds) - min(joint_thresholds)
        spread_two_phase = max(two_phase_thresholds) - min(two_phase_thresholds)
        acc_joint_mean = sum(joint_accs) / len(joint_accs)
        acc_two_phase_mean = sum(two_phase_accs) / len(two_phase_accs)

        metrics = {
            "spread_joint": round(spread_joint, 4),
            "spread_two_phase": round(spread_two_phase, 4),
            "final_thresholds_joint": [round(t, 4) for t in joint_thresholds],
            "final_thresholds_two_phase": [round(t, 4) for t in two_phase_thresholds],
            "acc_joint": round(acc_joint_mean, 4),
            "acc_two_phase": round(acc_two_phase_mean, 4),
        }

        if spread_two_phase < 0.30 and spread_two_phase < spread_joint * 0.5:
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Two-phase training reduces spread: spread_two_phase={spread_two_phase:.3f}<0.30 "
                f"and is less than 50% of spread_joint={spread_joint:.3f}."
            )
        elif spread_two_phase >= spread_joint * 0.8:
            outcome = OUTCOME_REFUTED
            justification = (
                f"Two-phase training does not help: spread_two_phase={spread_two_phase:.3f} >= "
                f"spread_joint*0.8={spread_joint*0.8:.3f}."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Partial improvement: spread_two_phase={spread_two_phase:.3f}, "
                f"spread_joint={spread_joint:.3f}. "
                f"Did not meet both criteria (spread<0.30 AND <50% of joint spread)."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp437TwoPhaseTraining().execute()
