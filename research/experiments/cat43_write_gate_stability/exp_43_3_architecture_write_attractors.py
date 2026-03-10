"""
Experiment 43.3 — Architecture-Specific Write Rate Attractors

Hypothesis: Different memory architectures converge to distinct write-rate equilibria:
DeltaRule converges to wr≈0.95 (writes almost always), EnergyGated converges to
wr≈0.50, and SoftGatedDelta converges to an intermediate wr≈0.70.
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
STEPS = 1000


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


class PlainDeltaRule(nn.Module):
    """Delta rule that writes every step — effective write rate = 1.0 by definition."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        self.kp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.vp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.qp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self._wr   = 1.0

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp_val = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            # Always write (gate=1)
            M = M + torch.bmm((v - vp_val).unsqueeze(-1), kn.unsqueeze(1))
        self._wr = 1.0
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


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


class SoftLearnedGateDelta(nn.Module):
    """Delta rule with learned sigmoid gate per token."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Linear(128, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        self.kp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.vp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.qp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.gate_linear = nn.Linear(HIDDEN_DIM, 1)
        self._wr   = 0.0

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        gate_sum = 0.0; total = 0
        for t in range(L - 1):
            k = self.kp(h[:, t, :]); v = self.vp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp_val = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            # Per-token learned gate
            gate = torch.sigmoid(self.gate_linear(h[:, t, :])).squeeze(-1)  # (B,)
            gate_sum = gate_sum + gate.sum().item(); total += B
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp_val).unsqueeze(-1), kn.unsqueeze(1))
        self._wr = gate_sum / max(total, 1)
        q = self.qp(h[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def train_and_eval(model, steps, n_eval_batches=60):
    opt = Adam(model.parameters(), lr=3e-4)
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = 0; total = 0; wr_total = 0.0
    with torch.no_grad():
        for _ in range(n_eval_batches):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt).sum().item()
            total += BATCH
            wr_total += model._wr
    acc = correct / total
    avg_wr = wr_total / n_eval_batches
    return round(avg_wr, 4), round(acc, 4)


class Exp433ArchitectureWriteAttractors(Experiment):
    experiment_id = "exp_43_3"
    hypothesis = (
        "Different memory architectures converge to distinct write-rate equilibria: "
        "DeltaRule converges to wr≈0.95, EnergyGated converges to wr≈0.50, "
        "and SoftGatedDelta converges to an intermediate wr≈0.70."
    )

    def run(self) -> ExperimentResult:
        model_plain = PlainDeltaRule()
        model_energy = EnergyGatedDelta(energy_thresh=0.4)
        model_soft = SoftLearnedGateDelta()

        wr_plain, acc_plain = train_and_eval(model_plain, STEPS)
        wr_energy, acc_energy = train_and_eval(model_energy, STEPS)
        wr_soft, acc_soft = train_and_eval(model_soft, STEPS)

        wr_spread = round(max(wr_plain, wr_energy, wr_soft) - min(wr_plain, wr_energy, wr_soft), 4)

        metrics = {
            "wr_plain": wr_plain,
            "wr_energy": wr_energy,
            "wr_soft": wr_soft,
            "acc_plain": acc_plain,
            "acc_energy": acc_energy,
            "acc_soft": acc_soft,
            "wr_spread": wr_spread,
        }

        diff_energy_soft = abs(wr_energy - wr_soft)
        diff_energy_plain = abs(wr_energy - wr_plain)
        diff_all_min = min(diff_energy_soft, diff_energy_plain, abs(wr_soft - wr_plain))
        diff_all_max = max(diff_energy_soft, diff_energy_plain, abs(wr_soft - wr_plain))

        if diff_energy_soft > 0.15:
            outcome = OUTCOME_SUPPORTED
            justification = (
                f"Distinct equilibria confirmed: |wr_energy-wr_soft|={diff_energy_soft:.3f}>0.15. "
                f"wr_plain={wr_plain:.3f}, wr_energy={wr_energy:.3f}, wr_soft={wr_soft:.3f}."
            )
        elif diff_all_max < 0.05:
            outcome = OUTCOME_REFUTED
            justification = (
                f"All architectures converge to same write rate: "
                f"max pairwise diff={diff_all_max:.3f}<0.05."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            justification = (
                f"Partial separation: |wr_energy-wr_soft|={diff_energy_soft:.3f}, "
                f"spread={wr_spread:.3f}. Differences present but not >0.15 threshold."
            )

        return self.result(outcome, metrics, justification)


if __name__ == "__main__":
    Exp433ArchitectureWriteAttractors().execute()
