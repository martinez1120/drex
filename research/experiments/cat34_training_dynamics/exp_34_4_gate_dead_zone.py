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

experiment_id = "exp_34_4"
hypothesis = (
    "At convergence, more than 40% of learned write gate activations are "
    "in the dead zone (<0.05) or saturated zone (>0.95), indicating bimodal "
    "gate collapse rather than graded, informative gating."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
STEPS = 600
BATCH = 32
EVAL_BATCHES = 200  # batches to collect gate statistics


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 3, (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


class LearnedGateDeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.gate_net = nn.Sequential(nn.Linear(HIDDEN_DIM, 16), nn.ReLU(),
                                       nn.Linear(16, 1), nn.Sigmoid())
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq, collect_gates=False):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        gate_vals = []
        for t in range(L - 1):
            k = h[:, t, :]
            g = self.gate_net(k)  # (B, 1) in [0,1]
            if collect_gates:
                gate_vals.append(g.detach().squeeze(-1))
            v_pred = torch.bmm(M.detach(), k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom
            dM = torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
            M = M + g.unsqueeze(-1) * dM
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        logits = self.output(self.read_proj(ctx))
        if collect_gates:
            return logits, torch.cat(gate_vals, dim=0)
        return logits, None


def compute_gate_entropy(gate_vals_flat):
    """Approximate entropy by binning into 20 bins."""
    hist = torch.histc(gate_vals_flat, bins=20, min=0.0, max=1.0)
    hist = hist / hist.sum()
    # Remove zero bins
    hist = hist[hist > 0]
    entropy = -(hist * hist.log()).sum().item()
    max_entropy = torch.tensor(20.0).log().item()
    return entropy / max_entropy  # normalized


class Exp344GateDeadZone(Experiment):
    experiment_id = "exp_34_4"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        model = LearnedGateDeltaModel()
        opt = Adam(model.parameters(), lr=3e-4)

        model.train()
        for _ in range(STEPS):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            logits, _ = model(seq)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

        # Eval accuracy
        model.eval()
        correct = total = 0
        all_gates = []
        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
                logits, gates = model(seq, collect_gates=True)
                pred = logits.argmax(-1)
                correct += (pred == tgt).sum().item()
                total += tgt.size(0)
                all_gates.append(gates)

        acc = correct / total
        gate_flat = torch.cat(all_gates)

        dead = (gate_flat < 0.05).float().mean().item()
        saturated = (gate_flat > 0.95).float().mean().item()
        bimodal_fraction = dead + saturated
        gate_mean = gate_flat.mean().item()
        gate_std = gate_flat.std().item()
        entropy_norm = compute_gate_entropy(gate_flat)

        metrics = dict(
            acc=round(acc, 4),
            gate_dead_frac=round(dead, 4),
            gate_saturated_frac=round(saturated, 4),
            bimodal_fraction=round(bimodal_fraction, 4),
            gate_mean=round(gate_mean, 4),
            gate_std=round(gate_std, 4),
            gate_entropy_normalized=round(entropy_norm, 4),
            total_gate_samples=gate_flat.numel(),
        )

        if bimodal_fraction > 0.40:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Gate is bimodal: dead={dead:.3f} + saturated={saturated:.3f} "
                     f"= {bimodal_fraction:.3f} > 0.40. Gate entropy={entropy_norm:.3f}.")
        elif bimodal_fraction < 0.15:
            outcome = OUTCOME_REFUTED
            notes = (f"Gate is well-spread: bimodal_fraction={bimodal_fraction:.3f} < 0.15. "
                     f"Entropy={entropy_norm:.3f}. No dead/saturation collapse.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Moderate bimodal fraction={bimodal_fraction:.3f} "
                     f"(threshold 0.40). Entropy={entropy_norm:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, eval_batches=EVAL_BATCHES))


if __name__ == "__main__":
    Exp344GateDeadZone().execute()
