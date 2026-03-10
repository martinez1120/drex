from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_34_2"
hypothesis = (
    "Delta rule memory has a narrow stable learning rate band spanning "
    "<1.5 decades, indicating higher LR sensitivity than standard architectures "
    "(which typically have stable bands of 2+ decades)."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
STEPS = 500
BATCH = 32


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


class DeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.norm(self.embed(seq) + self.ff(self.embed(seq)))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom
            M = M + torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx))


def train_and_eval(lr):
    model = DeltaModel()
    opt = Adam(model.parameters(), lr=lr)
    model.train()
    final_losses = []
    for step in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= STEPS - 50:
            final_losses.append(loss.item())

    # Check for divergence
    if any(math.isnan(l) or l > 100 for l in final_losses):
        return 0.0, True  # diverged

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(40):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            pred = model(seq).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total, False


class Exp342LRSensitivity(Experiment):
    experiment_id = "exp_34_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        lr_sweep = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

        accs = {}
        diverged = {}
        for lr in lr_sweep:
            print(f"  LR={lr:.0e} ...")
            acc, div = train_and_eval(lr)
            accs[lr] = round(acc, 4)
            diverged[lr] = div
            print(f"    acc={acc:.4f}  diverged={div}")

        max_acc = max(accs.values())
        threshold = max_acc * 0.50  # stable = achieves ≥50% of peak

        stable_lrs = [lr for lr, acc in accs.items() if acc >= threshold and not diverged[lr]]
        if len(stable_lrs) >= 2:
            band_decades = math.log10(max(stable_lrs)) - math.log10(min(stable_lrs))
        else:
            band_decades = 0.0

        metrics = {f"acc_lr_{lr:.0e}".replace("+", ""): v for lr, v in accs.items()}
        metrics["max_acc"] = round(max_acc, 4)
        metrics["stable_band_decades"] = round(band_decades, 3)
        metrics["n_stable_lrs"] = len(stable_lrs)
        metrics["min_stable_lr"] = min(stable_lrs) if stable_lrs else 0.0
        metrics["max_stable_lr"] = max(stable_lrs) if stable_lrs else 0.0

        if band_decades < 1.5:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Narrow stable band: {band_decades:.2f} decades. "
                     f"Stable LRs: {stable_lrs}. High LR sensitivity confirmed.")
        elif band_decades > 2.0:
            outcome = OUTCOME_REFUTED
            notes = (f"Wide stable band: {band_decades:.2f} decades > 2.0. Low LR sensitivity.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Moderate stable band: {band_decades:.2f} decades (threshold: <1.5). "
                     f"Stable LRs: {stable_lrs}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM, steps=STEPS))


if __name__ == "__main__":
    Exp342LRSensitivity().execute()
