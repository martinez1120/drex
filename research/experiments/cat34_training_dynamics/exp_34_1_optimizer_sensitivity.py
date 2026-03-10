from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

experiment_id = "exp_34_1"
hypothesis = (
    "Delta rule memory architecture shows strong optimizer preference: "
    "best optimizer outperforms worst by >10% accuracy, indicating "
    "sensitivity greater than typical for standard architectures."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
STEPS = 600
BATCH = 32
DEVICE = "cpu"


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
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x)
        return self.norm(h + self.ff(h))


class DeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.encoder(seq)
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


def train_and_eval(optimizer_name, lr, weight_decay=0.0):
    model = DeltaModel()
    if optimizer_name == "Adam":
        opt = Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        opt = SGD(model.parameters(), lr=lr)
    elif optimizer_name == "SGD_momentum":
        opt = SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "RMSprop":
        opt = RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(optimizer_name)

    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            pred = model(seq).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp341OptimizerSensitivity(Experiment):
    experiment_id = "exp_34_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        configs = [
            ("Adam",         3e-4, 0.0),
            ("AdamW",        3e-4, 0.01),
            ("SGD",          1e-2, 0.0),
            ("SGD_momentum", 5e-3, 0.0),
            ("RMSprop",      1e-3, 0.0),
        ]

        results = {}
        for name, lr, wd in configs:
            print(f"  Training {name}  lr={lr}  wd={wd} ...")
            acc = train_and_eval(name, lr, wd)
            results[name] = round(acc, 4)
            print(f"    acc={acc:.4f}")

        accs = list(results.values())
        spread = round(max(accs) - min(accs), 4)
        best = max(results, key=results.get)
        worst = min(results, key=results.get)

        metrics = {f"acc_{k}": v for k, v in results.items()}
        metrics["spread_max_min"] = spread
        metrics["best_optimizer"] = best
        metrics["worst_optimizer"] = worst

        if spread > 0.10:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Strong optimizer preference: spread={spread:.3f}>0.10. "
                     f"Best={best}({results[best]:.3f}), Worst={worst}({results[worst]:.3f}).")
        elif spread < 0.03:
            outcome = OUTCOME_REFUTED
            notes = (f"All optimizers within {spread:.3f}<0.03 — no strong preference.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Moderate spread={spread:.3f}. "
                     f"Best={best}({results[best]:.3f}), Worst={worst}({results[worst]:.3f}).")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM, steps=STEPS, batch=BATCH))


if __name__ == "__main__":
    Exp341OptimizerSensitivity().execute()
