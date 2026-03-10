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

experiment_id = "exp_34_3"
hypothesis = (
    "Delta rule memory quality degrades at larger batch sizes independently of "
    "gradient noise: accuracy at B=128 is >5% lower than B=8 even when "
    "effective LR is scaled proportionally (linear scaling rule)."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
BASE_LR = 3e-4
BASE_BATCH = 8
# Total gradient steps held constant; larger batches see more data per step
TOTAL_STEPS = 600


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
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
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


def train_and_eval(batch_size, lr, steps):
    model = DeltaModel()
    opt = Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_assoc_batch(32, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            pred = model(seq).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp343BatchSizeInteraction(Experiment):
    experiment_id = "exp_34_3"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        batch_sizes = [4, 8, 16, 32, 64, 128]

        accs = {}
        for B in batch_sizes:
            # Linear scaling rule: scale LR proportionally to batch size
            lr = BASE_LR * (B / BASE_BATCH)
            print(f"  B={B}  lr={lr:.5f} ...")
            acc = train_and_eval(B, lr, TOTAL_STEPS)
            accs[B] = round(acc, 4)
            print(f"    acc={acc:.4f}")

        # Compare B=8 vs B=128
        acc_small = accs.get(8, accs[min(accs)])
        acc_large = accs.get(128, accs[max(accs)])
        drop = round(acc_small - acc_large, 4)

        # Check if relationship is non-monotonic or shows systematic drop
        acc_list = [accs[b] for b in sorted(accs)]
        # Correlation: negative correlation (larger batch = lower acc) would support
        import statistics
        bs_list = sorted(accs.keys())
        if len(bs_list) >= 3:
            rank_b = list(range(len(bs_list)))
            rank_a = sorted(range(len(acc_list)), key=lambda i: acc_list[i])
            # simple rank correlation
            n = len(bs_list)
            d2 = sum((rank_b[i] - rank_a.index(i))**2 for i in range(n))
            rho = 1 - 6 * d2 / (n * (n**2 - 1))
        else:
            rho = 0.0

        metrics = {f"acc_B{b}": v for b, v in accs.items()}
        metrics["drop_B8_to_B128"] = drop
        metrics["spearman_batch_vs_acc"] = round(rho, 3)
        metrics["acc_B8"] = accs.get(8, 0)
        metrics["acc_B128"] = accs.get(128, 0)

        if drop > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Larger batch hurts: B8={acc_small:.3f} vs B128={acc_large:.3f}, "
                     f"drop={drop:.3f}>0.05. Memory quality degrades with batch.")
        elif drop < 0.01:
            outcome = OUTCOME_REFUTED
            notes = (f"No batch sensitivity: drop={drop:.3f}<0.01. "
                     f"Memory quality scales normally with batch size.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Moderate drop B8→B128: {drop:.3f}. Spearman ρ={rho:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM, steps=TOTAL_STEPS,
                                       base_lr=BASE_LR, base_batch=BASE_BATCH))


if __name__ == "__main__":
    Exp343BatchSizeInteraction().execute()
