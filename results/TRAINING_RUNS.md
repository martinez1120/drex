# results/TRAINING_RUNS.md — Drex Experiment A/B Baseline Comparison

**Purpose:** End-to-end comparison of DrexTransformer with and without episodic memory
(`MemoryModule`) on TinyStories, evaluated on passkey recall and BABILong. This is the
first published training result for the Phase 13–15 validated architecture.

---

## Setup

### Hardware

| Field | Value |
|---|---|
| Machine | MacBook Air, Apple M3, 16 GB |
| Compute | MPS |
| PyTorch | 2.8.0 |
| Python | 3.12.12 |
| Date | 2026-03-12 |

### Data

TinyStories (roneneldan/TinyStories on HuggingFace Hub).

- Train split: ~50M characters (~2.1M stories)
- Validation split: ~500k characters cap (--val-max-chars 500000)
- Tokenisation: char-level, vocab size 256 (raw byte values)
- Segment length: 512 tokens TBPTT chunks

If direct download is blocked by SSL (self-signed certificate error):

```bash
# Option 1: disable SSL verification for the download session (dev only)
# Pass --no-ssl-verify to train.py

# Option 2: download TinyStories as a local text file first
python3 -c "
import ssl, httpx, sys
# patch SSL (dev only)
import httpx as hx
class _C(hx.Client):
    def __init__(self, *a, **kw): kw['verify']=False; super().__init__(*a,**kw)
hx.Client = _C
class _CA(hx.AsyncClient):
    def __init__(self, *a, **kw): kw['verify']=False; super().__init__(*a,**kw)
hx.AsyncClient = _CA
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
with open('/tmp/tinystories_train.txt', 'w') as f:
    for row in ds: f.write(row['text'] + '\n')
print('done')
"
# Then use: python scripts/train.py --data-file /tmp/tinystories_train.txt ...
```

---

## Experiment A — Baseline (no episodic memory)

DrexTransformer without MemoryModule. This is the control condition.

### Config

| Parameter | Value |
|---|---|
| d_model | 256 |
| n_layers | 4 |
| n_heads | 4 |
| ff_mult | 4 |
| vocab_size | 256 (char-level) |
| window_size | 512 |
| batch_size | 8 |
| segment_len | 512 |
| dropout | 0.1 |
| lr | 3e-4 |
| warmup_steps | 2000 |
| steps | 50000 |
| grad_clip | 1.0 |
| use_episodic_memory | False |
| reset_on_boundary | True |
| seed | 42 |
| ~parameters | 4,264,464 (measured) |

### Command

```bash
PYTHONPATH=python python3.12 scripts/train.py \
  --steps 50000 --log-every 200 \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --batch-size 8 --segment-len 512 --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
  --val-every 1000 --val-max-chars 500000 \
  --reset-on-boundary \
  --seed 42 \
  --ckpt-dir checkpoints/exp_a \
  --save-every 5000 \
  --no-ssl-verify \
  2>&1 | tee results/exp_a_train.log
```

### Training log

| Metric | Value |
|---|---|
| Status | **IN PROGRESS** — step 16,800 / 50,000 (2026-03-13) |
| Best val_ppl so far | **1.82** at step 16,000 (resumed run, correct LR) |
| Steady-state throughput | ~13,000–15,000 tok/s (MPS, resumed from step 15k) |
| NaN skips total | 0 |
| Parameters | 4,264,464 |

**Checkpoint resume fix (Phase 16):** Run originally stopped at step 17,200 due to
session termination. Resume bug (LR restarted from 3e-05 instead of ~2.53e-04) was
diagnosed and fixed — optimizer and scheduler state now saved to `_opt.pt` companion
file. Full training resumed from step 15,000 checkpoint with correct cosine-decay LR.

**Partial val_ppl trajectory (to be completed at step 50k):**

| Step | val_ppl | Notes |
|---|---|---|
| 4,000 | 3.83 | |
| 9,000 | 3.57 | |
| 12,000 | 2.52 | |
| 14,000 | 2.82 | |
| 16,000 | **1.82** | resumed run (corrected LR) |

---

## Experiment B — Episodic Memory (thresh\*=0.70)

DrexTransformer with MemoryModule enabled. Identical config to Exp A except
`--use-episodic-memory --episodic-gate-thresh 0.70`.

### Config

Same as Experiment A, plus:

| Parameter | Value |
|---|---|
| use_episodic_memory | True |
| episodic_gate_thresh | 0.70 (thresh* confirmed exp_48_1) |
| seed | 42 |
| parameters | 4,794,900 (measured; +2,048 for norm_out LayerNorm vs earlier estimate) |

### Command

```bash
PYTHONPATH=python python3.12 scripts/train.py \
  --steps 50000 --log-every 200 \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --batch-size 8 --segment-len 512 --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --val-every 1000 --val-max-chars 500000 \
  --reset-on-boundary \
  --seed 42 \
  --ckpt-dir checkpoints/exp_b \
  --save-every 5000 \
  --no-ssl-verify \
  2>&1 | tee results/exp_b_train.log
```

### Training log

**Phase 16 fix:** Three-part write loop fix applied — CPU backend (move M/keys to CPU),
detached write (`torch.no_grad()` + `.detach()`), output LayerNorm (`norm_out`).
Throughput recovered from **543 → 2,310 tok/s** (4.3× improvement).

| Metric | Value |
|---|---|
| Final train loss | **0.3169** (step 2000) |
| Final train ppl | **1.37** (step 2000) |
| Final val loss | **1.4522** (step 2000) |
| Final val ppl | **4.27** (step 2000) |
| Throughput at step 200, pre-fix seg_len=512 | **543 tok/s** (20× slower than baseline) |
| Throughput at step 200, post-fix seg_len=512 | **2,310 tok/s** (4.3× improvement, 5× slower than baseline) |
| Throughput at step 400, post-fix seg_len=512 | **2,377 tok/s** (consistent) |
| Throughput at step 2000, post-fix seg_len=512 | **3,074 tok/s** (peak; cosine LR tail) |
| Write rate at step 200 (pre-fix, original run) | **0.969** — outside [0.10, 0.85] |
| Write rate at step 200 (post-fix probe) | **0.987** [0.746, 1.000] — outside target |
| Write rate at step 1000 (post-fix probe) | **0.968** [0.913, 0.993] — plateau beginning |
| Write rate at step 2000 (post-fix probe) | **0.963** [0.911, 0.986] — **plateau confirmed; NOT converging within 2k steps** |
| Write rate at seg_len=64 | 0.456–0.491 ✓ |
| Parameters (with norm_out LayerNorm) | **4,794,900** (+2,048 vs original) |

> **Throughput fix (Phase 16 — completed):** Three-part fix implemented and committed.
>
> | Fix | Throughput | Notes |
> |---|---|---|
> | Original (MPS sequential loop) | 543 tok/s | 20× slower than baseline |
> | CPU backend without detach | ~543–600 tok/s | Bottleneck shifted to O(L) autograd |
> | CPU backend with detach (no_grad) | ~1,158 tok/s | Python loop overhead remains |
> | CPU + detach + norm_out | **2,310 tok/s** | 4.3× improvement; 5× below Exp A baseline |
>
> Remaining gap to baseline (11,700 tok/s): Python interpreter overhead at 511 iterations × 4 layers × ~15 ops ≈ 30,660 Python/PyTorch calls per step. Full elimination requires parallel scan or custom Metal kernel.
>
> **α calibration note (write rate at L=512 — CONFIRMED):** The `α(L)` formula (`α = 0.95^(96/L)`)
> gives α≈0.990 at L=512. With `(1−α)=0.010`, each delta-rule update is only 1% of its
> full magnitude. Early in training the matrices are near-zero, so `vps ≈ 0` and the
> prediction error `||ks − vps|| = ||ks||` almost always exceeds `thresh × ||ks||` when
> thresh=0.70 < 1. This forces wr≈1.0 until the matrices populate enough to match keys.
> **Confirmed (Exp B 2000-step probe): wr plateau at 0.987 → 0.972 → 0.968 → 0.965 → 0.963
> through all 2000 steps. Write rate does NOT converge to [0.10, 0.85] within 2000 steps at L=512.**
> The earlier estimate of convergence by ~5000 steps was optimistic. Extended training
> (≥10k steps) is required to determine if the write rate converges to spec.
> The α formula was validated at L=32 (wr=0.581) and L=96 (wr=0.421); extrapolation to
> L=512 overestimated convergence speed.

---

## Evaluation

Run after both training experiments complete.

### Passkey Recall

```bash
# Experiment A baseline
PYTHONPATH=python python3.12 scripts/eval_passkey.py \
  --checkpoint checkpoints/exp_a/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --lengths 512 1024 2048 4096 8192 16384 \
  --trials 20 \
  --device mps \
  2>&1 | tee results/exp_a_passkey.log

# Experiment B episodic memory
PYTHONPATH=python python3.12 scripts/eval_passkey.py \
  --checkpoint checkpoints/exp_b/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --report-write-rate \
  --lengths 512 1024 2048 4096 8192 16384 \
  --trials 20 \
  --device mps \
  2>&1 | tee results/exp_b_passkey.log
```

### BABILong

```bash
# Experiment A baseline
PYTHONPATH=python python3.12 scripts/eval_babilong.py \
  --checkpoint checkpoints/exp_a/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --lengths 2048 4096 8192 \
  --tasks 1 2 3 4 5 --trials 20 \
  --device mps \
  2>&1 | tee results/exp_a_babilong.log

# Experiment B episodic memory
PYTHONPATH=python python3.12 scripts/eval_babilong.py \
  --checkpoint checkpoints/exp_b/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --report-write-rate \
  --lengths 2048 4096 8192 \
  --tasks 1 2 3 4 5 --trials 20 \
  --device mps \
  2>&1 | tee results/exp_b_babilong.log
```

---

## Results Table

**Status: PENDING.** Exp A full 50k training in progress (step 16,800 as of 2026-03-13).
Exp B will auto-start via `run_exp_b.sh` when Exp A final checkpoint appears. Passkey
and BABILong evaluations planned once both final checkpoints are ready (Phase 17 Step 2).

### Passkey Recall Accuracy

| Model | 512 ctx | 1k ctx | 2k ctx | 4k ctx | 8k ctx | 16k ctx |
|---|---|---|---|---|---|---|
| Exp A (baseline) | pending | pending | pending | pending | pending | pending |
| Exp B (episodic) | pending | pending | pending | pending | pending | pending |
| Δ (B − A) | — | — | — | — | — | — |

### BABILong Accuracy at 8k context

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Mean |
|---|---|---|---|---|---|---|
| Exp A (baseline) | pending | pending | pending | pending | pending | pending |
| Exp B (episodic) | pending | pending | pending | pending | pending | pending |

### Write Rate Validation (Exp B, seg_len=64 throughput probe)

| Segment Length | mean_wr | min_wr (per batch) | max_wr (per batch) | In range? |
|---|---|---|---|---|
| 64 | 0.456–0.491 | 0.000 | 1.000 | ✓ (overall mean; per-element extremes expected at short L) |

*min_wr=0.000 and max_wr=1.000 per batch element are expected at L=64: some elements
produce all-novel keys (every write fires) and some produce all-familiar keys (no write
fires). The gate is operating correctly. The mean across all batch elements stays within
[0.10, 0.85] from step 20 onward.*

---

## Interpretation Guide

**What a successful result looks like:**
- Exp B outperforms Exp A on passkey recall specifically at context lengths ≥ 2k (where
  the passkey reward is outside the 512-token sliding window)
- Exp B shows any improvement on BABILong Task 5 (count after drop) — the hardest task
  requiring multi-step reasoning over distractors
- Write rates at eval time fall within [0.10, 0.85]

**What a null result would mean:**
- If Exp A ≈ Exp B on both benchmarks: episodic memory provides no benefit at this
  model size / training scale. This is important to know before the paper goes out.
  It would motivate ablations at larger model sizes or different training data.

**What a concerning result would look like:**
- Write rate outside [0.10, 0.85] at eval time (after training): the trained model may
  have learned to suppress or saturate the gate. Check with `--report-write-rate`.
- Val loss significantly worse for Exp B than Exp A: MemoryModule may be adding
  harmful gradient noise at this scale. Check NaN skip counts in the training log.

---

## Ablations (Phase 16)

After Exp A/B complete, run these ablations to elevate medium-confidence components
(§12.2 in ARCHITECTURE_FINDINGS.md) to high confidence or refute them:

```bash
# Ablation C — no null retrieval gate
# Requires: add --no-null-gate flag to train.py (not yet implemented)

# Ablation D — full-sequence residual (not last-token-only)
# Requires: change x[:, -1] = x[:, -1] + mem_r to a full-sequence residual

# Ablation E — L4 only (disable L2 InfiniAttention)
# Requires: add --no-l2 flag to train.py
```
