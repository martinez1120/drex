# PLAN.md — Drex Implementation Roadmap

*Created: 2026-03-11 | Reflects research state after Phase 10 (46 categories, 217+ experiments)*

---

## Current State

Ten phases of hypothesis-driven research are complete. The validated minimal architecture
stack is:

> **Delta-rule associative matrix + EMA smoothing (α=0.95) + episodic/semantic split
> (two H/2 matrices) + relative-vector-norm write gate (‖k − vp‖ ≥ thresh·‖k‖)**

Three of four components are **high-confidence and seed-stable (≥7/9 seeds)**.
The write gate has a known, unresolved calibration failure at short sequences.

---

## The Blocker

**The EMA bootstrap problem at L=32 is unresolved and is a deployment blocker.**

At short contexts (L=32), the relative-norm write gate fires with wr≈0.81–0.96,
regardless of threshold (confirmed across all Phase 10 experiments). This means:

- The gate provides zero selectivity at short sequences — it is functionally equivalent
  to "write everything"
- At longer contexts (L=96) the gate works correctly: wr≈0.20–0.31 at thresh=0.70
- For an LLM where context prefix length varies (short prompts are common), this is
  a correctness failure, not just a performance gap

**Root cause:** With α=0.95 and L=32, the EMA-smoothed memory M never converges within
the sequence. `‖k − vp‖ ≈ ‖k‖` throughout, so any fixed `thresh × ‖k‖` is
under the gate energy. A fixed scalar threshold cannot compensate for sequence-length-
dependent EMA convergence state.

**What Phase 10 ruled out:**
- Fixed threshold sweep (best: thresh=0.70, but INCONCLUSIVE on 1/3 seeds)
- Velocity gate (fatal zero-init deadlock: wr=0.000 by design)
- Position schedule (collapses accuracy to ~0.03 — blocks informative early writes)
- Higher interference density (gate advantage is non-monotone with ρ, not a scale fix)

---

## Phase 11 Plan — Fix the Blocker

Before starting full implementation, run **one targeted experiment** to resolve the
EMA bootstrap problem. If it succeeds, we address the blocker and move directly to
implementation.

### Phase 11, Experiment 1 (Priority 1): Length-Adaptive EMA Decay

**Hypothesis:** Replace fixed α=0.95 with α = f(L) — a function of sequence length —
so the EMA converges faster at short sequences. This directly targets the root cause:
at L=32, a lower α (e.g., 0.80–0.85) would allow `‖k − vp‖` to shrink during the
sequence, restoring gate selectivity.

**Candidate forms:**
- `α(L) = 1 − (1 − α_base) × (L_ref / L)` — inversely scales convergence rate
- `α(L) = α_min + (α_max − α_min) × min(L, L_max) / L_max` — linearly interpolates
- Per-position schedule: `α_t = α_max − (α_max − α_min) × (L − t) / L` (decay within seq)

**Success criterion:** wr_L32 ∈ [0.20, 0.70] AND acc_ratio_L32 ≥ 0.97 × EMA-alone,
both satisfied on ≥2/3 seeds at both L=32 and L=96 simultaneously.

**Estimated scope:** 3 seeds × 4–6 α schedule variants = 12–18 runs (single category).

### Phase 11, Experiment 2 (Contingency): Learned Gate Threshold MLP

If length-adaptive α fails (or partially solves only one length), run a lightweight
MLP conditioned on sequence position and/or EMA state magnitude to predict an adaptive
threshold offset. This is more expensive but handles the general case.

**Only run if Experiment 1 fails.**

---

## Implementation Plan (After Blocker Resolved)

Once Phase 11 Experiment 1 produces a seed-stable result, proceed to implementation
in this order:

### Step 1 — Core Memory Module (python/drex/models/memory.py)

Implement the validated architecture stack exactly as specified in ARCHITECTURE_FINDINGS.md §10:

- [ ] `MemoryModule` class: M_sem ∈ ℝ^{H/2 × H/2}, M_epi ∈ ℝ^{H/2 × H/2}
- [ ] Delta-rule write: `Δ = (k − vp) ⊗ k_n`, EMA update with (1−α)
- [ ] Episodic recency weight: `w_epi = (t+1) / L`
- [ ] Relative-norm write gate: `‖k − vp‖ ≥ thresh × ‖k‖`
- [ ] Length-adaptive α (result from Phase 11)
- [ ] `QueryFormer`: dedicated feed-forward query projection
- [ ] Soft retrieval: `r_sem = M_sem · q_n`, `r_epi = M_epi · q_n`
- [ ] Null retrieval gate (learned, no supervision needed)
- [ ] Output: `concat(r_sem, r_epi)` (default; no learned read gate per exp_38_3)
- [ ] Write gate threshold init at 0.40 (hard requirement from exp_43_1)
- [ ] Validation assertion: write rate must be in [0.10, 0.85] during training

### Step 2 — Integration into DrexTransformer (python/drex/models/transformer.py)

- [ ] Wire MemoryModule into existing transformer layer stack
- [ ] Confirm Adam optimizer (exp_34_6); AdamW acceptable
- [ ] Pass sequence length L into MemoryModule for α scheduling

### Step 3 — Test Suite (tests/python/)

- [ ] Unit tests: write gate criterion (correct dimension-invariance)
- [ ] Unit tests: delta-rule update math
- [ ] Unit tests: EMA coefficient behavior at L=32 vs L=96
- [ ] Unit tests: write rate assertion in [0.10, 0.85]
- [ ] Integration test: associative recall (passkey-style), verify acc > random
- [ ] Integration test: both L=32 and L=96 length generalization
- [ ] Regression test: write gate does not fire at wr=0.000 or wr=1.000

### Step 4 — Evaluation Script

- [ ] Extend `scripts/eval_passkey.py` to report write rate alongside accuracy
- [ ] Add multi-density sweep (ρ ∈ {0.08, 0.30}) to confirm gate value at higher density

### Step 5 — Documentation

- [ ] Update `README.md` with architecture description and installation instructions
- [ ] Update `ARCHITECTURE_FINDINGS.md` with Phase 11 result
- [ ] Close out research log entry for Phase 11

---

## What Can Start Now (Without Blocker Resolution)

The three fully validated, gate-independent components can be implemented immediately
while Phase 11 experiments run:

| Component | Status | Can implement now? |
|---|---|---|
| Delta-rule update rule | High confidence, 9-seed stable | Yes |
| EMA smoothing α=0.95 | High confidence, 9-seed stable | Yes |
| Episodic/semantic split (50/50, fixed) | High confidence, 9-seed stable | Yes |
| Dedicated QueryFormer | Medium confidence | Yes |
| Null retrieval gate | Medium confidence | Yes |
| Soft retrieval (concat output) | Medium confidence | Yes |
| Write gate (relative-norm criterion) | Gate criterion: high confidence | Yes, with L caveat |
| α = f(L) calibration | **UNRESOLVED — BLOCKER** | **No — pending Phase 11** |

Recommended: implement Steps 1–3 above using fixed α=0.95 and thresh=0.70, with the
length-adaptive α as a stub/TODO. This lets us build and test the full module skeleton
while Phase 11 resolves the calibration.

---

## Decision Gate

```
Phase 11 Exp 1 result:
  SUPPORTED (wr_L32 in target, acc_ratio ≥ 0.97, ≥2/3 seeds)
    → Plug α(L) formula into Step 1 → proceed to full implementation
  REFUTED / INCONCLUSIVE
    → Run Phase 11 Exp 2 (learned MLP threshold)
    → If that also fails, escalate: the write gate may need to be removed from the
      short-context path entirely and only activated at L > 64
```

---

## Hard Constraints (from Research)

These are non-negotiable architectural constraints — all have ≥7/9 seed evidence:

1. **Use relative-norm gate, not matrix-mean energy.** Matrix-mean produces O(1/H) values
   that are always below any reasonable threshold (exp_45_1).
2. **Initialize thresh at 0.40.** Random init risks the low-accuracy equilibrium (exp_43_1).
3. **Use fixed 50/50 episodic/semantic split, not a learned router.** Learned router is
   10–24% worse (exp_38_1).
4. **Do not use REINFORCE for gate training.** Encoder gradient norm = 0 (exp_7_1).
5. **Validate write rate ∈ [0.10, 0.85] after any change to the write mechanism.**
6. **Use Adam. Not SGD.** >10% accuracy spread across optimizers (exp_34_6).

---

## Dead Ends to Avoid

Do not re-investigate: tiered memory, hierarchical write decisions, momentum delta rule,
bidirectional delta rule, velocity gate, matrix-mean energy gate, position-schedule gate,
offline consolidation, hindsight oracle distillation, three-gate auxiliary loss combos,
write rate regularization, two-phase gate training. All were tested to refutation.
Full list in ARCHITECTURE_FINDINGS.md §9.
