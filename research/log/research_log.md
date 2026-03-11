# Research Log

This is a living document. Every experiment run gets an entry. Failures get
as much detail as successes — more, if the failure was interesting.

Format per entry:
- Date
- Experiment ID
- Outcome (SUPPORTED / REFUTED / INCONCLUSIVE / ERROR)
- What actually happened (not what was expected)
- The most surprising thing
- What it changes about the next question to ask

---

## Log

---

### 2026-03-08 | exp_7_9 | SUPPORTED (marginally)
**Interpretability Baseline**

The gate is non-random before training (std=0.146). A single linear projection
establishes token-type preferences from embedding geometry alone. Numeric tokens
are the most written (+0.076 trained correlation). Punctuation is avoided (−0.047).
Position has near-zero signal.

**Surprising:** Training barely changed the correlations (0.055 → 0.059 interpretability
score). The preferences are established in the embedding geometry, not learned from
the task. This means a single linear gate's "intelligence" lives in the embedding
space, not the gate weights.

**Changes:** As controllers get more complex, check whether the added layers are
learning task-specific signals or just amplifying embedding biases. The baseline
is now established.

---

### 2026-03-08 | exp_7_1 | SUPPORTED (strongly)
**Differentiability: STE vs Gumbel vs REINFORCE**

Ranking: Gumbel > STE > REINFORCE on both accuracy and stability.
REINFORCE loss variance: 2.696 (380× worse than Gumbel 0.007).
REINFORCE encoder gradient norm: 0.000 — the encoder does not learn.

**Surprising:** REINFORCE's gradient norm is literally zero. The policy gradient
signal does not flow back to the encoder at all in this setup. REINFORCE trains the
policy over a static, randomly-initialized feature space. This isn't a subtle
degradation — it's a complete training failure for the encoder component.

**Changes:** REINFORCE is ruled out for encoder-memory joint training. Gumbel-softmax
with temperature annealing is the default training mechanism going forward. This
constrains the design space substantially.

*Note: original implementation had a shape bug in the REINFORCE log_prob computation.
Fixed and re-run. See commit history.*

---

### 2026-03-08 | exp_6_6 | INCONCLUSIVE
**Controller Catastrophic Forgetting**

Domain A accuracy fell from 22.7% to 1.9% after domain B training (−0.208).
EWC reduced this by only 1.8% — negligible protection.

**Surprising:** The EWC failure isn't surprising given the low baseline (22.7%).
EWC protects weights with high Fisher information on domain A, but if those weights
were themselves weakly trained, there is little to protect. The forgetting may be
real but is hard to interpret at this performance level.

**Changes:** Re-run with domain A baseline above 70% before introducing domain B.
The forgetting phenomenon needs a stronger signal to work from. Also: investigate
whether the controller's policy (not just its output layer) is what's being forgotten.
The policy could be forgotten even when the task performance metric stays constant.

---

### 2026-03-08 | exp_5_1 | REFUTED
**Read Gate Collapse Detection**

Regime A (task only) stabilized at 15.6% read rate — no collapse.
Regime B (sparsity regularizer) collapsed to 9.4% (never).
Regime D (confidence) stable at 65.6%.

**Surprising:** Same pattern as exp_3_2. The regime designed to prevent collapse
(sparsity) caused the opposite collapse. The unconstrained gate found a stable
~16% read rate naturally. Regime A and regime C (coverage penalty) settled at
identical rates (15.6%), suggesting the task loss already determines the equilibrium
and additional regularization provides no marginal benefit.

**Changes:** The "collapse prevention" problem is not the main challenge. The
challenge is designing tasks that require selective memory use so the equilibrium
rate reflects genuine capability, not just distributional convenience.

---

### 2026-03-08 | exp_4_9 | SUPPORTED (strongly)
**Compositional Retrieval**

Single-hop: 0.968. Two-hop: 1.000. Two-hop *outperforms* single-hop.
The gap is −0.032 in favor of two-hop — the opposite of the expected direction.

**Surprising:** The two-hop architecture is more accurate than single-hop on this
task. The intermediate aggregation step may be acting as a regularizer that prevents
overfitting to spurious correlations in the direct pattern-match. The model is
forced to form a meaningful intermediate representation.

**Changes:** Follow up on why two-hop generalizes better. Is it the architecture or
the task structure? Run this on a larger KB with interference to see if the advantage
holds.

---

### 2026-03-08 | exp_4_7 | SUPPORTED* (degenerate)
**Null Retrieval Learning**

Null precision: 1.000. Null recall: 0.800. Retrieval on matches: 0.000.
The gate learned to never fire. "Always null" achieves 80% accuracy when 80% of
queries are null.

**Surprising:** The shortcut is perfectly rational. The model correctly learned the
task distribution and found the highest-reward policy — which happens to be useless.
The hypothesis was technically supported but the experiment was gamed.

**Changes:** Redesign with 50/50 null/retrieval split, and separate loss terms for
missed retrievals vs false retrievals. The degenerate solution reveals a task design
flaw, not a controller capability.

---

### 2026-03-08 | exp_3_2 | REFUTED
**Write Gate Collapse Detection**

Regime A (no signal): stable at 19.3% — no collapse.
Regime D (anti-collapse penalty): collapsed to 95.1% ALWAYS-write.

**Surprising:** The anti-collapse penalty caused the opposite collapse. The
unconstrained gate found a stable natural equilibrium. The assumption that gates
collapse without explicit anti-collapse signal was wrong, at this scale.

**Changes:** Don't add gate regularizers before understanding the natural equilibrium.
Investigate why the unconstrained gate settles at ~20% — is this a property of the
task or the model size?

---

### 2026-03-08 | exp_2_9 | REFUTED
**Retrieval vs Storage Compression Objectives**

Both compressors: acc@1 = 1.000. Reconstruction slightly better for A (0.414 vs 0.397).

**Surprising:** The task didn't discriminate between objectives. 5% query noise was
too easy for 8x compression to fail on. Both objectives trained perfectly adequate
retrievers.

**Changes:** Re-run at 64x compression where the ratio curve showed real degradation.
The objective discrimination likely only manifests when compression is severe enough
to require genuine tradeoffs.

---

### 2026-03-08 | exp_2_1 | INCONCLUSIVE
**Compression Ratio Curve**

The curve is non-monotonic. 2x–8x: cosine sim ~0.052 (near zero).
16x: cosine sim 0.219 (peak — more compression is better).
Gradual decline from 16x to 100x. No catastrophic cliff.

**Surprising:** Low compression ratios train poorly. Large bottlenecks give the
autoencoder enough room to be lazy — high-capacity models with insufficient training
budget fail to learn. The 16x sweet spot shows that moderate bottleneck constraints
act as useful regularization.

**Changes:** The question is not "where does quality cliff?" but "why does
underconstrained compression fail to learn?" Investigate the training dynamics at
2x–8x: are the gradients too sparse or too small? Does more training or a lower LR
fix the low-compression regime?

---

### 2026-03-08 | exp_1_5 | SUPPORTED (weakly)
**Write Signal Ablation**

Ranking: learned (0.124) > attention (0.121) > random (0.120) > surprise (0.118).
All policies clustered between 11.8–12.4%. Delta of +0.003 for learned.

**Surprising:** Surprise-weighted writing is the worst policy. High perplexity
tokens — the ones with the most "information" by many definitions — are not the
ones most worth remembering for downstream retrieval. The surprise signal is
anticorrelated with retrieval value.

**Changes:** Investigate the surprise anti-correlation further. Does this hold across
task types? If surprise is reliably anti-predictive of retrieval value, it could be
used as a *negative* signal — write the low-surprise, predictable tokens that form
the stable context. This inverts a common assumption.

---

## Meta-Observations (across all phase 1 experiments)

**The natural equilibrium:** Both write and read gates settle at ~15–20% activity
rates when trained on task loss alone. This wasn't predicted and deserves focused
investigation. Is ~16% the right rate for these tasks, or is it a model-size artifact?

**Regularizers are counterproductive:** Anti-collapse penalties, sparsity regularizers,
and coverage bonuses all produced worse behavior than no regularization in these
experiments. The correct approach may be task design rather than loss engineering.

**REINFORCE is ruled out:** Zero gradient norm through the encoder. This narrows
the differentiability design space to Gumbel-softmax and STE. Gumbel is preferred.

**Degenerate solutions dominate:** When a shortcut exists, the model finds it.
Task design is the primary tool for obtaining meaningful results, not loss design.

---

## Meta-Observations (Phase 8)

**Date:** 2026-03-10

Phase 8 completed 29 experiments across categories 41-44 and produced a clear
integration failure signal.

**What held up:**
- EMA smoothing reliably reduces gradient variance (about 75% reduction).
- Global alpha is sufficient; per-position alpha did not add measurable value.
- The episodic/semantic split advantage persists at SEQ_LEN=96.
- Write-gate multi-stability is real, with multiple threshold equilibria.

**What failed critically:**
- The full combination (EMA + split + write gate) collapsed accuracy from
  approximately 0.27 to approximately 0.03.
- Three stabilization attempts all failed: lower gate LR, write-rate regularization,
  and two-phase training.

**Interpretation:**
The write gate appears to interact destructively with EMA+split, likely by pushing
write behavior into pathological regimes that starve useful memory updates.

**Next direction (Phase 9):**
Treat gate interaction failure as the primary blocker. Instrument write-rate
trajectories and gate logits under composition first, then redesign gating under
explicit anti-starvation constraints before further mechanism integration.

---

## Phase 9 — Gate-Writing Interaction Repair (Category 45)
*2026-03-10*

### Root Cause Confirmed

The Phase 8 interpretation was incorrect. The gate did not "interact destructively with
EMA+split" — the gate was broken in complete isolation. The exp_44_1 energy formula
`Delta.pow(2).mean([1,2])` produces values ~0.007-0.016 (O(1/H)) due to averaging over
H² = 4096 matrix elements. With threshold=0.4, the gate fires 0% of the time, write_rate
→ 0, and performance degrades to near-random (~0.03). This was confirmed by exp_45_1
across all 3 seeds: matrix_max_ever < 0.01, vecnorm_mean ≈ 5–8 (O(‖k‖)).

### Phase 9 Results Summary (18 runs, 6 experiments × 3 seeds)

| Exp    | Outcome     | Key finding |
|--------|-------------|-------------|
| 45_1   | SUPPORTED   | Matrix-mean energy maxes at 0.007–0.009; vector-norm at 5–8. Scale bug confirmed. |
| 45_2   | SUPPORTED   | Relative-norm gate restores acc_gate 0.03→0.22–0.23. acc_full/acc_ema_split = 0.99–1.01. |
| 45_3   | INCONCLUSIVE| rel_norm stays in [0.39–0.40] across dims; matrix_mean dead. abs_norm didn't spread (0.001). |
| 45_4   | REFUTED     | EMA+gate configs have wr=0.96–0.97 (above 0.80 threshold). Gate, split_gate: wr≈0.40. |
| 45_5   | SUPPORTED   | Corrected full system: acc_full/acc_ema_split = 0.99–1.07 across all seeds. |
| 45_6   | REFUTED     | wr at L=32 is 0.95 (above 0.85); wr at L=96 is 0.31. Accuracy comparable, not strictly ≥. |

### Key Learnings

**The fix works**: The relative vector-norm gate (‖k−vp‖ ≥ 0.4×‖k‖) fully restores
accuracy. acc_gate jumps from 0.03 → 0.22–0.23; full system matches EMA+split within 1%.

**Unexpected finding — EMA increases write rate**: EMA+gate configs have wr≈0.96, while
gate-only has wr≈0.40. The EMA slows memory convergence, keeping ‖k−vp‖ large, which
means the gate rarely blocks. This is an emergent interaction between EMA persistence and
the gate threshold.

**Seq-len dependence**: Short sequences (L=32) have high gate fire rates (~0.96) because
the memory hasn't been populated yet (vp≈0, so ‖k−vp‖≈‖k‖ > 0.4×‖k‖ always). Longer
sequences (L=96) have moderate fire rates (~0.31) because the memory builds up. The gate
is sequence-length-sensitive in a predictable way.

**abs_norm spread didn't materialize**: The absolute-norm criterion (fixed threshold=0.4)
produces nearly identical write rates across H∈{32,64,128} because ‖k−vp‖ grows
approximately as √H while the threshold stays fixed — but in practice these nearly cancel.
The relative-norm criterion is stable across dims at wr≈0.40 (gate, split_gate).

### Phase 10 Direction

The gate repair is validated. The key open question is whether the EMA + gate interaction
(wr=0.96 for EMA+gate configs) is benign or whether it effectively neutralizes the gate.
Next priorities:
1. Investigate whether wr=0.96 for EMA+gate is functionally different from no gate at all
2. Explore adaptive/learned gate thresholds that account for EMA persistence
3. Consider sequence-position-dependent thresholds for short vs. long sequence regimes

---

## Phase 10 — EMA-Gate Threshold Calibration (Category 46)
*2026-03-11*

### Phase 10 Results Summary (18 runs, 6 experiments × 3 seeds)

| Exp    | Outcome      | Key finding |
|--------|--------------|-------------|
| 46_1   | SUPPORTED    | thresh*=0.70 is the best universal candidate; wr_L32≈0.61, wr_L96≈0.20 (2/3 seeds). |
| 46_2   | REFUTED      | Velocity gate deadlock: M=0 init → vp=0 → velocity=0 → wr=0.000, 3/3 seeds. |
| 46_3   | REFUTED      | Position schedule achieves wr targets at L=32 (thresh_max=1.5/2.0) but accuracy collapses to ~0.03. |
| 46_4   | INCONCLUSIVE | wr_L96 healthy (≈0.198–0.200) but gate gain at L=32 ≤ 0.005 (thresh* insufficient at standard density). |
| 46_5   | REFUTED      | Gate advantage non-monotone with ρ; negative at ρ=0.12 in 2/3 seeds. No capacity-pressure scaling. |
| 46_6   | REFUTED      | wr_L32≈0.81 (bootstrap problem persists); gain_L96 ratio=0.875–0.983 (< 1.02 target). |

### Key Learnings

**thresh*=0.70 is the best candidate, but only marginally**: exp_46_1 found thresh*=0.70
achieves write-rate targets at both lengths (wr_L32≈0.61, wr_L96≈0.20) in 2/3 seeds.
Seed 42 remained INCONCLUSIVE — no single threshold simultaneously satisfied both length
targets with accuracy ≥0.97×EMA. The threshold calibration problem is real but barely
solvable at the standard task density.

**Velocity gate is broken by design**: exp_46_2 revealed a fatal structural flaw:
the memory matrix M initialises to zeros, so vp = M @ k_norm = 0, making velocity
‖vp_t − vp_{t-1}‖/‖k‖ = 0 for the first step and essentially throughout a short sequence.
The gate never fires (wr=0.000) across all 8 configs and all 3 seeds. This is not a
hyperparameter issue — the zero-init deadlock is intrinsic to the design.

**Position schedule trades accuracy for write-rate control**: exp_46_3 revealed a
fundamental tradeoff. thresh_max=1.5 and 2.0 successfully push wr_L32 into [0.20, 0.60],
but at the cost of accuracy collapsing from ~0.26 to ~0.03 (near random). The high early
threshold blocks too many informative writes during the critical bootstrap phase, causing
the memory to remain empty and the retrieval to fail.

**The gate adds no measurable value at standard task density**: exp_46_4 confirmed that
thresh*=0.70 produces healthy wr at L=96 (≈0.20) but the accuracy gain of the gate over
EMA-alone is consistently ≤0.005 (range: −0.027 to +0.005 across seeds). The task density
(ρ=0.078, N=5 pairs, H=64) is too easy for the gate's write selectivity to confer genuine
benefits. This suggests the gate's value is density-conditional.

**Gate advantage does not scale monotonically with interference density**: exp_46_5
tested ρ ∈ {0.08, 0.12, 0.19, 0.31, 0.50, 0.75}. In 2/3 seeds, the advantage went
negative at ρ=0.12 or ρ=0.19, violating the monotone hypothesis. This is unexpected:
the gate should be most useful under the highest memory pressure, but the relationship
appears non-monotone in practice. The gate may be disrupting useful early writes at
intermediate densities before the memory pressure is high enough to require selectivity.

**Root problem: the EMA bootstrap at L=32 is still unsolved**: exp_46_6 confirmed the
persistent failure mode. The calibrated full system (EMA + split + gate, thresh*=0.70)
achieves wr_L32≈0.81 across all seeds — well above the [0.20, 0.70] target. This means
the EMA keeps ‖k−vp‖ large throughout the short sequence (since vp hasn't convergied),
so the gate fires almost universally regardless of thresh*. The Phase 10 calibration
effort did not solve the sequence-length-dependent write-rate problem.

### Phase 11 Direction

The threshold calibration approach is fundamentally limited. The core problem is that
EMA persistence (α=0.95) at short sequences (L=32) keeps ‖k−vp‖ ≈ ‖k‖, making the
relative-norm gate fire near-universally. A fixed scalar threshold cannot solve this.

Next investigation directions:
1. **Length-adaptive decay**: Replace fixed α=0.95 with α = f(L) or a per-position
   schedule, so the EMA converges faster at shorter sequences and reduces the bootstrap
   problem. The EMA itself (not the gate) may need to be the target of calibration.
2. **Learned gate threshold**: Train a lightweight MLP to predict the gate threshold
   conditioned on EMA state or sequence position, rather than using a fixed scalar.
3. **Hard capacity constraints**: Instead of a gate, try a "memory budget" mechanism
   that enforces a fixed number of writes per sequence (e.g., top-k by gate score),
   making the write-rate constraint hard rather than soft.
4. **Higher-density benchmarks**: The current task (ρ=0.078) may be too easy. Move to
   ρ≥0.30 as the primary test case where gate selectivity has non-trivial consequences
   for accuracy.
