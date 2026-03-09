# Hypotheses

Each experiment has a single falsifiable hypothesis. A result is only useful if
it can prove the hypothesis false. Negative results are recorded with the same
rigor as positive ones.

---

## Category 1: What To Write

**H-1.1 — Relevance Signal Baseline**
Attention weight is a valid proxy for memory importance and correlates with what
a human would consider "worth remembering."
*Status: UNTESTED*

**H-1.2 — Surprise as a Write Signal**
A memory built from high-perplexity tokens supports better downstream retrieval
than one built from attention weights.
*Status: UNTESTED*

**H-1.3 — Gradient Magnitude as Write Signal**
Storing representations where gradient magnitude is highest produces a memory
store that generalizes better than attention-based selection.
*Status: UNTESTED*

**H-1.4 — Contrastive Write Selection**
Diversity-driven storage (maximally different entries) outperforms importance-driven
storage on recall tasks.
*Status: UNTESTED*

**H-1.5 — Write Signal Ablation**
A learned write gate outperforms random write, attention-weighted write, and
surprise-driven write on associative recall tasks. (PRIORITY)
*Status: UNTESTED*

**H-1.6 — Semantic Deduplication at Write Time**
Deduplication via cosine similarity improves retrieval precision without causing
information loss significant enough to hurt downstream task performance.
*Status: UNTESTED*

**H-1.7 — Write Frequency vs Write Quality**
For a fixed storage budget, infrequent writes with high compression outperform
frequent writes with low compression on long-document QA tasks.
*Status: UNTESTED*

**H-1.8 — Hierarchical Write Decisions**
A two-stage write decision (coarse gate then fine-grained selection) outperforms
a single write gate.
*Status: UNTESTED*

---

## Category 2: How To Write (Compression)

**H-2.1 — Compression Ratio vs Recall Fidelity Curve**
There exists a compression ratio threshold beyond which recall fidelity degrades
catastrophically rather than gracefully. (PRIORITY)
*Status: UNTESTED*

**H-2.2 — Autoencoder vs Attention-based Compression**
Attention-based compression produces more retrievable representations than
autoencoder compression on inferential recall tasks (though results may diverge
by task type).
*Status: UNTESTED*

**H-2.3 — Lossy vs Lossless Memory Representations**
A controller can learn without supervision which information should be stored
exactly (numbers, names) vs. approximately (themes, sentiment).
*Status: UNTESTED*

**H-2.4 — Chunk Size Sensitivity**
There exists an optimal chunk size for compression beyond which quality degrades
independent of compression ratio.
*Status: UNTESTED*

**H-2.5 — Structured vs Unstructured Compression**
Compressing into a structured representation (key-value, slot-based) improves
retrieval over compressing into a flat dense vector.
*Status: UNTESTED*

**H-2.6 — Compression Generalization**
A compressor trained on domain A produces meaningfully worse retrieval on domain B
than a domain-B-trained compressor, indicating compression overfits to domain.
*Status: UNTESTED*

**H-2.7 — Iterative Compression**
A hierarchy of increasingly abstract memory levels can be built by iterative
compression without catastrophic information loss at each stage.
*Status: UNTESTED*

**H-2.8 — Compression Under Distribution Shift**
A compressor degrades gracefully (not catastrophically) when input distribution
shifts significantly mid-context.
*Status: UNTESTED*

**H-2.9 — Retrieval-Oriented vs Storage-Oriented Compression**
Minimizing reconstruction loss and maximizing downstream retrieval accuracy are
fundamentally different objectives and produce measurably different representations.
(PRIORITY)
*Status: UNTESTED*

---

## Category 3: When To Write

**H-3.1 — Continuous vs Event-Driven Writing**
Event-driven writing (learned gate) produces better memory coverage than writing
every N tokens for a fixed storage budget.
*Status: UNTESTED*

**H-3.2 — Write Gate Collapse**
A learned write gate trained without explicit anti-collapse objectives will learn
to never write within N training steps on standard tasks. (PRIORITY)
*Status: UNTESTED*

**H-3.3 — Write Timing vs Content Quality**
Writing later in a context (more processed representations) outperforms writing
early (raw representations) for inferential downstream tasks.
*Status: UNTESTED*

**H-3.4 — Boundary Detection as Write Trigger**
Semantic-boundary-triggered writing outperforms fixed-interval writing on
long-document tasks with clear topical structure.
*Status: UNTESTED*

**H-3.5 — Write Latency Sensitivity**
Downstream retrieval quality degrades measurably when write latency exceeds a
specific token distance threshold.
*Status: UNTESTED*

**H-3.6 — Retroactive Writing**
A controller can learn to retroactively write tokens it initially skipped once
later context reveals their importance.
*Status: UNTESTED*

**H-3.7 — Write Budget Allocation**
A controller given a fixed write budget per context learns to allocate that budget
non-uniformly in a way that improves performance over uniform allocation.
*Status: UNTESTED*

---

## Category 4: What To Read

**H-4.1 — Query Formulation Quality**
A dedicated query formulation module outperforms direct use of the current hidden
state as a retrieval query.
*Status: UNTESTED*

**H-4.2 — Single vs Multi-Vector Retrieval**
Multi-vector retrieval captures more relevant memory content than single-vector
retrieval on tasks with multi-faceted information needs.
*Status: UNTESTED*

**H-4.3 — Retrieval Depth Sensitivity**
There exists an optimal retrieval depth (top-k) beyond which additional retrieved
entries introduce more noise than signal.
*Status: UNTESTED*

**H-4.4 — Soft vs Hard Retrieval**
Soft retrieval (weighted average) produces more stable training than hard retrieval
(discrete selection), though hard retrieval may achieve higher peak task performance.
*Status: UNTESTED*

**H-4.5 — Cross-Level Retrieval**
Simultaneous cross-tier retrieval achieves better recall than sequential cascading
retrieval (working → episodic → semantic).
*Status: UNTESTED*

**H-4.6 — Retrieval by Reconstruction vs Similarity**
For tasks requiring exact recall, similarity-based retrieval outperforms
reconstruction-based retrieval. For inferential completion, the relationship inverts.
*Status: UNTESTED*

**H-4.7 — Null Retrieval Learning**
A learned read gate can be trained to return null (no retrieval) on tasks where
most queries have no relevant memory content, without explicit null supervision.
(PRIORITY)
*Status: UNTESTED*

**H-4.8 — Retrieval Interference**
Retrieval quality degrades non-linearly as the number of near-duplicate entries
in memory increases, with a specific saturation point.
*Status: UNTESTED*

**H-4.9 — Compositional Retrieval**
A learned retrieval mechanism can be trained to retrieve two separate memory entries
and compose them to answer questions neither entry answers alone. (PRIORITY)
*Status: UNTESTED*

---

## Category 5: When To Read

**H-5.1 — Read Gate Collapse**
A learned read gate trained without explicit anti-collapse objectives will learn
a degenerate policy (always read or never read) within N training steps. (PRIORITY)
*Status: UNTESTED*

**H-5.2 — Read Frequency vs Task Performance**
Optimal read frequency is task-dependent and cannot be determined by a single
fixed schedule across task types.
*Status: UNTESTED*

**H-5.3 — Predictive Read Triggering**
Anticipatory retrieval (predicting retrieval need before it arises) improves
end-to-end latency without measurably hurting task quality.
*Status: UNTESTED*

**H-5.4 — Read vs Recompute Decision**
A controller can learn to prefer recomputation over retrieval for information
that is cheap to recompute and prefer retrieval for information that is expensive.
*Status: UNTESTED*

**H-5.5 — Cascading Read Depth**
Confidence-gated cascading retrieval (shallow first, deeper only if low confidence)
matches full-depth retrieval quality at significantly lower average compute cost.
*Status: UNTESTED*

**H-5.6 — Read Suppression Under High Confidence**
Suppressing memory reads when next-token prediction confidence exceeds a threshold
costs less than 1% task quality on standard benchmarks.
*Status: UNTESTED*

**H-5.7 — Attention-Memory Arbitration**
When local attention and external memory produce conflicting predictions, a learned
arbitration policy outperforms both fixed-priority policies (always prefer attention,
always prefer memory).
*Status: UNTESTED*

---

## Category 6: How To Forget

**H-6.1 — Eviction Policy Comparison**
A learned importance-scored eviction policy significantly outperforms LRU on
tasks requiring retention of low-frequency but high-importance information.
*Status: UNTESTED*

**H-6.2 — Forgetting as Compression**
Graceful degradation via iterative compression outperforms hard eviction for
long-context tasks where storage budget is the binding constraint.
*Status: UNTESTED*

**H-6.3 — Selective Forgetting Under Distribution Shift**
A controller can learn to evict domain-mismatched memories when input distribution
shifts, without explicit domain labels.
*Status: UNTESTED*

**H-6.4 — Protected Memory Slots**
A controller can learn which memories deserve protection (never evict) without
explicit supervision, and performance degrades predictably as protected set size
grows beyond an optimal threshold.
*Status: UNTESTED*

**H-6.5 — Forgetting Curve Mimicry**
A biologically-inspired memory decay function (Ebbinghaus curve) improves
long-horizon task performance compared to instant eviction.
*Status: UNTESTED*

**H-6.6 — Catastrophic Forgetting of the Controller Itself**
The memory controller itself (as a neural network) suffers measurable catastrophic
forgetting of its learned policies when exposed to a new domain. (PRIORITY)
*Status: UNTESTED*

**H-6.7 — Write-Evict Coupling**
Joint optimization of write and evict decisions outperforms treating them as
independent operations on tasks where storage pressure is constant.
*Status: UNTESTED*

**H-6.8 — Memory Consolidation**
Periodic offline consolidation (merging multiple entries into one higher-level
representation) improves long-horizon performance without active context.
*Status: UNTESTED*

---

## Category 7: Cross-Cutting

**H-7.1 — End-to-End Controller Differentiability**
Gumbel-softmax relaxation produces more stable training than straight-through
estimators, and both outperform RL-based approaches for discrete memory selection.
(PRIORITY)
*Status: UNTESTED*

**H-7.2 — Controller Overhead Budget**
There exists a maximum controller complexity (measured in FLOPs) beyond which
the controller's overhead exceeds its efficiency contribution.
*Status: UNTESTED*

**H-7.3 — Controller Generalization Across Task Types**
A controller trained on factual QA learns memory management policies that
generalize to reasoning tasks but not to generation tasks.
*Status: UNTESTED*

**H-7.4 — Minimal Controller Architecture Search**
Meaningful memory management behavior requires at minimum two layers of
non-linearity in the controller network.
*Status: UNTESTED*

**H-7.5 — Controller Stability Under Scale**
A controller's learned policy trained at 100M parameters does not transfer
directly to a 1B parameter model without additional fine-tuning.
*Status: UNTESTED*

**H-7.6 — Adversarial Memory Probing**
The memory controller is measurably vulnerable to inputs designed to maximize
write activity, and this vulnerability does not self-correct during training.
*Status: UNTESTED*

**H-7.7 — Memory Controller as Bottleneck Identification**
Write quality (not read quality, compression ratio, or eviction policy) is the
first performance bottleneck encountered during controller training.
*Status: UNTESTED*

**H-7.8 — Joint vs Sequential Controller Training**
Curriculum training (one component at a time) produces more stable controller
behavior than joint training from the start.
*Status: UNTESTED*

**H-7.9 — Controller Interpretability Baseline**
The controller's write and read decisions are interpretable (non-random, correlating
with human-meaningful features) in their simplest form before any task-specific
training. (PRIORITY)
*Status: UNTESTED*

---

## Category 8: Mechanistic Investigations (Phase 2)

**H-8.1 — Attention Anti-Correlation is Normalization Artifact**
The Pearson r = −0.503 correlation between attention weight and write-gate importance
is an artifact of softmax normalization; raw pre-softmax dot products will show
positive or near-zero correlation with the same importance signal.
*Status: UNTESTED*

**H-8.2 — Gate Equilibrium Scales With Task Demand**
The natural ~16–20% write-gate equilibrium observed across 3 independent experiments
scales with task memory demand: higher KV-pair density drives higher equilibrium
write rate (Pearson r > 0.8 across difficulty levels).
*Status: UNTESTED*

**H-8.3 — Write-Evict Collapse is Gradient Aliasing**
The write-evict collapse (r = 0.990) observed in exp_6_7 is caused by gradient
aliasing from a shared task loss; oracle pre-training on independent labels before
joint fine-tuning breaks the collapse and produces r < 0.5.
*Status: UNTESTED*

**H-8.4 — Write Gate Exploits Position as Content Proxy**
The learned write gate relies on token position as a proxy for content importance;
removing positional embeddings degrades gate quality significantly (accuracy drop
> 0.5%) while position correlation drops below 0.05.
*Status: UNTESTED*

---

## Category 9: Inconclusive Redesigns (Phase 2)

**H-9.1 — Compression Objective Gap at 64× Compression**
At 64× compression with a 100-way gallery, an InfoNCE retrieval objective produces
>15% absolute Acc@1 gain over an MSE reconstruction objective, confirming that
the Phase 1 result was inconclusive only because 8× was too easy.
*Status: UNTESTED*

**H-9.2 — Null Retrieval With Balanced Distribution**
A memory controller trained on a 50/50 null/retrieval split achieves F1 > 0.60
on both null detection and genuine retrieval, demonstrating real selectivity rather
than distributional shortcut exploitation.
*Status: UNTESTED*

**H-9.3 — EWC Effectiveness With Strong Domain-A Baseline**
Elastic Weight Consolidation reduces forward transfer forgetting by > 50% relative
to unregularized sequential training when the domain-A baseline accuracy exceeds 70%.
*Status: UNTESTED*

**H-9.4 — Protected Slots Interior Optimum Above K=5**
The write-accuracy peak for protected memory slots lies at K > 5 when MEMORY_SLOTS = 12;
the Phase 1 result (K=5 best, at K_max) was truncated and does not represent a true
interior optimum.
*Status: UNTESTED*

**H-9.5 — Write Budget Non-Uniform Allocation on Biased Task**
A learned write budget produces measurably non-uniform slot allocation (Gini > 0.50)
and accuracy gain > 5% over uniform allocation when the task structure requires
concentrated writes in one segment of the sequence.
*Status: UNTESTED*

---

## Category 10: Retroactive Writing Mechanism (Phase 2)

**H-10.1 — Retroactive Benefit Decays With Lookahead Window**
The accuracy benefit of retroactive writing falls below 5% of its full-sequence
value when the lookahead window is restricted to fewer than 6 tokens of right context.
*Status: UNTESTED*

**H-10.2 — Retroactive Gain is Primarily New-Write, Not Re-Encoding**
More than 80% of the combined retroactive accuracy gain comes from writing previously
skipped tokens (new-write), not from re-encoding already-written tokens with updated
context (overwrite).
*Status: UNTESTED*

**H-10.3 — Retroactive Gain Scales With Sequence Length**
The absolute accuracy benefit of retroactive writing increases monotonically with
sequence length (Pearson r > 0.8) across seq_len ∈ {24, 32, 48, 64}.
*Status: UNTESTED*

---

## Category 11: Read Bottleneck Interventions (Phase 2)

**H-11.1 — Two-Step Query Former Reduces Read Bottleneck**
A two-step query former (linear projection followed by cross-attention over the
last 4 hidden states) improves oracle read accuracy by > 10% compared to a single
linear query projection, directly addressing the bottleneck identified in exp_7_7.
*Status: UNTESTED*

**H-11.2 — Read-Before-Write Deduplication Improves Retrieval**
Suppressing writes for tokens with cosine similarity > 0.8 to existing memory
improves retrieval F1 by > 3% without reducing recall by more than 5%.
*Status: UNTESTED*

**H-11.3 — Optimal Suppression Threshold Varies By Task Type**
The optimal read-suppression confidence threshold from exp_5_6 varies by more than
0.15 across factual, pattern, and completion retrieval tasks, indicating that a
single universal threshold cannot be optimal.
*Status: UNTESTED*

---

## Category 12: Compression Hard Regimes (Phase 2)

**H-12.1 — Retrieval Objective Dominates at 64× Compression**
At 64× token compression with a 100-way gallery, a retrieval (InfoNCE) training
objective outperforms a reconstruction (MSE) objective by > 15% absolute Acc@1.
*Status: UNTESTED*

**H-12.2 — Extended LR Warmup Rescues Gradient-Starved Compression**
The 2×–8× compression failure in exp_2_1 is gradient starvation from a too-wide
bottleneck; a cosine LR schedule with extended warmup recovers > 10% training
accuracy relative to constant LR at the same step budget.
*Status: UNTESTED*

---

## Category 13: Compositional Retrieval at Scale (Phase 2)

**H-13.1 — Two-Hop Regularization Persists With Interference**
The accuracy advantage of two-hop compositional retrieval (exp_4_9) persists
at a 64-entity knowledge base with 40% near-duplicate interference, with
the gap remaining ≥ the original 2-hop advantage on the interference subset.
*Status: UNTESTED*

**H-13.2 — Three-Hop Chain Is Feasible At Hidden Dim 64**
Three-hop chain retrieval is achievable with hidden_dim=64; accuracy degrades
less than 50% relative to two-hop accuracy on the same knowledge base.
*Status: UNTESTED*

---

## Category 14: System Integration (Phase 2)

**H-14.1 — Retroactive Write and Read Suppression Are Super-Additive**
Combining retroactive writing (cat10) and read confidence suppression (cat11)
produces accuracy gains that are super-additive (combined gain > sum of individual
gains + 1%), because each mechanism targets an independent bottleneck.
*Status: UNTESTED*

**H-14.2 — Write-First Curriculum Outperforms Joint Training**
Training the write gate for 1000 steps before adding read-head training produces
higher final accuracy than joint training from step 0, because the write gate first
learns a stable policy before the read head introduces competing gradients.
*Status: UNTESTED*

**H-14.3 — Cosine Gumbel Temperature Annealing Improves Accuracy**
Annealing Gumbel-softmax temperature from 1.0 to 0.1 via cosine schedule improves
final accuracy by > 2% compared to a constant temperature of 0.5, because gradual
discretization prevents early collapse to a near-deterministic policy.
*Status: UNTESTED*

---

## Category 15: Delta Rule / Associative Matrix Writes (Phase 3)

**H-15.1 — Delta Rule Outperforms Slot Write Via Interference Correction**
Delta rule matrix writes (M += (v − Mk^T/‖k‖²)k^T) outperform standard fixed-slot
writes by > 5% accuracy due to explicit interference correction, which prevents
over-writing when multiple keys share similarity.
*Status: UNTESTED*

**H-15.2 — Correction Term is Critical for Key-Interference Tasks**
Removing the correction term from the delta rule (degrading to Hebbian M += vk^T)
causes accuracy to drop > 10% on tasks where 50% of keys are deliberately set to
interfering (near-duplicate) values.
*Status: UNTESTED*

**H-15.3 — Energy-Gated Delta Rule Achieves Sparse Writes at Low Cost**
An energy-gated delta rule (write only when ΔE < 0) achieves > 90% of the accuracy
of continuous writes while reducing write rate to < 70%, because the energy
criterion naturally gates out redundant or interference-increasing writes.
*Status: UNTESTED*

**H-15.4 — Delta Rule Outperforms Outer-Product Write on Overwrite Tasks**
On tasks requiring key overwrite (same key appearing twice with different values),
delta rule writes outperform Larimar-style outer-product writes (M += vk^T) by
> 10% accuracy, because the correction term explicitly handles prior associations.
*Status: UNTESTED*

---

## Category 16: Online Gradient Descent Memory / Titans-Style (Phase 3)

**H-16.1 — Parametric MLP Memory Outperforms Slot Memory at Matched Budget**
A parametric MLP memory updated via 1 SGD step per token outperforms a fixed-slot
memory array at a matched parameter budget (≈256 params each) on associative recall,
because gradient steps enable finer-grained interference management than slot-level
writes.
*Status: UNTESTED*

**H-16.2 — Surprise-Gated Updates Recover Accuracy at Reduced Step Count**
A surprise-gated parametric memory (update skipped for low-surprise tokens, as
measured by L2 distance from running mean) achieves accuracy within 2% of full
updates while reducing update steps by > 40% at the optimal surprise threshold.
*Status: UNTESTED*

**H-16.3 — Parametric Memory Scales Better With Sequence Length Than Slot Memory**
Parametric MLP memory retains a higher fraction of peak accuracy as sequence length
scales from 24 to 96 tokens compared to slot-based memory (accuracy retention ratio
difference > 15%), because gradient steps adapt the representation to sequence length.
*Status: UNTESTED*

---

## Category 17: Prospective / Query-Conditioned Writing (Phase 3)

**H-17.1 — Query-Conditioned Write Gate Outperforms Context-Only Gate**
A write gate conditioned on a predicted future query type outperforms a context-only
gate by > 5% on multi-query-type tasks, because anticipating query structure enables
proactive selection of task-relevant tokens.
*Status: UNTESTED*

**H-17.2 — K-Token Lookahead Async Write Has an Optimal Lookahead Distance**
An asynchronous write gate that decides K steps before the write token uses lookahead
context to outperform a same-time gate by > 3% at some K ∈ {2, 4, 6}.
*Status: UNTESTED*

**H-17.3 — Prospective and Retroactive Writing are Redundant Mechanisms**
Combining prospective write (query-conditioned gate) and retroactive write (revision
gate) yields less than 1.5× the accuracy gain of either mechanism alone, because both
mechanisms compensate for the same forward-pass gate limitation.
*Status: UNTESTED*

**H-17.4 — Query-Conditioned Write Gain Scales With Query Predictability**
The accuracy advantage of query-conditioned writing correlates linearly (r > 0.85)
with the predictability of query type from context, because the write gate can only
exploit query information it can reliably predict.
*Status: UNTESTED*

---

## Category 18: Tiered Memory Architecture (Phase 3)

**H-18.1 — Two-Tier Memory Outperforms Flat Memory on Long-Context Tasks**
A two-tier memory (16-slot fast + 64-slot slow with learned demotion) outperforms
a flat 64-slot memory by > 5% on tasks where critical pairs are seeded early and
must survive 64-token contexts.
*Status: UNTESTED*

**H-18.2 — Learned Demotion Controller Discovers Frequency Over Recency**
A trained demotion controller learns a frequency-based policy rather than a
recency-based policy, producing positive correlation (> 0.15) between access
count and demotion probability and negative correlation (< −0.10) with recency.
*Status: UNTESTED*

**H-18.3 — Tiered Memory Has a Capacity Crossover Point**
Flat memory is more parameter-efficient below ~32 total slots; tiered architecture
outperforms flat above that threshold, with a crossover point in the range [16, 64].
*Status: UNTESTED*

**H-18.4 — Simultaneous Cross-Tier Retrieval Outperforms Sequential**
Parallel attention over both memory tiers simultaneously outperforms cascaded
sequential retrieval (fast first, then slow if confidence low) by > 3% accuracy,
because sequential retrieval introduces greedy commitment errors.
*Status: UNTESTED*

---

## Category 19: Sparse Hopfield Addressing (Phase 3)

**H-19.1 — Sparse Hopfield Retrieval Outperforms Soft Attention on Interference**
Sparse Hopfield retrieval (top-k=2 SparseMAP-style masking) outperforms standard
soft attention by > 5% precision@1 on tasks with 40% near-duplicate interference,
because hard zeros suppress spurious activation from interfering patterns.
*Status: UNTESTED*

**H-19.2 — Energy Write Criterion Produces Sparse Writes at Accuracy Gain**
Writing to Hopfield memory only when the energy change ΔE < 0 produces a write
rate < 35% while improving accuracy by > 3%, because the energy criterion
filters writes that would increase interference.
*Status: UNTESTED*

**H-19.3 — Sparse Hopfield Sustains Capacity Longer Before Cliff**
Sparse Hopfield addressing maintains retrieval accuracy for at least 2 more
patterns than dense soft attention before catastrophic capacity cliff, as measured
by the pattern count at which accuracy drops below 50%.
*Status: UNTESTED*

---

## Category 20: Three-Gate Coordinated Controller (Phase 3)

**H-20.1 — Write Sparsity Auxiliary Loss Improves Accuracy and Gate Health**
An L1 auxiliary loss targeting ~15% write rate equilibrium improves downstream
task accuracy while keeping write rate in the healthy range [5%, 35%], avoiding
the degenerate modes observed in exp_3_2.
*Status: UNTESTED*

**H-20.2 — Read Accuracy Auxiliary Loss Reduces Read Bottleneck**
An explicit cross-entropy auxiliary loss on oracle read accuracy (with detached
gradients to the write gate) reduces the read bottleneck identified in exp_7_7
more than implicit task-loss signal alone (oracle read accuracy gain > 5%).
*Status: UNTESTED*

**H-20.3 — Combined Auxiliary Losses Outperform Any Single Auxiliary**
A three-gate controller trained with all auxiliary losses (write sparsity + read
accuracy + forget usefulness) outperforms any single-auxiliary or no-auxiliary
baseline by > 2% accuracy, because each loss targets an independent failure mode.
*Status: UNTESTED*

**H-20.4 — Optimal Auxiliary Weight Range is [0.01, 0.1]**
Task accuracy is maximized at auxiliary loss weight λ ∈ [0.01, 0.1]; at λ < 0.01
the auxiliary has no effect, and at λ > 0.1 it dominates the task signal.
*Status: UNTESTED*

---

## Category 21: Feedforward Controller + Hindsight Distillation (Phase 3)

**H-21.1 — Feedforward Controller Achieves Higher Memory Utilization**
A feedforward-only controller (no recurrence) achieves higher external memory
utilization (measured by slot-access entropy) than an LSTM controller at equal
parameter count, because recurrent controllers can short-circuit memory access
via their hidden state.
*Status: UNTESTED*

**H-21.2 — Hindsight Oracle Labels Improve Write Gate Training Signal**
A write gate trained with hindsight oracle labels (retroactively marking which
writes were causally necessary for correct predictions) achieves higher task
accuracy than end-to-end gradient training on the same task and step budget.
*Status: UNTESTED*

**H-21.3 — Distilled Gate Outperforms End-to-End Learned Gate**
A write gate distilled from oracle hindsight labels via alternating BCE + task
training achieves higher accuracy than an end-to-end trained gate, because the
oracle supervision provides a cleaner gradient signal for sparse discrete decisions.
*Status: UNTESTED*

**H-21.4 — Feedforward Controller + Hindsight Distillation is Strongest Write Policy**
Combining a feedforward controller with hindsight oracle distillation produces
higher accuracy than any single-mechanism baseline (LSTM+task, FF+task, LSTM+oracle),
representing the strongest overall write policy tested in Phase 3.
*Status: UNTESTED*

---

## Category 22: Read Architecture Redesigns (Phase 4)

**H-22.1 — Slot-Conditioned Read Reduces Read Error**
Computing the read query as a linear combination of slot embeddings (soft attention
over slots → query) reduces read error by > 5% compared to a fixed linear projection
from the final hidden state, because the query adapts to what is already stored.
*Status: UNTESTED*

**H-22.2 — Iterative Message-Passing Read Outperforms Single-Pass**
Two rounds of slot→query→slot attention refinement outperform single-pass dot-product
read by > 3% on multi-fact retrieval tasks, because iterative refinement narrows
ambiguity in which slot to attend to.
*Status: UNTESTED*

**H-22.3 — Orthogonal Slot Initialization Prevents Collapse and Improves Recall**
Orthogonal initialization of slot weight matrices via Gram-Schmidt prevents slot
collapse (measurably lower mean pairwise cosine similarity) and independently
improves read accuracy by > 5% without changing the read mechanism.
*Status: UNTESTED*

**H-22.4 — Read Gate Transfers From Single-Hop to Multi-Hop Without Retuning**
An entropy-threshold read gate trained on single-hop associative recall transfers
to multi-hop tasks without retuning, achieving accuracy within 5% of a gate retrained
on the multi-hop distribution.
*Status: UNTESTED*

**H-22.5 — Contrastive Slot Training Improves Precision Under Interference**
An InfoNCE auxiliary loss that pushes apart slot embeddings improves retrieval
precision@1 by > 5% on high-interference tasks (many similar keys), because
contrastive training increases the angular separation between stored representations.
*Status: UNTESTED*

---

## Category 23: Retroactive Re-Encoding Variants (Phase 4)

**H-23.1 — Multi-Head Re-Encoding Outperforms Single-Head**
Multi-head cross-attention (num_heads=4) for re-encoding existing memory slots
outperforms single-head cross-attention re-encoding by > 3%, because multiple
attention heads capture richer slot-context interactions.
*Status: UNTESTED*

**H-23.2 — Second Re-Encoding Pass Yields Diminishing Returns**
A second re-encoding pass (applying cross-attention re-encoding to already
re-encoded slots) contributes less than 20% of the gain from the first pass,
confirming that re-encoding has rapidly diminishing returns.
*Status: UNTESTED*

**H-23.3 — Selective Re-Encoding Achieves High Accuracy at Low Rate**
Re-encoding only slots with cosine distance > T from the context mean achieves
> 90% of full re-encoding accuracy while processing < 60% of slots, because
slots already similar to context do not benefit from re-encoding.
*Status: UNTESTED*

**H-23.4 — Re-Encoding Gain is Task-Type Specific**
Factual recall tasks benefit > 2× more from re-encoding than pattern completion
tasks (measured by accuracy gain), because factual associations require binding
arbitrary tokens whereas patterns can be inferred from local context.
*Status: UNTESTED*

---

## Category 24: Scale and Length Generalization (Phase 4)

**H-24.1 — Parametric Memory Retains > 80% Accuracy at 4× Length**
Parametric MLP memory trained at seq_len=24 retains > 80% of its peak accuracy
at seq_len=96 (4×), while slot memory drops below 40%, confirming the length-scaling
advantage observed in exp_16_3.
*Status: UNTESTED*

**H-24.2 — Two-Hop Retrieval Sustains > 70% Accuracy Under 60% Interference**
Compositional two-hop retrieval sustains > 70% accuracy when 60% of bridge
mappings are deliberately corrupted (vs. the 40% interference level tested in
exp_13_1), because compositional structure provides redundant retrieval paths.
*Status: UNTESTED*

**H-24.3 — Energy-Gated Delta Rule is Dimension-Robust**
The energy-gated delta rule (exp_15_3) achieves the same accuracy-to-write-rate
efficiency ratio (within 5%) across HIDDEN_DIM ∈ {32, 64, 128}, confirming that
the energy criterion is not an artifact of low-dimensional geometry.
*Status: UNTESTED*

**H-24.4 — Four-Hop Chains Are Infeasible at HIDDEN_DIM=64**
Four-hop compositional chain accuracy drops > 50% relative to two-hop accuracy
at HIDDEN_DIM=64, and a hop-by-hop training curriculum does not close more than
10% of this gap, indicating a fundamental capacity limitation.
*Status: UNTESTED*

---

## Category 25: Hard Benchmarks (Phase 4)

**H-25.1 — Multi-Domain Retrieval Caps Below 70% Without Routing**
A single memory architecture confronted with mixed factual, pattern, and temporal
retrieval tasks in the same sequence achieves < 70% joint accuracy, while domain-
specific routing models achieve significantly higher accuracy per domain.
*Status: UNTESTED*

**H-25.2 — Slot Memory Degrades > 20% Under Query Noise; Parametric < 10%**
Adding Gaussian noise ε~N(0, 0.1) to the query embedding at test time degrades
slot memory by > 20% accuracy while degrading parametric memory by < 10%, because
gradient-adapted representations are inherently more noise-robust.
*Status: UNTESTED*

**H-25.3 — Temporal Ordering Accuracy Degrades Monotonically With Ordinal Position**
Retrieval accuracy for the kth event in a temporal ordering task decreases
monotonically with k (Pearson r < −0.7), reflecting the read bottleneck identified
in exp_7_7 applied to ordered sequence retrieval.
*Status: UNTESTED*

---

## Category 26: Seed Stability Validation (Phase 4)

**H-26.1 — Protected Slot Interior Optimum Is Seed-Stable**
The interior accuracy peak at K=3–6 protected slots (exp_9_4) replicates across
5 additional independent seeds (13, 99, 256, 512, 1024), confirming that the
optimum is not a seed artifact.
*Status: UNTESTED*

**H-26.2 — Write Budget Non-Uniform Allocation Is Seed-Stable**
The write budget advantage of oracle protected-slot allocation over uniform allocation
(exp_9_5) replicates across 5 additional independent seeds, confirming that non-
uniform budget allocation is a reliable accuracy lever.
*Status: UNTESTED*

**H-26.3 — Query-Conditioned Write Gate Advantage Is Seed-Stable**
The accuracy advantage of query-conditioned write gates over context-only gates
(exp_17_1) replicates across 5 additional independent seeds, warranting a Phase 5
architectural redesign of this mechanism.
*Status: UNTESTED*

---

## Category 27: Parametric-Delta Hybrid (Phase 4)

**H-27.1 — Isolated Pre-Training Enables Super-Additive Hybrid Memory**
A hybrid memory combining energy-gated delta-rule matrix and parametric MLP
components, each pre-trained independently for 200 steps before joint fine-tuning,
outperforms either component alone by > 10% and outperforms cold joint training
by > 2%, because pre-training isolates each component's loss landscape.
*Status: UNTESTED*

**H-27.2 — Delta-Rule and Parametric Components Spontaneously Specialize**
In a trained hybrid model, the delta-rule component achieves higher accuracy on
short-range (distance ≤ 4) retrieval than the parametric component, while the
parametric component achieves higher accuracy on long-range (distance ≥ 8) retrieval,
reflecting spontaneous division of labor.
*Status: UNTESTED*

**H-27.3 — Sequential Pre-Training Is Strictly Better Than Cold Joint Training**
The sequential pre-training strategy (delta pre-train → parametric pre-train →
joint fine-tune with lower LR and gradient clipping) yields strictly higher accuracy
than cold joint training at the same total step budget AND is super-additive relative
to the best single component.
*Status: UNTESTED*

---

## Category 28: Explicit Scaling Laws (Phase 5)

**H-28.1 — Parametric Memory Dominates at 8× Sequence Length**
Parametric memory retains >90% accuracy at SEQ_LEN=192 (8× baseline) while slot
memory drops below 30%, confirming a qualitative crossover point in length scaling.
*Status: ~ INCONCLUSIVE (exp_28_1)*

**H-28.2 — Hidden Dimension Power Law With α > 0.3 For Delta Rule**
Accuracy scales as dim^α with α > 0.3 for the energy-gated delta rule, confirmed by
log-log fit (R² > 0.95) across HIDDEN_DIM ∈ {32, 64, 128, 256}.
*Status: ~ INCONCLUSIVE (exp_28_2)*

**H-28.3 — Parametric Memory Has Steepest Per-Step Accuracy Slope**
Across STEPS ∈ {200, 400, 800, 1600, 3200}, parametric memory's per-step accuracy
gain exceeds both slot and delta rule models, confirming highest sample efficiency.
*Status: ~ INCONCLUSIVE (exp_28_3)*

**H-28.4 — Slot Count Peaks at 1.5–2× KV Pairs; Collapses Beyond**
Memory slot accuracy peaks when NUM_SLOTS is 1.5–2× NUM_PAIRS and degrades with
excess slots (slot collapse), identifying an optimal slot ratio.
*Status: ~ INCONCLUSIVE (exp_28_4)*

**H-28.5 — Vocabulary Size Does Not Affect Delta Rule Capacity**
Delta rule accuracy at fixed num_pairs varies by <2% across VOCAB_SIZE ∈ {32, 64,
128, 256}, confirming the mechanism is capacity-limited by hidden_dim, not vocab.
*Status: ✗ REFUTED (exp_28_5)*

---

## Category 29: TTT / Titans-Inspired Memory (Phase 5)

**H-29.1 — Outer-Product Linear Memory Matches Slot Memory Within 2%**
A pure outer-product linear associative memory (M += v⊗k / ||k||²) without any
test-time SGD matches slot memory accuracy within 2% at matched parameter count.
*Status: ✓ SUPPORTED (exp_29_1)*

**H-29.2 — Adam-at-Inference Outperforms SGD-at-Inference By >5%**
Replacing test-time SGD with Adam (storing m1, m2 momentum states per MLP weight)
for the parametric memory improves accuracy by >5% over SGD at the same inner steps.
*Status: ~ INCONCLUSIVE (exp_29_2)*

**H-29.3 — Gradient-Surprise-Gated TTT Achieves 90% Accuracy at <50% Update Rate**
Updating parametric memory only when the gradient-norm ratio (||∇||² / EMA||∇||²)
exceeds 1.5 achieves >90% of full-update accuracy at <50% update rate.
*Status: ✓ SUPPORTED (exp_29_3)*

**H-29.4 — Weight Decay at Inference Recovers >20% Lost Accuracy at 8× Length**
Applying L2 weight decay (wd=0.01) during test-time MLP updates prevents saturation;
accuracy at SEQ_LEN=192 is >20% higher with decay than without.
*Status: ~ INCONCLUSIVE (exp_29_4)*

---

## Category 30: Multi-Head & Extended Delta Rule (Phase 5)

**H-30.1 — 4-Head Delta Rule Outperforms Single-Head by >5% at 8-Pair Recall**
Multi-head delta rule (4 heads × H/4 dimensions, matched total params) outperforms
single-head by >5% on 8-pair associative recall.
*Status: ✓ SUPPORTED (exp_30_1)*

**H-30.2 — Momentum Delta Rule Matches Energy Gating With Lower Loss Variance**
Momentum delta (M_t = β×M_{t-1} + (1−β)×ΔM, β=0.9) achieves the same acc_ratio as
energy-gated delta (within 5%) while producing <80% of energy gating's loss variance.
*Status: ✗ REFUTED (exp_30_2)*

**H-30.3 — Bidirectional Delta Rule Improves Late-Query Accuracy By >8%**
Adding a backward retroactive delta pass (re-applying updates weighted by future
context similarity) improves late-query accuracy by >8% without hurting early queries.
*Status: ✗ REFUTED (exp_30_3)*

**H-30.4 — Energy-Gated Delta Pareto Knee Is Universally at 40–60% Write Rate**
The accuracy–write-rate Pareto frontier of energy-gated delta rule has its knee at
40–60% write rate across HIDDEN_DIMS ∈ {32, 64, 128}, confirming a universal optimum.
*Status: ~ INCONCLUSIVE (exp_30_4)*

---

## Category 31: Top Mechanism Integration (Phase 5)

**H-31.1 — Retroactive Writing + Two-Hop Retrieval Combined Beats Both by >5%**
Pre-training each mechanism independently then jointly fine-tuning achieves >5% higher
accuracy than either individually on combined two-hop + re-encoding tasks.
*Status: ✗ REFUTED (exp_31_1)*

**H-31.2 — Retroactive Re-Encoding Gap Persists >0.08 At 8× Sequence Length**
The +0.133 retroactive writing accuracy gap at SEQ_LEN=24 remains above +0.08 at
SEQ_LEN=192 (8× length, 10 KV pairs), confirming mechanism scalability.
*Status: ✗ REFUTED (exp_31_2)*

**H-31.3 — Delta Rule + Retroactive Re-Encoding Combined Beats Delta-Only by >8%**
Using delta rule for memory writes and retroactive cross-attention for slot refinement
achieves >8% higher accuracy than delta-only and >2% higher than re-encoding-only.
*Status: ~ INCONCLUSIVE (exp_31_3)*

**H-31.4 — Learned Eviction Policy + Parametric Memory Beats FIFO by >10%**
When parametric memory is capacity-limited to 8 KV pairs, applying a learned
importance eviction policy achieves >10% higher accuracy than first-in-first-out.
*Status: ~ INCONCLUSIVE (exp_31_4)*

---

## Category 32: Deep Seed Validation (Phase 5)

**H-32.1 — Retroactive Writing Gap >0.09 On ≥7 of 9 Seeds**
The exp_3_6 result (two-pass vs forward-only accuracy gap) replicates with gap >0.09
on at least 7 of 9 seeds {0,1,2,7,13,42,99,123,777}.
*Status: ✓ SUPPORTED (exp_32_1)*

**H-32.2 — Energy-Gated Delta acc_ratio >0.90 On ≥7 of 9 Seeds**
The exp_15_3 result (acc_ratio=0.919, write_rate=0.519) replicates with acc_ratio >0.90
and write_rate <0.70 on at least 7 of 9 seeds.
*Status: ✓ SUPPORTED (exp_32_2)*

**H-32.3 — Three-Hop Chain Retention >2× On ≥7 of 9 Seeds**
The exp_13_2 result (three-hop retention=4×) replicates with retention >2.0 on at
least 7 of 9 seeds, confirming that three-hop compositional retrieval is reliable.
*Status: ✓ SUPPORTED (exp_32_3)*

**H-32.4 — Parametric Memory Length Retention Gap >0.35 On ≥7 of 9 Seeds**
The exp_16_3 result (parametric retention gap=0.440 over slot memory) replicates with
gap >0.35 on at least 7 of 9 seeds {0,1,2,7,13,42,99,123,777}.
*Status: ✓ SUPPORTED (exp_32_4)*

---

## Category 33: Capacity Physics / Interference Density Law (Phase 5)

**H-33.1 — Slot Memory Accuracy Follows Power Law acc ~ ρ^(−γ) With R² > 0.90**
Slot memory accuracy decays as a power law of interference density ρ = N_pairs/hidden_dim
across ρ ∈ {0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0}, with log-log fit R² > 0.90.
*Status: ✗ REFUTED (exp_33_1)*

**H-33.2 — Architecture Interference Exponents Ordered: γ_parametric < γ_slot < γ_delta**
Fitting separate power laws to slot, parametric, and delta rule architectures yields
γ values ordered parametric < slot < delta with total spread > 0.3, identifying
parametric memory as the most interference-resistant design.
*Status: ~ INCONCLUSIVE (exp_33_2)*

**H-33.3 — Interference Exponent γ Is Independent of Hidden Dimension (±0.1)**
The power-law exponent γ fitted at H ∈ {32, 64, 128} differs by less than 0.1 for
each architecture, confirming γ is a property of the mechanism, not the scale.
*Status: ✗ REFUTED (exp_33_3)*

**H-33.4 — Tripling Training Budget Recovers >50% Of Capacity-Lost Accuracy**
At ρ=1.0 (N_pairs = hidden_dim), tripling STEPS from 400 to 1200 recovers >50%
of the accuracy difference between ρ=1.0 and ρ=0.5 for at least one architecture.
*Status: ✓ SUPPORTED (exp_33_4)*

---

## Category 34: Training Dynamics (Phase 6)

**H-34.1 — Delta Rule Shows Sharper Phase Transition Than Slot Memory**
Delta rule memory shows a phase transition (≥30% accuracy gain within 200 steps)
while slot memory does not — revealing a qualitative difference in learning dynamics.
*Status: ✗ REFUTED (exp_34_1)*

**H-34.2 — Memory Projections Receive Larger Gradients Early, Converge Late**
Memory projection parameters (k/v/q) receive gradient ratio (mem/enc) > 2.0 at step 100
and < 1.5 at step 1000, indicating early specialisation of memory writes.
*Status: ~ INCONCLUSIVE (exp_34_2)*

**H-34.3 — Energy-Gated Write Rate Decreases Naturally During Training**
The energy-gated delta rule write rate decreases from ≥0.70 at step 100 to ≤0.40
at step 1500 as the model learns more accurate representations.
*Status: ~ INCONCLUSIVE (exp_34_3)*

**H-34.4 — Easy-First Curriculum Outperforms Random Training By >0.08**
Training with an easy-first curriculum (2→8 pairs over 1500 steps) improves final
accuracy vs random mixed training by >0.08 on a 4-pair evaluation task.
*Status: ~ INCONCLUSIVE (exp_34_4)*

**H-34.5 — Memory Warmup Improves Accuracy By >3%**
Gradual memory warmup (write scale 0→1 over first 200 steps) improves final accuracy
by >3% over full-memory training from step 0, allowing backbone pre-training.
*Status: ✗ REFUTED (exp_34_5)*

**H-34.6 — Delta Rule Shows Strong Optimizer Preference (>10% Spread)**
The best optimizer outperforms the worst by >10% accuracy across Adam, AdamW, SGD,
SGD+momentum, and RMSprop, indicating higher optimizer sensitivity than typical
architectures.
*Status: ✓ SUPPORTED (exp_34_6)*

**H-34.7 — Delta Rule Has Narrow Stable LR Band (<1.5 Decades)**
The stable learning rate band (LRs achieving ≥50% of peak accuracy) spans <1.5 decades,
indicating higher LR sensitivity than standard architectures (which have 2+ stable decades).
*Status: ✗ REFUTED (exp_34_7)*

**H-34.8 — Memory Quality Degrades at Larger Batch Sizes (>5% Drop B8→B128)**
Accuracy at B=128 is >5% lower than B=8 even with proportional LR scaling, indicating
batch-size-dependent memory quality independent of gradient noise.
*Status: ✗ REFUTED (exp_34_8)*

**H-34.9 — Gate Dead-Zone Collapse: >40% Activations Bimodal at Convergence**
More than 40% of learned write gate activations are in the dead zone (<0.05) or
saturated zone (>0.95) at convergence, indicating bimodal gate collapse rather than
graded, informative gating.
*Status: ✗ REFUTED (exp_34_9)*

---

## Category 35: Failure Modes (Phase 6)

**H-35.1 — Delta Rule Degrades Gracefully Under Post-Hoc Noise Injection**
Accuracy at 30% additive noise to the memory matrix M is >60% of the clean baseline,
not a catastrophic cliff — the delta rule writes are robust to partial corruption.
*Status: ✗ REFUTED (exp_35_1)*

**H-35.2 — Slot Memory Does Not Hallucinate on OOD Queries**
Querying with keys never presented in the context produces accuracy near random chance
(< random + 5%), confirming that slot memory does not fabricate associations.
*Status: ✓ SUPPORTED (exp_35_2)*

**H-35.3 — OOD Inputs Cause Abnormal Write Gate Behavior (>2× Rate Deviation)**
Out-of-distribution tokens cause write gate activations that deviate by more than 2×
from in-distribution rates, revealing non-robustness to distributional shift.
*Status: ✓ SUPPORTED (exp_35_3)*

---

## Category 36: Biological Analogues (Phase 6)

**H-36.1 — Offline Consolidation Improves Recall By >3%**
Replaying all written key-value pairs offline (without new input) improves associative
recall accuracy by >3% over single-pass writing — analogous to hippocampal-cortical
consolidation during sleep.
*Status: ✗ REFUTED (exp_36_1)*

**H-36.2 — Predictive Coding Residuals Match Full Representation Performance**
Storing prediction residuals (what the model predicted wrong) rather than full token
representations produces equivalent or better associative recall accuracy with the
same memory capacity.
*Status: ~ INCONCLUSIVE (exp_36_2)*

**H-36.3 — Split Episodic/Semantic Memory Outperforms Unified by >5%**
Separating episodic (temporal/event order) memory from semantic (content association)
memory outperforms a unified memory store by >5% on tasks requiring both recall types.
*Status: ✓ SUPPORTED (exp_36_3)*

---

## Category 37: Robustness (Phase 7)

**H-37.1 — Noise-Augmented Training Reduces Catastrophic Memory Collapse**
Training DeltaModel with Gaussian noise (σ=0.05) on M at read-time reduces the
accuracy collapse at σ_test=0.10 from ratio=0.08 (exp_35_1 baseline) to ≥0.50,
showing that noise augmentation provides denoising regularization on M.
*Status: ~ INCONCLUSIVE (exp_37_1) — Normalized-key delta already resilient (ratio~0.95); augmentation Δ≈+0.04 but below 0.10 threshold*

**H-37.2 — Row-Normalizing M After Each Write Resists Noise Collapse**
Normalizing each row of M to unit L2 norm after each delta-rule update bounds M
magnitude; the accuracy-retention ratio at σ=0.10 rises to ≥0.50 vs 0.08 baseline.
Rationale: bounded M.std() makes additive noise less destructive relatively.
*Status: ~ INCONCLUSIVE (exp_37_2) — Baseline already robust (ratio~0.86-1.01); row-norm neither helps nor hurts consistently*

**H-37.3 — EMA-Smoothed M Updates Improve Noise Resistance**
Applying exponential moving average (α=0.85) to M updates — writing only (1-α) of
each new delta — produces a lower spectral-radius M; acc_ratio at σ=0.10 ≥ 0.50
while retaining ≥95% of clean accuracy.
*Status: ✓ SUPPORTED (exp_37_3) — EMA (α=0.85-0.95) achieves ratio≥0.96 AND improves clean accuracy by 5-10%*

---

## Category 38: Episodic/Semantic Architecture (Phase 7)

**H-38.1 — Learned Soft Routing Outperforms Fixed 50/50 Episodic/Semantic Split**
A trainable logistic gate g_t ∈ (0,1) that blends episodic vs semantic write weights
per-timestep outperforms the fixed 50/50 split of exp_36_3 by >5%, indicating the
model can route content to the more appropriate memory module.
*Status: ✗ REFUTED (exp_38_1) — Fixed split beats learned router by 10-24%. Inductive bias (recency-weighted episodic) outperforms learned routing*

**H-38.2 — Asymmetric Capacity (25% Episodic, 75% Semantic) Outperforms 50/50**
Allocating only 25% of HIDDEN_DIM to episodic memory and 75% to semantic outperforms
the symmetric split by >3%, because the primary task (KV content recall) is semantic.
*Status: ~ INCONCLUSIVE (exp_38_2) — Seed-dependent: 1/3 seeds show 25% epi wins (+12%), 2/3 show 50/50 optimal*

**H-38.3 — Learned Gated Read Combination Outperforms Simple Concatenation**
Using a query-conditioned softmax gate over [M_sem_read, M_epi_read] outputs improves
accuracy by >5% over direct concatenation, allowing the model to selectively attend to
the more informative memory module per query.
*Status: ~ INCONCLUSIVE (exp_38_3) — Seed-dependent results; gated wins by +22% on seed 777 but loses on seed 123*

---

## Category 39: Write Controller Adaptation (Phase 7)

**H-39.1 — Write Rate Sweet Spot Near 0.50 (Concave Accuracy-vs-Rate Curve)**
Forcing write rate to 0.10 or 0.90 via threshold scaling each degrades accuracy by
>10% relative to the near-optimal write rate, confirming a concave accuracy curve
peaking near 0.50 and validating the observed equilibrium ~0.54 as near-optimal.
*Status: ~ INCONCLUSIVE (exp_39_1) — Asymmetric curve: low write rate hurts (-21%) but high write rate shows no penalty. One-sided concavity only*

**H-39.2 — Write Rate Equilibrium Increases With Interference Density ρ**
The steady-state write rate of EnergyGatedDelta rises with ρ = N_pairs/HIDDEN_DIM:
wr(ρ=0.75) exceeds wr(ρ=0.08) by >0.15, showing the locked-at-0.534 finding
(exp_34_3, ρ≈0.078) is task-load-dependent, not universally fixed.
*Status: ~ INCONCLUSIVE (exp_39_2) — Moderate variation (Δ=0.083, below 0.15 threshold); write rate 0.70-0.78 across ρ, though not conclusively density-driven*

**H-39.3 — Learnable Threshold Converges to 0.40–0.55 From Any Initialization**
When the energy gate threshold is a learnable parameter optimized jointly with model
weights, it converges from any initial value {0.05, 0.20, 0.50, 0.80, 1.20} to the
range [0.40, 0.55], confirming ~0.54 as a gradient-dynamics attractor.
*Status: ✗ REFUTED (exp_39_3) — No convergence: spread=1.022, final thresholds stay near initialization [0.049, 0.188, 0.464, 0.814, 1.072]. System is multi-stable, not attracted to 0.54*
