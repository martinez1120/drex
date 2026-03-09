# drex Research — Master Results Report

**Generated:** 2026-03-09 19:20 UTC
**Experiments:** 176  |  **Seeds per experiment:** 0, 1, 2, 7, 13, 42, 99, 123, 777
**Total runs evaluated:** 725

## Overall Scoreboard

| Outcome | Count | % |
|---------|-------|---|
| ✓ SUPPORTED    | 43    | 24% |
| ~ INCONCLUSIVE | 57 | 32% |
| ✗ REFUTED      | 76      | 43% |
| ! ERROR        | 0        | 0% |

**Seed consistency:** 129/176 experiments gave the same verdict across all seeds. 47 inconsistent.

## Summary Table

| ID | Outcome | Consistent | Key Metric (mean ± std) | Notes |
|----|---------|------------|------------------------|-------|
| exp_1_1 | ✗ REFUTED | ✓ | attention_correlation=-0.503±0.000 | Pearson r=-0.5026. Attention acc=0.088 vs random=0.087 vs or… |
| exp_1_2 | ✗ REFUTED | ✓ | attention_acc=0.144±0.000 | Surprise acc=0.139, attention acc=0.144, no-memory baseline=… |
| exp_1_3 | ~ INCONCLUSIVE | ✓ | attention_acc=0.090±0.000 | Gradient acc=0.107, attention=0.090, random=0.109. Gap=+0.01… |
| exp_1_4 | ~ INCONCLUSIVE | ✓ | diversity_acc=0.138±0.000 | Diversity acc=0.138 vs importance acc=0.144. Gap=-0.0067. Di… |
| exp_1_5 | ✓ SUPPORTED | ✓ | acc_attention=0.121±0.000 | Learned gate delta over best baseline: +0.003. Ranking: ['le… |
| exp_1_6 | ~ INCONCLUSIVE | ✓ | coverage_delta=0.000±0.000 | Dedup: precision=0.104 (std=0.120), coverage=1.000 (std=1.00… |
| exp_1_7 | ~ INCONCLUSIVE | ✓ | freq_acc=0.066±0.000 | Infreq acc=0.058 vs freq acc=0.066. Gap=-0.0076. Budget: fre… |
| exp_1_8 | ✗ REFUTED | ✓ | gap_two_minus_single=-0.015±0.000 | Two-stage acc=0.131 vs single-stage acc=0.147. Gap=-0.0151. … |
| exp_2_1 | ~ INCONCLUSIVE | ✓ | cliff_after_ratio=64.000±0.000 | Largest quality drop: 0.051 between 32x and 64x compression.… |
| exp_2_2 | ✗ REFUTED | ✓ | exact_cosim_attention=0.321±0.000 | Inferential recall: attention=0.990, autoenc=0.998, gain=-0.… |
| exp_2_3 | ~ INCONCLUSIVE | ✓ | avg_error_approx=0.000±0.000 | avg_error_exact=0.0000, avg_error_approx=0.0004, diff (appro… |
| exp_2_4 | ✗ REFUTED | ✓ | cosine_sim.16=0.340±0.000 | Quality scores [0.642, 0.473, 0.34, 0.227, 0.153] for chunk … |
| exp_2_5 | ✗ REFUTED | ✓ | acc_gain_structured_over_flat=-0.313±0.000 | Structured acc=0.056, flat acc=0.369, gain=-0.313. Structure… |
| exp_2_6 | ✓ SUPPORTED | ✓ | comp_a_on_a=0.610±0.000 | Comp_A: in-domain=0.610, cross-domain=0.221, drop=0.390 (thr… |
| exp_2_7 | ~ INCONCLUSIVE | ✓ | cosim_stage1=0.487±0.000 | Cosine similarities after iterative compression: stage1=0.48… |
| exp_2_8 | ~ INCONCLUSIVE | ✓ | quality_a_in_mixed_seq=0.609±0.000 | Domain A (in-distribution): 0.605. Domain B (shifted): 0.290… |
| exp_2_9 | ✗ REFUTED | ✓ | recon_a_cosine=0.413±0.000 | Reconstruction A=0.414 vs B=0.397 (gap 0.016). Retrieval A=1… |
| exp_3_1 | ✓ SUPPORTED | ✓ | acc_delta=0.005±0.000 | Event-driven acc delta: +0.005. Coverage delta: +0.059. Stri… |
| exp_3_2 | ✗ REFUTED | ✓ | A_no_signal.final_write_rate=0.193±0.000 | Regime A collapsed: False. Other regimes collapsed: True. Wr… |
| exp_3_3 | ✗ REFUTED | ✓ | acc_gap=0.000±0.000 | Late vs early accuracy gap: +0.000. Threshold for SUPPORTED:… |
| exp_3_4 | ✗ REFUTED | ✓ | acc_gap=-0.173±0.000 | Boundary vs fixed accuracy gap: -0.173. Boundary per-segment… |
| exp_3_5 | ✓ SUPPORTED | ✓ | acc_at_latency_0=0.216±0.000 | Baseline (L=0): 0.216. Accuracy range across latencies: 0.20… |
| exp_3_6 | ✓ SUPPORTED | ✓ | acc_gap=0.133±0.000 | Two-pass vs forward-only accuracy gap: +0.133. Retroactive w… |
| exp_3_7 | ✗ REFUTED | ✓ | acc_delta=0.027±0.000 | Adaptive vs uniform accuracy delta: +0.027. Allocation Gini … |
| exp_4_1 | ✓ SUPPORTED | ✓ | direct_acc=0.047±0.000 | Learned acc=0.0709 vs Direct acc=0.0469, gap=+0.0241 (thresh… |
| exp_4_2 | ✗ REFUTED | ✓ | gap_multi_minus_single=-0.020±0.000 | Multi acc=0.9545 vs Single acc=0.9744, gap=-0.0199 (threshol… |
| exp_4_3 | ~ INCONCLUSIVE | ✓ | acc_at_k_1=0.016±0.000 | Accuracy is flat across k values (range=0.0026).… |
| exp_4_4 | ✓ SUPPORTED | ✓ | hard_final_acc=0.017±0.000 | Soft var=0.000033 vs Hard var=0.000076; Soft acc=0.0164 vs H… |
| exp_4_5 | ~ INCONCLUSIVE | ✓ | avg_tiers_queried_sequential=3.000±0.000 | Simultaneous acc=0.0163 vs Sequential acc=0.0156, gap=+0.000… |
| exp_4_6 | ~ INCONCLUSIVE | ✓ | recon_exact_acc=0.015±0.000 | Exact(Sim=0.0163, Rec=0.0145), Inferential(Sim=0.0289, Rec=0… |
| exp_4_7 | ✓ SUPPORTED | ✓ | null_f1=0.889±0.000 | Null precision 1.000 vs threshold 0.7.… |
| exp_4_8 | ~ INCONCLUSIVE | ✓ | acc_at_N_0=0.017±0.000 | Accuracy is flat across N values; no interference detected.… |
| exp_4_9 | ✓ SUPPORTED | ✓ | compositional_gap=-0.032±0.000 | Single-hop=0.968, Two-hop=1.000, Gap=-0.032, Random=0.062.… |
| exp_5_1 | ✗ REFUTED | ✓ | A_task_only.final_read_rate=0.156±0.000 | Regime A collapsed: False (stable). Read rates: A_task_only=… |
| exp_5_2 | ✓ SUPPORTED | ✓ | factual_qa_freq1=0.996±0.000 | Optimal frequencies — task1(factual_qa):4, task2(seq_complet… |
| exp_5_3 | ~ INCONCLUSIVE | ✓ | acc_gap=0.006±0.000 | Reactive: acc=0.990, read_rate=0.125. Predictive: acc=0.984,… |
| exp_5_4 | ✗ REFUTED | ✓ | rate_diff=0.000±0.000 | Type A (cheap/recompute): retrieval_rate=0.000, acc=1.000. T… |
| exp_5_5 | ✗ REFUTED | ✓ | cascading_acc=0.396±0.000 | Full-depth: acc=0.961, tiers=3.00. Cascading: acc=0.396, tie… |
| exp_5_6 | ✓ SUPPORTED | ✓ | acc_at_T_50=0.972±0.000 | Baseline acc (no suppression): 0.997. Optimal T=0.8, quality… |
| exp_5_7 | ✗ REFUTED | ✓ | arbitrated_acc=0.482±0.000 | attn_only_acc=0.511, mem_only_acc=0.501, arbitrated_acc=0.48… |
| exp_6_1 | ✓ SUPPORTED | ✓ | learned_acc=0.059±0.000 | Learned vs LRU gap: 0.040. Threshold for SUPPORTED: >0.03. R… |
| exp_6_2 | ~ INCONCLUSIVE | ✓ | avg_compression_level_at_query_time=1.933±0.000 | Compression vs LRU gap: -0.004. Average compression level (0… |
| exp_6_3 | ✓ SUPPORTED | ✓ | gap_selective_minus_lru=0.076±0.000 | Selective vs LRU gap on phase-2 queries: 0.076. Eviction rat… |
| exp_6_4 | ~ INCONCLUSIVE | ✓ | acc_at_K_0=0.018±0.000 | Optimal K=5 with acc=0.063. K=0 acc=0.018, K=5 acc=0.063. In… |
| exp_6_5 | ~ INCONCLUSIVE | ✓ | ebbinghaus_acc=0.024±0.000 | Ebbinghaus vs LRU gap: 0.004. Learned S=11.367 steps. Mean r… |
| exp_6_6 | ~ INCONCLUSIVE | ✓ | acc_a_after_ewc=0.023±0.000 | Standard forgetting: 0.208. EWC forgetting: 0.204. Significa… |
| exp_6_7 | ✗ REFUTED | ✓ | gap_joint_minus_independent=-0.035±0.000 | Joint vs independent gap: -0.035. Write-evict correlation — … |
| exp_6_8 | ~ INCONCLUSIVE | ✓ | consolidation_acc=0.021±0.000 | Consolidation vs no-consolidation gap: 0.003. Consolidation … |
| exp_7_1 | ✓ SUPPORTED | ✓ | Gumbel.accuracy=0.128±0.000 | Gumbel most stable: True. Both beat REINFORCE: True. Acc: ST… |
| exp_7_2 | ✓ SUPPORTED | ✓ | acc_per_complexity.Large=0.108±0.000 | Peak efficiency at: Medium. Efficiency ratios: {'Tiny': 0.0,… |
| exp_7_3 | ~ INCONCLUSIVE | ✓ | factual_acc=0.094±0.000 | Reasoning gap=0.040 (threshold <0.15: True). Generation gap=… |
| exp_7_4 | ~ INCONCLUSIVE | ✓ | acc_per_depth.0=0.122±0.000 | Min depth for meaningful behavior: None. depth=0 meaningful:… |
| exp_7_5 | ✗ REFUTED | ✓ | finetuned_transfer_acc=0.101±0.000 | Transfer gap (fresh - zero_shot): 0.028. Supported threshold… |
| exp_7_6 | ✗ REFUTED | ✓ | adversarial_ratio=1.015±0.000 | Adversarial ratio: 1.015 (threshold >1.5: False). Self-corre… |
| exp_7_7 | ✗ REFUTED | ✓ | early_averages.compression_fidelity=0.520±0.000 | Bottleneck at 20%: read_accuracy. Bottleneck at 50%: read_ac… |
| exp_7_8 | ~ INCONCLUSIVE | ✓ | curriculum_acc=0.075±0.000 | Curriculum acc >= joint acc: False. Curriculum loss_var < jo… |
| exp_7_9 | ✓ SUPPORTED | ✓ | interpretability_score_trained=0.059±0.000 | Interp score: trained=0.059 untrained=0.055. Gate is non-ran… |
| exp_8_1 | ✗ REFUTED | ⚠ ['INCONCLUSIVE', 'REFUTED', 'REFUTED'] | pearson_r_entropic=-0.077±0.446 | Softmax r=0.2286, raw dot-product r=-0.0775, entropy-normali… |
| exp_8_2 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE'] | pearson_r_difficulty_vs_rate=0.398±0.405 | KV levels [1, 2, 4, 6]: write rates [0.2829, 0.6731, 0.5961,… |
| exp_8_3 | ✗ REFUTED | ✓ | acc_A=0.212±0.051 | Condition A (joint): corr=0.1756, acc=0.272. Condition B (or… |
| exp_8_4 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED'] | acc_A=0.150±0.110 | A (full): acc=0.216, pos_r=-0.1920. B (pos-blind): acc=0.036… |
| exp_9_1 | ✗ REFUTED | ✓ | acc_at1_autoencoder=1.000±0.000 | AE Acc@1=1.0000, CL Acc@1=1.0000, gap=0.0000. Recon cosim: A… |
| exp_9_2 | ✓ SUPPORTED | ✓ | null_f1_A=0.812±0.011 | Cond A (learned, p=0.5): null_f1=0.800, retr_recall=0.825, d… |
| exp_9_3 | ~ INCONCLUSIVE | ✓ | acc_a_after_ewc=0.087±0.066 | Precondition failed: acc_A_before=0.031 < 0.70.… |
| exp_9_4 | ✓ SUPPORTED | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED'] | acc_range=0.149±0.003 | K values [0, 2, 4, 6, 8, 10], accs [0.152, 0.144, 0.022, 0.0… |
| exp_9_5 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'SUPPORTED', 'REFUTED'] | acc_delta=0.018±0.139 | Uniform acc=0.159, adaptive acc=0.227, delta=0.068. Block-1 … |
| exp_10_1 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE'] | acc_w0=0.259±0.007 | gap@24=-0.059, gap@4=-0.100 — pattern inconclusive.… |
| exp_10_2 | ✗ REFUTED | ✓ | acc_A=0.230±0.000 | Overwrite gain (0.002) exceeds new-write gain (-0.031) by >0… |
| exp_10_3 | ~ INCONCLUSIVE | ✓ | acc_fwd_24=0.188±0.000 | Pearson r=0.773 — scaling relationship inconclusive.… |
| exp_11_1 | ✗ REFUTED | ✓ | normal_acc_A=0.211±0.000 | oracle_read_acc_B=0.175 not > oracle_read_acc_A=0.292 by 0.0… |
| exp_11_2 | ✗ REFUTED | ✓ | f1_A=0.343±0.000 | F1_B=0.309 not > F1_A=0.343 by 0.01.… |
| exp_11_3 | ✓ SUPPORTED | ✓ | acc_copy_T0.2=0.061±0.000 | Max pairwise T difference=0.750>0.15. Optimal T: factual=0.2… |
| exp_12_1 | ✗ REFUTED | ✓ | acc1_A=1.000±0.000 | Both models >80% Acc@1 or gap <5%.… |
| exp_12_2 | ~ INCONCLUSIVE | ✓ | a_fails_at_2x_4x=0.000±0.000 | A does not fail at 2x-4x; hypothesis conditions not triggere… |
| exp_13_1 | ✓ SUPPORTED | ✓ | interference_gap=0.120±0.000 | Two-hop acc within 5% of single-hop on interference subset (… |
| exp_13_2 | ✓ SUPPORTED | ✓ | acc_single=0.042±0.018 | Three-hop retains 4.00 of two-hop accuracy (>0.50).… |
| exp_14_1 | ✗ REFUTED | ✓ | acc_A=0.211±0.000 | gap_D=-0.138 < max(gap_B,gap_C)+0.005=-0.001.… |
| exp_14_2 | ✗ REFUTED | ✓ | acc_A=0.209±0.000 | Joint training A >= curriculum B - 0.01: acc_A=0.209, acc_B=… |
| exp_14_3 | ✗ REFUTED | ✓ | acc_A=0.191±0.000 | Constant A >= cosine C - 0.01: acc_A=0.191, acc_C=0.153.… |
| exp_15_1 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_delta=0.106±0.017 | Slot memory acc=0.105 >= delta acc=0.109 - 0.02.… |
| exp_15_2 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | delta_acc_clean=0.110±0.009 | Hebbian acc_int=0.096 within 0.03 of delta acc_int=0.106.… |
| exp_15_3 | ✓ SUPPORTED | ✓ | acc_A_continuous=0.131±0.029 | Energy gate: acc_ratio=0.919>0.90, write_rate=0.519<0.70. Hy… |
| exp_15_4 | ~ INCONCLUSIVE | ⚠ ['REFUTED', 'INCONCLUSIVE', 'INCONCLUSIVE'] | delta_acc_normal=0.166±0.016 | Larimar acc_update=0.177 >= delta acc_update=0.169 - 0.02.… |
| exp_16_1 | ✗ REFUTED | ✓ | acc_gap=-0.012±0.011 | Slot acc=0.045 >= parametric acc=0.045 - 0.02.… |
| exp_16_2 | ~ INCONCLUSIVE | ✓ | baseline_acc=0.042±0.004 | Smooth degradation curve; no clear pareto knee found.… |
| exp_16_3 | ✓ SUPPORTED | ✓ | acc_parametric.24=0.035±0.011 | Parametric retention=1.000 vs slot retention=0.560; diff=0.4… |
| exp_17_1 | ✗ REFUTED | ⚠ ['REFUTED', 'SUPPORTED', 'REFUTED'] | acc_A=0.142±0.082 | Context-only gate matches or beats query-conditioned (gap=-0… |
| exp_17_2 | ✗ REFUTED | ⚠ ['REFUTED', 'SUPPORTED', 'REFUTED'] | acc_K0=0.130±0.068 | All lookahead K within 0.02 of K=0 (best gap=-0.014)… |
| exp_17_3 | ~ INCONCLUSIVE | ✓ | acc_A=0.147±0.073 | gap_B=-0.133 or gap_C=-0.127 too small for valid comparison… |
| exp_17_4 | ✗ REFUTED | ⚠ ['REFUTED', 'INCONCLUSIVE', 'REFUTED'] | acc_A_p000=0.148±0.076 | Gap is effectively constant (variance=0.000077)… |
| exp_18_1 | ✗ REFUTED | ✓ | acc_A=0.045±0.012 | Flat memory matches tiered (gap=-0.003)… |
| exp_18_2 | ~ INCONCLUSIVE | ✓ | corr_access=0.212±0.024 | corr_access=0.237 > 0.15 but corr_recency=0.175 > -0.10… |
| exp_18_3 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE'] | acc_flat_128=0.047±0.024 | Crossover at 8 outside expected range 16-64… |
| exp_18_4 | ~ INCONCLUSIVE | ✓ | acc_A=0.032±0.001 | Simultaneous best but gap=-0.066 < 0.03… |
| exp_19_1 | ✗ REFUTED | ✓ | acc_A=0.023±0.000 | Soft attention matches or beats sparse on interference tasks… |
| exp_19_2 | ✗ REFUTED | ✓ | acc_A=0.129±0.000 | Energy writes too frequent (0.8214) or acc dropped vs learne… |
| exp_19_3 | ✗ REFUTED | ✓ | acc_soft_12=0.020±0.000 | Capacity cliff differs by only 0 (< 1).… |
| exp_20_1 | ~ INCONCLUSIVE | ✓ | acc_A=0.167±0.000 | Auxiliary adjusts write rate but accuracy gain (0.0219) belo… |
| exp_20_2 | ✗ REFUTED | ✓ | acc_A=0.167±0.000 | Task acc unchanged (gap=0.0019 <= 0.005).… |
| exp_20_3 | ✗ REFUTED | ✓ | acc_A=0.235±0.000 | No-auxiliary baseline (acc_A=0.2353) nearly matches full sys… |
| exp_20_4 | ~ INCONCLUSIVE | ✓ | acc_lam_0_0=0.166±0.000 | Peak at lambda=0.01 (in_range=True); boundary conditions not… |
| exp_21_1 | ✗ REFUTED | ✓ | acc_A=0.031±0.004 | LSTM: acc=0.0269 util=0.1542. FF: acc=0.0384 util=0.0175. ac… |
| exp_21_2 | ✗ REFUTED | ✓ | acc_A=0.032±0.004 | Task-only: acc=0.0275. Oracle-augmented: acc=0.0331. gap=0.0… |
| exp_21_3 | ✗ REFUTED | ✓ | acc_A=0.032±0.004 | A(e2e): acc=0.0275 gq=-0.0108. B(distilled): acc=0.0244 gq=0… |
| exp_21_4 | ✗ REFUTED | ✓ | acc_A=0.035±0.005 | A(LSTM+task)=0.0325  B(FF+task)=0.0394  C(FF+oracle)=0.0319 … |
| exp_22_1 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_A=0.281±0.024 | Standard read outperforms slot-conditioned by 0.036.… |
| exp_22_2 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_A=0.281±0.024 | Single-pass outperforms iterative by 0.048.… |
| exp_22_3 | ~ INCONCLUSIVE | ✓ | acc_A=0.281±0.024 | Gap=0.005, collapse_reduction=0.065.… |
| exp_22_4 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_gate_easy=0.272±0.186 | Easy gap=0.476 too large or hard benefit negative (-0.053).… |
| exp_22_5 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_A=0.285±0.010 | Standard outperforms contrastive by 0.249.… |
| exp_23_1 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE'] | acc_A=0.263±0.014 | Gap=0.015, between -0.02 and +0.03.… |
| exp_23_2 | ✓ SUPPORTED | ⚠ ['REFUTED', 'INCONCLUSIVE', 'SUPPORTED'] | acc_0pass=0.201±0.044 | No diminishing returns: ratio=1.396>0.80.… |
| exp_23_3 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'SUPPORTED', 'INCONCLUSIVE'] | acc_A=0.201±0.044 | Selective: acc_ratio=1.000>0.90 at reenc_rate=0.000<0.60.… |
| exp_23_4 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'SUPPORTED'] | acc_factual_base=0.201±0.044 | No task-type specificity: ratio=-8.436<0.5.… |
| exp_24_1 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED'] | acc_parametric.24=0.035±0.011 | Param retention_4x=1.876, slot acc_96=0.019. Threshold not f… |
| exp_24_2 | ✓ SUPPORTED | ✓ | retention_at_60pct_interf=0.789±0.000 | Two-hop acc at 60% interference=0.789>0.70.… |
| exp_24_3 | ✗ REFUTED | ✓ | max_ratio=0.413±0.009 | Ratio spread=0.387>0.15. Mechanism is dim-sensitive.… |
| exp_24_4 | ✗ REFUTED | ✓ | acc_2hop=1.000±0.000 | 4-hop achievable: drop=18.7%<20% vs 2-hop.… |
| exp_25_1 | ✓ SUPPORTED | ✓ | acc_domain_specific=0.064±0.026 | Generic acc=0.145<0.70. Domain-specific: 0.048.… |
| exp_25_2 | ✗ REFUTED | ⚠ ['REFUTED', 'SUPPORTED', 'REFUTED'] | param_accs_by_sigma.0.0=0.033±0.009 | Slot is noise-robust (deg=-0.584) — no architecture gap.… |
| exp_25_3 | ✗ REFUTED | ⚠ ['REFUTED', 'INCONCLUSIVE', 'REFUTED'] | acc_by_k.1=0.131±0.112 | Temporal ordering is nearly flat (drop=-0.049<0.05).… |
| exp_26_1 | ✓ SUPPORTED | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED'] | acc_range=0.149±0.003 | K values [0, 2, 4, 6, 8, 10], accs [0.152, 0.144, 0.022, 0.0… |
| exp_26_2 | ✗ REFUTED | ⚠ ['REFUTED', 'INCONCLUSIVE', 'REFUTED'] | acc_learned=0.174±0.094 | Uniform=0.256, Oracle=0.255, Learned=0.231. Oracle gap=-0.00… |
| exp_26_3 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_A=0.140±0.081 | Context-only=0.221, QueryCond=0.033, gap=-0.188, qtype_pred=… |
| exp_27_1 | ✗ REFUTED | ✓ | acc_cold_joint=0.121±0.027 | Delta=0.075, Param=0.008, Cold=0.117, PretrainedHybrid=0.033… |
| exp_27_2 | ~ INCONCLUSIVE | ✓ | delta_long=0.106±0.017 | Delta: short=0.087, long=0.087. Param: short=0.031, long=0.0… |
| exp_27_3 | ✗ REFUTED | ✓ | acc_cold_joint=0.121±0.027 | A=0.075, B=0.008, C(cold)=0.117, D(seq)=0.037, superadd=0.50… |
| exp_28_1 | ~ INCONCLUSIVE | ✓ | acc_delta_len192=0.180±0.026 | Slot retention 8×=1.702, param retention 8×=0.737. Slot@192=… |
| exp_28_2 | ~ INCONCLUSIVE | ⚠ ['REFUTED', 'INCONCLUSIVE', 'INCONCLUSIVE'] | acc_delta_H128=0.217±0.015 | Delta: α=0.228, R²=0.450. Slot: α=-0.618, R²=0.503. Param: α… |
| exp_28_3 | ~ INCONCLUSIVE | ✓ | acc_delta_s1600=0.230±0.003 | Slopes (×10³/step): slot=-0.0004, delta=0.0100, param=-0.005… |
| exp_28_4 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED'] | acc_k12=0.130±0.040 | Peak at k=16 (acc=0.174). k=24 acc=0.148 (drop=0.026).… |
| exp_28_5 | ✗ REFUTED | ✓ | acc_mean_n2=0.480±0.009 | Max variance across vocab sizes = 0.1703 (>0.02). Vocab-inde… |
| exp_29_1 | ✓ SUPPORTED | ✓ | acc_linear=0.253±0.018 | Linear acc=0.261, slot acc=0.028, gap=+0.234.… |
| exp_29_2 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE'] | acc_adam=0.076±0.020 | Adam acc=0.096, SGD acc=0.062, gap=+0.033.… |
| exp_29_3 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'SUPPORTED', 'REFUTED'] | acc_full=0.031±0.015 | Full acc=0.019, Surprise acc=0.033, No-TTT acc=0.029. acc_ra… |
| exp_29_4 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED'] | acc_gain_wd_192=-0.002±0.006 | Gain at SEQ_LEN=192: +0.003. Gaps: len24=-0.009, len96=+0.01… |
| exp_30_1 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'INCONCLUSIVE', 'SUPPORTED'] | acc_multi=0.258±0.097 | Multi-head acc=0.366, Single-head acc=0.152, gap=0.214. Para… |
| exp_30_2 | ✗ REFUTED | ✓ | acc_energy=0.183±0.016 | Momentum acc=0.228, Energy acc=0.171, gap=0.057. Var ratio=1… |
| exp_30_3 | ✗ REFUTED | ⚠ ['REFUTED', 'INCONCLUSIVE', 'REFUTED'] | acc_bidir_early=0.182±0.007 | Late: bidir=0.169, unidir=0.188, improvement=-0.019. Early: … |
| exp_30_4 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'INCONCLUSIVE', 'SUPPORTED'] | all_knees_in_range=0.333±0.577 | Knee write rates: H32=0.612, H64=0.599, H128=0.593. All in [… |
| exp_31_1 | ✗ REFUTED | ✓ | acc_combined=0.163±0.060 | Retro=0.647, TwoHop=0.718, Combined=0.135. Gain vs retro: -0… |
| exp_31_2 | ✗ REFUTED | ⚠ ['INCONCLUSIVE', 'REFUTED', 'REFUTED'] | acc_forward_len192=0.129±0.015 | Gaps: len24=-0.027, len48=-0.033, len96=-0.023, len192=0.017… |
| exp_31_3 | ~ INCONCLUSIVE | ✓ | acc_delta=0.160±0.012 | Delta=0.154, Retro=0.129, DeltaRetro=0.179. Gain vs delta: +… |
| exp_31_4 | ~ INCONCLUSIVE | ✓ | acc_fifo=0.221±0.055 | FIFO=0.214, Learned=0.241, Gap=+0.028. Capacity=8, NumPairs=… |
| exp_32_1 | ✓ SUPPORTED | ⚠ ['REFUTED', 'SUPPORTED', 'SUPPORTED', 'INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED', 'REFUTED', 'SUPPORTED', 'SUPPORTED'] | acc_gap=0.043±0.118 | Seed=0. Two-pass vs forward gap=-0.072. Retroactive write ra… |
| exp_32_2 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'REFUTED', 'SUPPORTED', 'REFUTED', 'SUPPORTED', 'REFUTED'] | acc_A_continuous=0.140±0.016 | Seed=0. acc_ratio=0.905, write_rate=0.515. acc_A=0.138, acc_… |
| exp_32_3 | ✓ SUPPORTED | ⚠ ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED'] | acc_single=0.062±0.035 | Seed=0. degradation_ratio=2.000. Accs: 1-hop=0.094, 2-hop=0.… |
| exp_32_4 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'REFUTED', 'INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED'] | acc_param_len24=0.029±0.012 | Seed=0. Param retention gap=0.498. Required gap > 0.35. Slot… |
| exp_33_1 | ✗ REFUTED | ✓ | acc_n16=0.009±0.004 | Slot memory γ=-0.004, R²=0.000. Accs: [0.0078, 0.0055, 0.009… |
| exp_33_2 | ~ INCONCLUSIVE | ⚠ ['INCONCLUSIVE', 'INCONCLUSIVE', 'SUPPORTED'] | acc_delta_n16=0.013±0.001 | γ ordering: param=-0.081 < slot=-0.196 < delta=1.338, spread… |
| exp_33_3 | ✗ REFUTED | ✓ | gamma_delta_H128=0.802±0.165 | Slot γ spread=0.465, Delta γ spread=0.086. Max of both=0.465… |
| exp_33_4 | ✓ SUPPORTED | ⚠ ['REFUTED', 'SUPPORTED', 'SUPPORTED'] | acc_delta_1200=0.006±0.002 | Slot recovery=0.000, Delta recovery=0.000. ρ=0.5 ref: slot=0… |
| exp_34_1 | ✗ REFUTED | ✓ | delta_chk1000=0.285±0.028 | Delta transition: False (gain=0.000, window -1--1). Slot tra… |
| exp_34_2 | ~ INCONCLUSIVE | ✓ | enc_gnorm_s100=2.131±0.241 | Gradient ratio (mem/enc): early(s100)=2.155, late(s1000)=1.9… |
| exp_34_3 | ~ INCONCLUSIVE | ✓ | write_rate_s100=0.534±0.003 | Write rate: early(s100)=0.534, late(s1500)=0.543. Full traje… |
| exp_34_4 | ~ INCONCLUSIVE | ✓ | acc_easy_first=0.278±0.031 | Random=0.225, Easy-first=0.269, Hard-first=0.271. Gap(easy-r… |
| exp_34_5 | ✗ REFUTED | ✓ | acc_full_immediate=0.209±0.008 | No significant warmup benefit: gain=0.013 (threshold >0.03).… |
| exp_34_6 | ✓ SUPPORTED | ⚠ ['INCONCLUSIVE', 'REFUTED', 'SUPPORTED'] | acc_Adam=0.209±0.008 | Moderate spread=0.049. Best=SGD(0.242), Worst=AdamW(0.193).… |
| exp_34_7 | ✗ REFUTED | ✓ | acc_lr_1e-02=0.245±0.012 | Wide stable band: 2.48 decades > 2.0. Low LR sensitivity.… |
| exp_34_8 | ✗ REFUTED | ✓ | acc_B128=0.233±0.017 | No batch sensitivity: drop=-0.047<0.01. Memory quality scale… |
| exp_34_9 | ✗ REFUTED | ✓ | acc=0.248±0.005 | Gate is well-spread: bimodal_fraction=0.000 < 0.15. Entropy=… |
| exp_35_1 | ✗ REFUTED | ✓ | acc_noise_0pct=0.205±0.011 | Catastrophic degradation: baseline=0.194, @25%noise=0.016 (r… |
| exp_35_2 | ✓ SUPPORTED | ✓ | acc_in_distribution=0.192±0.017 | No hallucination: OOD acc=0.040 ≈ random baseline=0.031 (gap… |
| exp_35_3 | ✓ SUPPORTED | ⚠ ['SUPPORTED', 'REFUTED', 'INCONCLUSIVE'] | acc_ood_0=0.296±0.003 | Abnormal OOD write rate: baseline=0.000, OOD=0.000, ratio=0.… |
| exp_36_1 | ✗ REFUTED | ✓ | acc_no_consolidation=0.178±0.008 | No consolidation benefit: gain=0.005 (threshold >0.03). Offl… |
| exp_36_2 | ~ INCONCLUSIVE | ✓ | acc_full_representation=0.205±0.011 | Roughly equivalent: full=0.194, residual=0.190, gap=-0.004. … |
| exp_36_3 | ✓ SUPPORTED | ✓ | acc_split=0.236±0.019 | Split memory wins: unified=0.156, split=0.215, advantage=0.0… |
| exp_37_1 | ~ INCONCLUSIVE | ✓ | aug_improvement=0.007±0.051 | Partial improvement: ratio_aug=0.976, ratio_std=0.939. aug_i… |
| exp_37_2 | ~ INCONCLUSIVE | ✓ | baseline_norm=0.225±0.005 | Partial: ratio_norm=0.942, ratio_std=1.006, improvement=-0.0… |
| exp_37_3 | ✓ SUPPORTED | ✓ | best_alpha=0.917±0.058 | EMA (α=0.95) resilient: ratio=1.079 ≥ 0.50, clean_loss=-0.05… |
| exp_38_1 | ✗ REFUTED | ⚠ ['REFUTED', 'REFUTED', 'INCONCLUSIVE'] | acc_fixed=0.354±0.107 | Fixed split wins: fixed=0.335, router=0.231, gap=-0.104<-0.0… |
| exp_38_2 | ✗ REFUTED | ⚠ ['REFUTED', 'SUPPORTED', 'REFUTED'] | acc_25_epi=0.339±0.074 | 50/50 optimal: best=epi050(0.328), 50/50=0.328, gap=0.000 ≤ … |
| exp_38_3 | ✓ SUPPORTED | ⚠ ['REFUTED', 'INCONCLUSIVE', 'SUPPORTED'] | acc_concat=0.354±0.107 | Concat wins: concat=0.335, gated=0.302, gap=-0.033<-0.03.… |
| exp_39_1 | ~ INCONCLUSIVE | ✓ | acc_ts01=0.232±0.017 | Asymmetric: peak_wr=0.532, low_drop=0.209, high_drop=0.000. … |
| exp_39_2 | ~ INCONCLUSIVE | ✓ | acc_n02=0.526±0.012 | Moderate variation: low=0.696, high=0.779, Δ=0.083. Non-mono… |
| exp_39_3 | ✗ REFUTED | ✓ | acc_init005=0.235±0.016 | No convergence: spread=1.022>0.50. Thresholds: [0.049, 0.188… |

---

## Detailed Results by Category

### Category 1 — What To Write
*1 supported / 3 refuted / 4 inconclusive / 0 error*

#### exp_1_1  ✗ REFUTED
**Hypothesis:** Attention weight correlates positively with memory importance and attention-based memory outperforms random memory on retrieval tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 115s

**Metrics (mean ± std across seeds):**

- `attention_correlation` = **-0.5026**  *(stable across seeds)*
- `attention_memory_acc` = **0.0877**  *(stable across seeds)*
- `oracle_memory_acc` = **0.0869**  *(stable across seeds)*
- `random_memory_acc` = **0.0874**  *(stable across seeds)*

**Notes:** Pearson r=-0.5026. Attention acc=0.088 vs random=0.087 vs oracle=0.087.

---
#### exp_1_2  ✗ REFUTED
**Hypothesis:** A memory built from high-surprise (high-perplexity) tokens supports better retrieval than attention-based memory.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 54s

**Metrics (mean ± std across seeds):**

- `attention_acc` = **0.1442**  *(stable across seeds)*
- `baseline_acc` = **0.0306**  *(stable across seeds)*
- `gap_surprise_minus_attention` = **-0.0048**  *(stable across seeds)*
- `surprise_acc` = **0.1394**  *(stable across seeds)*

**Notes:** Surprise acc=0.139, attention acc=0.144, no-memory baseline=0.031. Gap=-0.0048.

---
#### exp_1_3  ~ INCONCLUSIVE
**Hypothesis:** Storing tokens where gradient magnitude is highest produces memories that generalize better than attention-selected memories.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 51s

**Metrics (mean ± std across seeds):**

- `attention_acc` = **0.0900**  *(stable across seeds)*
- `gap_gradient_minus_attention` = **0.0168**  *(stable across seeds)*
- `gradient_acc` = **0.1068**  *(stable across seeds)*
- `random_acc` = **0.1086**  *(stable across seeds)*

**Notes:** Gradient acc=0.107, attention=0.090, random=0.109. Gap=+0.0168.

---
#### exp_1_4  ~ INCONCLUSIVE
**Hypothesis:** Diversity-driven storage (maximally dissimilar entries) outperforms importance-driven storage on recall tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 41s

**Metrics (mean ± std across seeds):**

- `diversity_acc` = **0.1377**  *(stable across seeds)*
- `diversity_mean_pairwise_sim` = **0.3732**  *(stable across seeds)*
- `gap_diversity_minus_importance` = **-0.0067**  *(stable across seeds)*
- `importance_acc` = **0.1444**  *(stable across seeds)*
- `importance_mean_pairwise_sim` = **0.5095**  *(stable across seeds)*

**Notes:** Diversity acc=0.138 vs importance acc=0.144. Gap=-0.0067. Diversity sim=0.3732 vs importance sim=0.5095.

---
#### exp_1_5  ✓ SUPPORTED
**Hypothesis:** A learned write gate outperforms random write, attention-weighted write, and surprise-driven write on associative recall tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 136s

**Metrics (mean ± std across seeds):**

- `acc_attention` = **0.1210**  *(stable across seeds)*
- `acc_learned` = **0.1242**  *(stable across seeds)*
- `acc_random` = **0.1201**  *(stable across seeds)*
- `acc_surprise` = **0.1176**  *(stable across seeds)*
- `loss_attention` = **3.1545**  *(stable across seeds)*
- `loss_learned` = **3.0535**  *(stable across seeds)*
- `loss_random` = **3.1498**  *(stable across seeds)*
- `loss_surprise` = **3.1317**  *(stable across seeds)*
- `ranking` = [['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise'], ['learned', 'attention', 'random', 'surprise']]

**Notes:** Learned gate delta over best baseline: +0.003. Ranking: ['learned', 'attention', 'random', 'surprise'].

---
#### exp_1_6  ~ INCONCLUSIVE
**Hypothesis:** Cosine-similarity deduplication at write time improves retrieval precision without dangerous information loss.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 70s

**Metrics (mean ± std across seeds):**

- `coverage_delta` = **0.0000**  *(stable across seeds)*
- `dedup_coverage` = **1.0000**  *(stable across seeds)*
- `dedup_precision` = **0.1036**  *(stable across seeds)*
- `precision_delta` = **-0.0160**  *(stable across seeds)*
- `standard_coverage` = **1.0000**  *(stable across seeds)*
- `standard_precision` = **0.1197**  *(stable across seeds)*

**Notes:** Dedup: precision=0.104 (std=0.120), coverage=1.000 (std=1.000). Threshold=0.7.

---
#### exp_1_7  ~ INCONCLUSIVE
**Hypothesis:** For a fixed storage budget, infrequent writes with high compression outperform frequent writes with low compression on downstream retrieval.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 50s

**Metrics (mean ± std across seeds):**

- `freq_acc` = **0.0658**  *(stable across seeds)*
- `freq_budget_floats` = **256.0000**  *(stable across seeds)*
- `gap_infreq_minus_freq` = **-0.0076**  *(stable across seeds)*
- `infreq_acc` = **0.0582**  *(stable across seeds)*
- `infreq_budget_floats` = **256.0000**  *(stable across seeds)*

**Notes:** Infreq acc=0.058 vs freq acc=0.066. Gap=-0.0076. Budget: freq=256 floats, infreq=256 floats.

---
#### exp_1_8  ✗ REFUTED
**Hypothesis:** A two-stage write decision (coarse filter then fine ranking) outperforms a single-stage write gate.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 44s

**Metrics (mean ± std across seeds):**

- `gap_two_minus_single` = **-0.0151**  *(stable across seeds)*
- `single_stage_acc` = **0.1466**  *(stable across seeds)*
- `two_stage_acc` = **0.1315**  *(stable across seeds)*

**Notes:** Two-stage acc=0.131 vs single-stage acc=0.147. Gap=-0.0151. Stage1 threshold=0.5.

---

### Category 2 — How To Write (Compression)
*1 supported / 4 refuted / 4 inconclusive / 0 error*

#### exp_2_1  ~ INCONCLUSIVE
**Hypothesis:** There exists a compression ratio threshold beyond which recall fidelity degrades catastrophically rather than gracefully.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 1593s

**Metrics (mean ± std across seeds):**

- `catastrophic_cliff_detected` = [False, False, False, False, False, False]
- `cliff_after_ratio` = **64.0000**  *(stable across seeds)*
- `cliff_at_ratio` = **32.0000**  *(stable across seeds)*
- `cosine_sims.100` = **0.1211**  *(stable across seeds)*
- `cosine_sims.16` = **0.2188**  *(stable across seeds)*
- `cosine_sims.2` = **0.0515**  *(stable across seeds)*
- `cosine_sims.32` = **0.1983**  *(stable across seeds)*
- `cosine_sims.4` = **0.0529**  *(stable across seeds)*
- `cosine_sims.64` = **0.1475**  *(stable across seeds)*
- `cosine_sims.8` = **0.0530**  *(stable across seeds)*
- `max_single_step_drop` = **0.0508**  *(stable across seeds)*
- `mse_values.100` = **0.9937**  *(stable across seeds)*
- `mse_values.16` = **0.9603**  *(stable across seeds)*
- `mse_values.2` = **1.0064**  *(stable across seeds)*
- `mse_values.32` = **0.9689**  *(stable across seeds)*
- `mse_values.4` = **1.0058**  *(stable across seeds)*
- `mse_values.64` = **0.9863**  *(stable across seeds)*
- `mse_values.8` = **1.0058**  *(stable across seeds)*

**Notes:** Largest quality drop: 0.051 between 32x and 64x compression. Cosine sim at 2x=0.052, at 100x=0.121.

---
#### exp_2_2  ✗ REFUTED
**Hypothesis:** Attention-based compression produces more retrievable representations than autoencoder compression, especially on inferential recall tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 18s

**Metrics (mean ± std across seeds):**

- `exact_cosim_attention` = **0.3209**  *(stable across seeds)*
- `exact_cosim_autoenc` = **0.4716**  *(stable across seeds)*
- `fuzzy_acc_attention` = **0.0000**  *(stable across seeds)*
- `fuzzy_acc_autoenc` = **0.2047**  *(stable across seeds)*
- `inferential_acc_attention` = **0.9905**  *(stable across seeds)*
- `inferential_acc_autoenc` = **0.9978**  *(stable across seeds)*
- `inferential_gain_attn_over_autoenc` = **-0.0073**  *(stable across seeds)*

**Notes:** Inferential recall: attention=0.990, autoenc=0.998, gain=-0.007. Attention wins on inferential task by >5pp: False. Autoencoder wins all three tasks: True.

---
#### exp_2_3  ~ INCONCLUSIVE
**Hypothesis:** A controller can learn without supervision which information should be stored exactly (numbers, names) vs approximately (context, themes).

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 3s

**Metrics (mean ± std across seeds):**

- `avg_error_approx` = **0.0004**  *(stable across seeds)*
- `avg_error_exact` = **0.0000**  *(stable across seeds)*
- `error_ratio` = **33.8341**  *(stable across seeds)*
- `high_precision_rate_for_approx` = **0.9852**  *(stable across seeds)*
- `high_precision_rate_for_exact` = **0.9969**  *(stable across seeds)*

**Notes:** avg_error_exact=0.0000, avg_error_approx=0.0004, diff (approx-exact)=+0.0004. HP rate: exact=0.997, approx=0.985. System learned to protect exact tokens: True.

---
#### exp_2_4  ✗ REFUTED
**Hypothesis:** There exists an optimal chunk size for compression beyond which quality degrades independent of compression ratio.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 272s

**Metrics (mean ± std across seeds):**

- `clear_peak_detected` = [False, False, False, False, False, False]
- `cosine_sim.16` = **0.3402**  *(stable across seeds)*
- `cosine_sim.32` = **0.2274**  *(stable across seeds)*
- `cosine_sim.4` = **0.6425**  *(stable across seeds)*
- `cosine_sim.64` = **0.1531**  *(stable across seeds)*
- `cosine_sim.8` = **0.4728**  *(stable across seeds)*
- `is_flat` = [False, False, False, False, False, False]
- `is_monotone_decrease` = [True, True, True, True, True, True]
- `optimal_chunk_size` = **4.0000**  *(stable across seeds)*
- `quality_at_16` = **0.3402**  *(stable across seeds)*
- `quality_at_32` = **0.2274**  *(stable across seeds)*
- `quality_at_4` = **0.6425**  *(stable across seeds)*
- `quality_at_64` = **0.1531**  *(stable across seeds)*
- `quality_at_8` = **0.4728**  *(stable across seeds)*
- `quality_range` = **0.4894**  *(stable across seeds)*

**Notes:** Quality scores [0.642, 0.473, 0.34, 0.227, 0.153] for chunk sizes [4, 8, 16, 32, 64]. Best quality at chunk_size=4 (index 0). Clear peak at middle: False. Monotone decrease: True. Flat: False.

---
#### exp_2_5  ✗ REFUTED
**Hypothesis:** Compressing into a structured key-value representation improves retrieval over compressing into a flat dense vector.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `acc_gain_structured_over_flat` = **-0.3127**  *(stable across seeds)*
- `flat_acc` = **0.3686**  *(stable across seeds)*
- `flat_cosim` = **0.4555**  *(stable across seeds)*
- `structured_acc` = **0.0559**  *(stable across seeds)*
- `structured_cosim` = **0.3111**  *(stable across seeds)*

**Notes:** Structured acc=0.056, flat acc=0.369, gain=-0.313. Structured beats flat by >3pp: False. Flat beats structured: True.

---
#### exp_2_6  ✓ SUPPORTED
**Hypothesis:** A compressor trained on domain A produces meaningfully worse retrieval on domain B, indicating compression overfits to domain.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 10s

**Metrics (mean ± std across seeds):**

- `comp_a_on_a` = **0.6104**  *(stable across seeds)*
- `comp_a_on_b` = **0.2206**  *(stable across seeds)*
- `comp_ab_on_a` = **0.5300**  *(stable across seeds)*
- `comp_ab_on_b` = **0.5337**  *(stable across seeds)*
- `comp_b_on_a` = **0.2342**  *(stable across seeds)*
- `comp_b_on_b` = **0.6136**  *(stable across seeds)*
- `drop_a_cross_domain` = **0.3897**  *(stable across seeds)*
- `drop_b_cross_domain` = **0.3794**  *(stable across seeds)*

**Notes:** Comp_A: in-domain=0.610, cross-domain=0.221, drop=0.390 (threshold=0.1). Comp_B: in-domain=0.614, cross-domain=0.234, drop=0.379. Mixed compressor: A=0.530, B=0.534.

---
#### exp_2_7  ~ INCONCLUSIVE
**Hypothesis:** A hierarchy of increasingly abstract memory levels can be built by iterative compression without catastrophic information loss at each stage.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 9s

**Metrics (mean ± std across seeds):**

- `cosim_stage1` = **0.4873**  *(stable across seeds)*
- `cosim_stage2` = **0.2303**  *(stable across seeds)*
- `cosim_stage3` = **0.0911**  *(stable across seeds)*
- `info_retention_pct_per_stage.stage1` = **0.4873**  *(stable across seeds)*
- `info_retention_pct_per_stage.stage2` = **0.4727**  *(stable across seeds)*
- `info_retention_pct_per_stage.stage3` = **0.3957**  *(stable across seeds)*
- `total_compression_stage1` = **4.0000**  *(stable across seeds)*
- `total_compression_stage2` = **16.0000**  *(stable across seeds)*
- `total_compression_stage3` = **64.0000**  *(stable across seeds)*

**Notes:** Cosine similarities after iterative compression: stage1=0.487 (4x), stage2=0.230 (16x), stage3=0.091 (64x). Stage2 above support threshold 0.3: False. Stage2 below refute threshold 0.1: False.

---
#### exp_2_8  ~ INCONCLUSIVE
**Hypothesis:** A compressor trained on a fixed distribution degrades gracefully (not catastrophically) under mid-context distribution shift.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 7s

**Metrics (mean ± std across seeds):**

- `catastrophic` = [False, False, False, False, False, False]
- `quality_a_in_mixed_seq` = **0.6087**  *(stable across seeds)*
- `quality_domain_a` = **0.6050**  *(stable across seeds)*
- `quality_domain_b_shifted` = **0.2901**  *(stable across seeds)*
- `quality_drop` = **0.3149**  *(stable across seeds)*

**Notes:** Domain A (in-distribution): 0.605. Domain B (shifted): 0.290. Drop: 0.315 vs catastrophic threshold 0.2. Graceful degradation (drop < threshold): False. Catastrophic failure: False.

---
#### exp_2_9  ✗ REFUTED
**Hypothesis:** Minimizing reconstruction loss and maximizing downstream retrieval accuracy are fundamentally different objectives and produce measurably different compressed representations.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 207s

**Metrics (mean ± std across seeds):**

- `objectives_diverge` = [False, False, False, False, False, False]
- `recon_a_cosine` = **0.4135**  *(stable across seeds)*
- `recon_b_cosine` = **0.3972**  *(stable across seeds)*
- `recon_gap` = **0.0163**  *(stable across seeds)*
- `retrieval_acc1_a` = **1.0000**  *(stable across seeds)*
- `retrieval_acc1_b` = **1.0000**  *(stable across seeds)*
- `retrieval_gap` = **0.0000**  *(stable across seeds)*

**Notes:** Reconstruction A=0.414 vs B=0.397 (gap 0.016). Retrieval A=1.000 vs B=1.000 (gap 0.000). Diverge=False.

---

### Category 3 — When To Write
*3 supported / 4 refuted / 0 inconclusive / 0 error*

#### exp_3_1  ✓ SUPPORTED
**Hypothesis:** Event-driven writing (learned gate) produces better memory coverage than writing every N tokens for a fixed storage budget.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 35s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.0047**  *(stable across seeds)*
- `continuous_acc` = **0.0266**  *(stable across seeds)*
- `continuous_coverage` = **0.2500**  *(stable across seeds)*
- `coverage_delta` = **0.0588**  *(stable across seeds)*
- `event_driven_acc` = **0.0312**  *(stable across seeds)*
- `event_driven_coverage` = **0.3088**  *(stable across seeds)*

**Notes:** Event-driven acc delta: +0.005. Coverage delta: +0.059. Stride used for continuous: 4. Top-k=6 used for event-driven.

---
#### exp_3_2  ✗ REFUTED
**Hypothesis:** A learned write gate trained without explicit anti-collapse objectives will learn to never write (or always write) within N training steps on standard tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `A_no_signal.collapsed_always` = [False, False, False, False, False, False]
- `A_no_signal.collapsed_zero` = [False, False, False, False, False, False]
- `A_no_signal.final_write_rate` = **0.1927**  *(stable across seeds)*
- `B_entropy.collapsed_always` = [False, False, False, False, False, False]
- `B_entropy.collapsed_zero` = [False, False, False, False, False, False]
- `B_entropy.final_write_rate` = **0.7448**  *(stable across seeds)*
- `C_reconstruction.collapsed_always` = [False, False, False, False, False, False]
- `C_reconstruction.collapsed_zero` = [False, False, False, False, False, False]
- `C_reconstruction.final_write_rate` = **0.2786**  *(stable across seeds)*
- `D_penalty.collapsed_always` = [True, True, True, True, True, True]
- `D_penalty.collapsed_zero` = [False, False, False, False, False, False]
- `D_penalty.final_write_rate` = **0.9505**  *(stable across seeds)*

**Notes:** Regime A collapsed: False. Other regimes collapsed: True. Write rates: A_no_signal=0.19, B_entropy=0.74, C_reconstruction=0.28, D_penalty=0.95.

---
#### exp_3_3  ✗ REFUTED
**Hypothesis:** Writing later in context (more processed representations) outperforms early writing (raw representations) for inferential tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 83s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.0000**  *(stable across seeds)*
- `early_final_loss` = **0.0410**  *(stable across seeds)*
- `early_write_acc` = **1.0000**  *(stable across seeds)*
- `late_final_loss` = **0.0054**  *(stable across seeds)*
- `late_write_acc` = **1.0000**  *(stable across seeds)*

**Notes:** Late vs early accuracy gap: +0.000. Threshold for SUPPORTED: >0.02. Task: inferential rule application (4 rules, 3 examples).

---
#### exp_3_4  ✗ REFUTED
**Hypothesis:** Semantic-boundary-triggered writing outperforms fixed-interval writing on long-document tasks with clear topical structure.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 38s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **-0.1729**  *(stable across seeds)*
- `boundary_acc` = **0.0359**  *(stable across seeds)*
- `boundary_per_segment_acc` = [[0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281], [0.0391, 0.0406, 0.0281]]
- `fixed_acc` = **0.2089**  *(stable across seeds)*
- `fixed_per_segment_acc` = [[0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891], [0.0234, 0.0141, 0.5891]]
- `segment_balance_score` = **0.9932**  *(stable across seeds)*

**Notes:** Boundary vs fixed accuracy gap: -0.173. Boundary per-segment: [0.039, 0.041, 0.028]. Fixed per-segment: [0.023, 0.014, 0.589]. Boundary balance score: 0.993.

---
#### exp_3_5  ✓ SUPPORTED
**Hypothesis:** Downstream retrieval quality degrades measurably when write latency exceeds a specific token distance threshold.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 117s

**Metrics (mean ± std across seeds):**

- `acc_at_latency_0` = **0.2156**  *(stable across seeds)*
- `acc_at_latency_16` = **0.0312**  *(stable across seeds)*
- `acc_at_latency_2` = **0.2375**  *(stable across seeds)*
- `acc_at_latency_4` = **0.2016**  *(stable across seeds)*
- `acc_at_latency_8` = **0.0406**  *(stable across seeds)*
- `acc_range` = **0.2062**  *(stable across seeds)*
- `latency_threshold` = **8.0000**  *(stable across seeds)*

**Notes:** Baseline (L=0): 0.216. Accuracy range across latencies: 0.206. First threshold where drop >0.05: 8. Flat (no degradation): False.

---
#### exp_3_6  ✓ SUPPORTED
**Hypothesis:** A controller can learn to retroactively write tokens it initially skipped once later context reveals their importance.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 37s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.1328**  *(stable across seeds)*
- `forward_acc` = **0.0391**  *(stable across seeds)*
- `retroactive_write_rate` = **0.0833**  *(stable across seeds)*
- `two_pass_acc` = **0.1719**  *(stable across seeds)*

**Notes:** Two-pass vs forward-only accuracy gap: +0.133. Retroactive write rate (fraction of tokens upgraded): 0.083. Forward pass writes 4 slots; revision adds up to 2 more.

---
#### exp_3_7  ✗ REFUTED
**Hypothesis:** A controller given a fixed write budget learns to allocate it non-uniformly in a way that improves performance over uniform allocation.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 55s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.0266**  *(stable across seeds)*
- `adaptive_acc` = **0.2437**  *(stable across seeds)*
- `allocation_gini` = **0.1639**  *(stable across seeds)*
- `allocation_per_block` = [[1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906], [1.9859, 1.0, 1.0234, 1.9906]]
- `uniform_acc` = **0.2172**  *(stable across seeds)*
- `uniform_alloc_per_block` = [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]

**Notes:** Adaptive vs uniform accuracy delta: +0.027. Allocation Gini coefficient: 0.164 (threshold 0.3). Adaptive per-block: [1.99, 1.0, 1.02, 1.99]. Uniform per-block: [1.0, 1.0, 2.0, 2.0] (fixed: [1, 1, 2, 2]).

---

### Category 4 — What To Read
*4 supported / 1 refuted / 4 inconclusive / 0 error*

#### exp_4_1  ✓ SUPPORTED
**Hypothesis:** A dedicated query formulation module outperforms direct use of the current hidden state as a retrieval query.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 5s

**Metrics (mean ± std across seeds):**

- `direct_acc` = **0.0469**  *(stable across seeds)*
- `direct_loss` = **4.1343**  *(stable across seeds)*
- `gap_learned_minus_direct` = **0.0241**  *(stable across seeds)*
- `learned_acc` = **0.0709**  *(stable across seeds)*
- `learned_loss` = **4.1128**  *(stable across seeds)*

**Notes:** Learned acc=0.0709 vs Direct acc=0.0469, gap=+0.0241 (threshold ±0.02).

---
#### exp_4_2  ✗ REFUTED
**Hypothesis:** Multi-vector retrieval captures more relevant content than single-vector retrieval for queries with multi-faceted information needs.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 45s

**Metrics (mean ± std across seeds):**

- `gap_multi_minus_single` = **-0.0199**  *(stable across seeds)*
- `multi_acc` = **0.9545**  *(stable across seeds)*
- `multi_loss` = **0.5983**  *(stable across seeds)*
- `single_acc` = **0.9744**  *(stable across seeds)*
- `single_loss` = **0.4819**  *(stable across seeds)*

**Notes:** Multi acc=0.9545 vs Single acc=0.9744, gap=-0.0199 (threshold +0.03 for SUPPORTED).

---
#### exp_4_3  ~ INCONCLUSIVE
**Hypothesis:** There exists an optimal retrieval depth (top-k) beyond which additional retrieved entries introduce more noise than signal.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 20s

**Metrics (mean ± std across seeds):**

- `acc_at_k_1` = **0.0161**  *(stable across seeds)*
- `acc_at_k_12` = **0.0175**  *(stable across seeds)*
- `acc_at_k_16` = **0.0158**  *(stable across seeds)*
- `acc_at_k_2` = **0.0149**  *(stable across seeds)*
- `acc_at_k_4` = **0.0150**  *(stable across seeds)*
- `acc_at_k_8` = **0.0171**  *(stable across seeds)*
- `acc_at_max_k` = **0.0158**  *(stable across seeds)*
- `optimal_k` = **12.0000**  *(stable across seeds)*
- `peak_acc` = **0.0175**  *(stable across seeds)*

**Notes:** Accuracy is flat across k values (range=0.0026).

---
#### exp_4_4  ✓ SUPPORTED
**Hypothesis:** Soft retrieval (weighted average) produces more stable training than hard retrieval (discrete selection), though hard retrieval may achieve higher peak task accuracy.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 6s

**Metrics (mean ± std across seeds):**

- `hard_final_acc` = **0.0166**  *(stable across seeds)*
- `hard_is_more_accurate` = [True, True, True, True, True, True]
- `hard_loss_variance` = **0.0001**  *(stable across seeds)*
- `soft_final_acc` = **0.0164**  *(stable across seeds)*
- `soft_is_more_stable` = [True, True, True, True, True, True]
- `soft_loss_variance` = **0.0000**  *(stable across seeds)*

**Notes:** Soft var=0.000033 vs Hard var=0.000076; Soft acc=0.0164 vs Hard acc=0.0166.

---
#### exp_4_5  ~ INCONCLUSIVE
**Hypothesis:** Simultaneous cross-tier retrieval achieves better recall than sequential cascading retrieval.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `avg_tiers_queried_sequential` = **3.0000**  *(stable across seeds)*
- `gap_sim_minus_seq` = **0.0006**  *(stable across seeds)*
- `sequential_acc` = **0.0156**  *(stable across seeds)*
- `simultaneous_acc` = **0.0163**  *(stable across seeds)*

**Notes:** Simultaneous acc=0.0163 vs Sequential acc=0.0156, gap=+0.0006 (threshold +0.02 for SUPPORTED). Sequential avg tiers queried=3.00.

---
#### exp_4_6  ~ INCONCLUSIVE
**Hypothesis:** For exact recall tasks similarity-based retrieval wins; for inferential completion tasks reconstruction-based retrieval wins.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 13s

**Metrics (mean ± std across seeds):**

- `rec_wins_inferential` = [False, False, False, False, False, False]
- `recon_exact_acc` = **0.0145**  *(stable across seeds)*
- `recon_inferential_acc` = **0.0340**  *(stable across seeds)*
- `sim_wins_exact` = [False, False, False, False, False, False]
- `similarity_exact_acc` = **0.0163**  *(stable across seeds)*
- `similarity_inferential_acc` = **0.0289**  *(stable across seeds)*

**Notes:** Exact(Sim=0.0163, Rec=0.0145), Inferential(Sim=0.0289, Rec=0.0340). Specialisation A=False, B=False.

---
#### exp_4_7  ✓ SUPPORTED
**Hypothesis:** A learned read gate can be trained to return null on tasks where most queries have no relevant memory content, without explicit null supervision.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 9s

**Metrics (mean ± std across seeds):**

- `null_f1` = **0.8886**  *(stable across seeds)*
- `null_precision` = **1.0000**  *(stable across seeds)*
- `null_recall` = **0.7996**  *(stable across seeds)*
- `p_relevant` = **0.2000**  *(stable across seeds)*
- `retrieval_rate_on_matches` = **0.0000**  *(stable across seeds)*

**Notes:** Null precision 1.000 vs threshold 0.7.

---
#### exp_4_8  ~ INCONCLUSIVE
**Hypothesis:** Retrieval quality degrades non-linearly as the number of near-duplicate entries in memory increases, with a specific saturation point.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 20s

**Metrics (mean ± std across seeds):**

- `acc_at_N_0` = **0.0172**  *(stable across seeds)*
- `acc_at_N_12` = **0.0148**  *(stable across seeds)*
- `acc_at_N_16` = **0.0152**  *(stable across seeds)*
- `acc_at_N_2` = **0.0167**  *(stable across seeds)*
- `acc_at_N_4` = **0.0160**  *(stable across seeds)*
- `acc_at_N_8` = **0.0144**  *(stable across seeds)*
- `degradation_is_nonlinear` = [True, True, True, True, True, True]
- `saturation_point_N` = **4.0000**  *(stable across seeds)*

**Notes:** Accuracy is flat across N values; no interference detected.

---
#### exp_4_9  ✓ SUPPORTED
**Hypothesis:** A learned retrieval mechanism can be trained to retrieve two separate memory entries and compose them to answer questions neither entry alone can answer.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 10s

**Metrics (mean ± std across seeds):**

- `compositional_gap` = **-0.0316**  *(stable across seeds)*
- `random_baseline` = **0.0625**  *(stable across seeds)*
- `single_hop_accuracy` = **0.9684**  *(stable across seeds)*
- `two_hop_accuracy` = **1.0000**  *(stable across seeds)*

**Notes:** Single-hop=0.968, Two-hop=1.000, Gap=-0.032, Random=0.062.

---

### Category 5 — When To Read
*2 supported / 4 refuted / 1 inconclusive / 0 error*

#### exp_5_1  ✗ REFUTED
**Hypothesis:** A learned read gate trained without explicit anti-collapse objectives will learn a degenerate policy (always read or never read) within N training steps.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 8s

**Metrics (mean ± std across seeds):**

- `A_task_only.collapsed` = [False, False, False, False, False, False]
- `A_task_only.final_read_rate` = **0.1562**  *(stable across seeds)*
- `A_task_only.mode` = ['stable', 'stable', 'stable', 'stable', 'stable', 'stable']
- `B_sparsity.collapsed` = [True, True, True, True, True, True]
- `B_sparsity.final_read_rate` = **0.0938**  *(stable across seeds)*
- `B_sparsity.mode` = ['NEVER', 'NEVER', 'NEVER', 'NEVER', 'NEVER', 'NEVER']
- `C_coverage.collapsed` = [False, False, False, False, False, False]
- `C_coverage.final_read_rate` = **0.1562**  *(stable across seeds)*
- `C_coverage.mode` = ['stable', 'stable', 'stable', 'stable', 'stable', 'stable']
- `D_confidence.collapsed` = [False, False, False, False, False, False]
- `D_confidence.final_read_rate` = **0.6562**  *(stable across seeds)*
- `D_confidence.mode` = ['stable', 'stable', 'stable', 'stable', 'stable', 'stable']

**Notes:** Regime A collapsed: False (stable). Read rates: A_task_only=0.16, B_sparsity=0.09, C_coverage=0.16, D_confidence=0.66.

---
#### exp_5_2  ✓ SUPPORTED
**Hypothesis:** Optimal read frequency is task-dependent and cannot be determined by a single fixed schedule across task types.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 412s

**Metrics (mean ± std across seeds):**

- `factual_qa_freq1` = **0.9956**  *(stable across seeds)*
- `factual_qa_freq2` = **0.9969**  *(stable across seeds)*
- `factual_qa_freq4` = **0.9975**  *(stable across seeds)*
- `factual_qa_freq8` = **0.9944**  *(stable across seeds)*
- `frequencies_differ` = [True, True, True, True, True, True]
- `optimal_freq_task1` = **4.0000**  *(stable across seeds)*
- `optimal_freq_task2` = **2.0000**  *(stable across seeds)*
- `optimal_freq_task3` = **4.0000**  *(stable across seeds)*
- `pattern_matching_freq1` = **0.9513**  *(stable across seeds)*
- `pattern_matching_freq2` = **0.9413**  *(stable across seeds)*
- `pattern_matching_freq4` = **0.9806**  *(stable across seeds)*
- `pattern_matching_freq8` = **0.9456**  *(stable across seeds)*
- `seq_completion_freq1` = **0.9900**  *(stable across seeds)*
- `seq_completion_freq2` = **0.9981**  *(stable across seeds)*
- `seq_completion_freq4` = **0.9931**  *(stable across seeds)*
- `seq_completion_freq8` = **0.9950**  *(stable across seeds)*

**Notes:** Optimal frequencies — task1(factual_qa):4, task2(seq_completion):2, task3(pattern_matching):4. Frequencies differ: True. Meaningful accuracy gap between frequencies: True.

---
#### exp_5_3  ~ INCONCLUSIVE
**Hypothesis:** Anticipatory retrieval (predicting need before it arises) improves latency without measurably hurting task quality.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 93s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.0062**  *(stable across seeds)*
- `predictive_acc` = **0.9838**  *(stable across seeds)*
- `predictive_read_rate` = **0.4050**  *(stable across seeds)*
- `reactive_acc` = **0.9900**  *(stable across seeds)*
- `reactive_read_rate` = **0.1250**  *(stable across seeds)*

**Notes:** Reactive: acc=0.990, read_rate=0.125. Predictive: acc=0.984, read_rate=0.405. Acc gap: 0.0062 (positive = reactive better). Read rate reduction: -0.2800.

---
#### exp_5_4  ✗ REFUTED
**Hypothesis:** A controller can learn to prefer recomputation for cheap information and retrieval for expensive information.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `rate_diff` = **0.0000**  *(stable across seeds)*
- `retrieval_rate_typeA` = **0.0003**  *(stable across seeds)*
- `retrieval_rate_typeB` = **0.0003**  *(stable across seeds)*
- `typeA_acc` = **1.0000**  *(stable across seeds)*
- `typeB_acc` = **0.0163**  *(stable across seeds)*

**Notes:** Type A (cheap/recompute): retrieval_rate=0.000, acc=1.000. Type B (expensive/retrieve): retrieval_rate=0.000, acc=0.016. Rate difference (B-A)=-0.000. Threshold: A<0.3 AND B>0.6 for SUPPORTED.

---
#### exp_5_5  ✗ REFUTED
**Hypothesis:** Confidence-gated cascading retrieval matches full-depth retrieval quality at significantly lower average compute cost.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 42s

**Metrics (mean ± std across seeds):**

- `cascading_acc` = **0.3962**  *(stable across seeds)*
- `cascading_tiers_avg` = **1.1690**  *(stable across seeds)*
- `compute_savings_pct` = **61.0200**  *(stable across seeds)*
- `full_acc` = **0.9613**  *(stable across seeds)*
- `full_tiers_avg` = **3.0000**  *(stable across seeds)*

**Notes:** Full-depth: acc=0.961, tiers=3.00. Cascading: acc=0.396, tiers=1.17. Acc drop=0.5650. Compute savings=61.0%. Threshold: acc_drop<=0.02 AND casc_tiers<2.0 for SUPPORTED.

---
#### exp_5_6  ✓ SUPPORTED
**Hypothesis:** Suppressing memory reads when next-token prediction confidence exceeds a threshold costs less than 1% task quality.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 54s

**Metrics (mean ± std across seeds):**

- `acc_at_T_50` = **0.9719**  *(stable across seeds)*
- `acc_at_T_60` = **0.9862**  *(stable across seeds)*
- `acc_at_T_70` = **0.9894**  *(stable across seeds)*
- `acc_at_T_80` = **0.9919**  *(stable across seeds)*
- `acc_at_T_90` = **0.9938**  *(stable across seeds)*
- `baseline_acc` = **0.9969**  *(stable across seeds)*
- `optimal_T` = **0.8000**  *(stable across seeds)*
- `quality_cost_at_optimal_T` = **0.0050**  *(stable across seeds)*
- `suppression_rate_at_T_50` = **0.5633**  *(stable across seeds)*
- `suppression_rate_at_T_60` = **0.4933**  *(stable across seeds)*
- `suppression_rate_at_T_70` = **0.3950**  *(stable across seeds)*
- `suppression_rate_at_T_80` = **0.3092**  *(stable across seeds)*
- `suppression_rate_at_T_90` = **0.0983**  *(stable across seeds)*

**Notes:** Baseline acc (no suppression): 0.997. Optimal T=0.8, quality_cost=0.0050, suppression_rate=0.309. SUPPORTED criterion: quality_cost<0.01 AND sup_rate>0.3 met at any T: True.

---
#### exp_5_7  ✗ REFUTED
**Hypothesis:** When local attention and external memory produce conflicting predictions, a learned arbitration policy outperforms both fixed-priority policies.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 44s

**Metrics (mean ± std across seeds):**

- `arbitrated_acc` = **0.4825**  *(stable across seeds)*
- `arbitration_advantage` = **-0.0281**  *(stable across seeds)*
- `attn_only_acc` = **0.5106**  *(stable across seeds)*
- `best_fixed_policy_acc` = **0.5106**  *(stable across seeds)*
- `mem_only_acc` = **0.5006**  *(stable across seeds)*

**Notes:** attn_only_acc=0.511, mem_only_acc=0.501, arbitrated_acc=0.482. Arbitration advantage over best fixed: -0.0281. SUPPORTED requires advantage > 0.05 over BOTH fixed policies.

---

### Category 6 — How To Forget
*2 supported / 1 refuted / 5 inconclusive / 0 error*

#### exp_6_1  ✓ SUPPORTED
**Hypothesis:** A learned importance-scored eviction policy significantly outperforms LRU on tasks requiring retention of low-frequency but high-importance information.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 139s

**Metrics (mean ± std across seeds):**

- `eviction_policy_ranking` = ['lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)', 'lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020)']
- `learned_acc` = **0.0591**  *(stable across seeds)*
- `learned_vs_lru_gap` = **0.0395**  *(stable across seeds)*
- `lfu_acc` = **0.0594**  *(stable across seeds)*
- `lru_acc` = **0.0195**  *(stable across seeds)*
- `random_acc` = **0.0220**  *(stable across seeds)*

**Notes:** Learned vs LRU gap: 0.040. Threshold for SUPPORTED: >0.03. Ranking: lfu(0.059) > learned(0.059) > random(0.022) > lru(0.020).

---
#### exp_6_2  ~ INCONCLUSIVE
**Hypothesis:** Graceful degradation via iterative compression outperforms hard eviction for long-context tasks where storage budget is the binding constraint.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 131s

**Metrics (mean ± std across seeds):**

- `avg_compression_level_at_query_time` = **1.9333**  *(stable across seeds)*
- `compression_acc` = **0.0289**  *(stable across seeds)*
- `gap_compression_minus_lru` = **-0.0044**  *(stable across seeds)*
- `lru_acc` = **0.0333**  *(stable across seeds)*

**Notes:** Compression vs LRU gap: -0.004. Average compression level (0=full, 1=half, 2=quarter): 1.933.

---
#### exp_6_3  ✓ SUPPORTED
**Hypothesis:** A controller can learn to evict domain-mismatched memories when input distribution shifts, without explicit domain labels.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 351s

**Metrics (mean ± std across seeds):**

- `gap_selective_minus_lru` = **0.0764**  *(stable across seeds)*
- `lru_phase2_acc` = **0.0364**  *(stable across seeds)*
- `selective_eviction_rate_for_phase1_entries` = **0.7012**  *(stable across seeds)*
- `selective_phase2_acc` = **0.1128**  *(stable across seeds)*

**Notes:** Selective vs LRU gap on phase-2 queries: 0.076. Eviction rate for phase-1 entries: 0.701.

---
#### exp_6_4  ~ INCONCLUSIVE
**Hypothesis:** A controller can learn which memories deserve protection (never evict) without explicit supervision, and performance degrades predictably outside an optimal protected-set size.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 498s

**Metrics (mean ± std across seeds):**

- `acc_at_K_0` = **0.0183**  *(stable across seeds)*
- `acc_at_K_1` = **0.0208**  *(stable across seeds)*
- `acc_at_K_2` = **0.0338**  *(stable across seeds)*
- `acc_at_K_3` = **0.0427**  *(stable across seeds)*
- `acc_at_K_4` = **0.0530**  *(stable across seeds)*
- `acc_at_K_5` = **0.0630**  *(stable across seeds)*
- `optimal_K` = **5.0000**  *(stable across seeds)*
- `optimal_K_acc` = **0.0630**  *(stable across seeds)*
- `protection_recall_at_K_0` = **0.0000**  *(stable across seeds)*
- `protection_recall_at_K_1` = **0.1009**  *(stable across seeds)*
- `protection_recall_at_K_2` = **0.0931**  *(stable across seeds)*
- `protection_recall_at_K_3` = **0.0413**  *(stable across seeds)*
- `protection_recall_at_K_4` = **0.1758**  *(stable across seeds)*
- `protection_recall_at_K_5` = **0.3758**  *(stable across seeds)*
- `protection_recall_at_optimal_K` = **0.3758**  *(stable across seeds)*

**Notes:** Optimal K=5 with acc=0.063. K=0 acc=0.018, K=5 acc=0.063. Interior peak detected: False.

---
#### exp_6_5  ~ INCONCLUSIVE
**Hypothesis:** A biologically-inspired memory decay function (Ebbinghaus-style) improves long-horizon task performance compared to instant eviction.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 459s

**Metrics (mean ± std across seeds):**

- `ebbinghaus_acc` = **0.0241**  *(stable across seeds)*
- `gap_ebbinghaus_minus_lru` = **0.0036**  *(stable across seeds)*
- `learned_stability_S` = **11.3669**  *(stable across seeds)*
- `lru_acc` = **0.0205**  *(stable across seeds)*
- `mean_retention_at_query_time` = **0.7500**  *(stable across seeds)*

**Notes:** Ebbinghaus vs LRU gap: 0.004. Learned S=11.367 steps. Mean retention at query: 0.750.

---
#### exp_6_6  ~ INCONCLUSIVE
**Hypothesis:** The memory controller suffers measurable catastrophic forgetting of its learned policies when fine-tuned on a new domain, absent explicit anti-forgetting mechanisms.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 19s

**Metrics (mean ± std across seeds):**

- `acc_a_after_ewc` = **0.0228**  *(stable across seeds)*
- `acc_a_after_std` = **0.0191**  *(stable across seeds)*
- `acc_a_before` = **0.2267**  *(stable across seeds)*
- `acc_b_ewc` = **0.1250**  *(stable across seeds)*
- `acc_b_std` = **0.1061**  *(stable across seeds)*
- `forgetting_ewc` = **0.2039**  *(stable across seeds)*
- `forgetting_reduction_pct` = **1.8059**  *(stable across seeds)*
- `forgetting_std` = **0.2077**  *(stable across seeds)*

**Notes:** Standard forgetting: 0.208. EWC forgetting: 0.204. Significant: True.

---
#### exp_6_7  ✗ REFUTED
**Hypothesis:** Joint optimization of write and evict decisions outperforms treating them as independent operations when storage pressure is constant.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 700s

**Metrics (mean ± std across seeds):**

- `gap_joint_minus_independent` = **-0.0348**  *(stable across seeds)*
- `independent_acc` = **0.0642**  *(stable across seeds)*
- `joint_acc` = **0.0294**  *(stable across seeds)*
- `write_evict_correlation_independent` = **-0.1911**  *(stable across seeds)*
- `write_evict_correlation_joint` = **0.9904**  *(stable across seeds)*

**Notes:** Joint vs independent gap: -0.035. Write-evict correlation — independent: -0.191, joint: 0.990.

---
#### exp_6_8  ~ INCONCLUSIVE
**Hypothesis:** Periodic offline consolidation (merging memory entries into higher-level representations) improves long-horizon recall without active context.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 279s

**Metrics (mean ± std across seeds):**

- `consolidation_acc` = **0.0213**  *(stable across seeds)*
- `consolidation_compression_ratio` = **2.0000**  *(stable across seeds)*
- `gap_consolidation_minus_none` = **0.0033**  *(stable across seeds)*
- `mean_entries_at_query_time_consolidation` = **8.0000**  *(stable across seeds)*
- `mean_entries_at_query_time_no_consolidation` = **8.0000**  *(stable across seeds)*
- `no_consolidation_acc` = **0.0180**  *(stable across seeds)*

**Notes:** Consolidation vs no-consolidation gap: 0.003. Consolidation compresses 8→4 entries (2.00x ratio). Mean entries at query: no-cons=8.0, cons=8.0.

---

### Category 7 — Cross-Cutting
*3 supported / 3 refuted / 3 inconclusive / 0 error*

#### exp_7_1  ✓ SUPPORTED
**Hypothesis:** Gumbel-softmax relaxation produces more stable training than straight-through estimators, and both outperform REINFORCE for discrete memory selection.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 30s

**Metrics (mean ± std across seeds):**

- `Gumbel.accuracy` = **0.1284**  *(stable across seeds)*
- `Gumbel.loss_variance` = **0.0071**  *(stable across seeds)*
- `Gumbel.mean_grad_norm` = **1.0596**  *(stable across seeds)*
- `REINFORCE.accuracy` = **0.1025**  *(stable across seeds)*
- `REINFORCE.loss_variance` = **2.6959**  *(stable across seeds)*
- `REINFORCE.mean_grad_norm` = **0.0000**  *(stable across seeds)*
- `STE.accuracy` = **0.1255**  *(stable across seeds)*
- `STE.loss_variance` = **0.0094**  *(stable across seeds)*
- `STE.mean_grad_norm` = **1.4468**  *(stable across seeds)*
- `ranking_by_accuracy` = [['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE']]
- `ranking_by_stability` = [['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE'], ['Gumbel', 'STE', 'REINFORCE']]

**Notes:** Gumbel most stable: True. Both beat REINFORCE: True. Acc: STE=0.126, Gumbel=0.128, REINFORCE=0.102

---
#### exp_7_2  ✓ SUPPORTED
**Hypothesis:** There exists a maximum controller complexity (measured in parameter count) beyond which the controller's overhead exceeds its efficiency contribution.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 51s

**Metrics (mean ± std across seeds):**

- `acc_per_complexity.Large` = **0.1083**  *(stable across seeds)*
- `acc_per_complexity.Medium` = **0.1227**  *(stable across seeds)*
- `acc_per_complexity.Small` = **0.0939**  *(stable across seeds)*
- `acc_per_complexity.Tiny` = **0.1220**  *(stable across seeds)*
- `acc_per_complexity.XL` = **0.1044**  *(stable across seeds)*
- `declines_at_large_xl` = [True, True, True, True, True, True]
- `efficiency_ratio_per_complexity.Large` = **-0.0018**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.Medium` = **0.0001**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.Small` = **-0.0067**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.Tiny` = **0.0000**  *(stable across seeds)*
- `efficiency_ratio_per_complexity.XL` = **-0.0020**  *(stable across seeds)*
- `flops_per_complexity.Large` = **164096.0000**  *(stable across seeds)*
- `flops_per_complexity.Medium` = **16640.0000**  *(stable across seeds)*
- `flops_per_complexity.Small` = **4160.0000**  *(stable across seeds)*
- `flops_per_complexity.Tiny` = **64.0000**  *(stable across seeds)*
- `flops_per_complexity.XL` = **426240.0000**  *(stable across seeds)*
- `largest_is_best` = [False, False, False, False, False, False]
- `optimal_complexity_level` = ['Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium']
- `params_per_complexity.Large` = **164865.0000**  *(stable across seeds)*
- `params_per_complexity.Medium` = **16897.0000**  *(stable across seeds)*
- `params_per_complexity.Small` = **4225.0000**  *(stable across seeds)*
- `params_per_complexity.Tiny` = **65.0000**  *(stable across seeds)*
- `params_per_complexity.XL` = **427521.0000**  *(stable across seeds)*
- `peaks_early` = [True, True, True, True, True, True]

**Notes:** Peak efficiency at: Medium. Efficiency ratios: {'Tiny': 0.0, 'Small': -0.006737515755360065, 'Medium': 0.00011239988114518264, 'Large': -0.0017541632748579459, 'XL': -0.0020083612849882103}. Acc: {'Tiny': 0.12203125, 'Small': 0.09390625, 'Medium': 0.12265625, 'Large': 0.10828125, 'XL': 0.104375}.

---
#### exp_7_3  ~ INCONCLUSIVE
**Hypothesis:** A controller trained on factual QA generalizes its memory management policies to reasoning tasks but not to generation tasks.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 9s

**Metrics (mean ± std across seeds):**

- `factual_acc` = **0.0939**  *(stable across seeds)*
- `factual_to_generation_gap` = **0.0923**  *(stable across seeds)*
- `factual_to_reasoning_gap` = **0.0395**  *(stable across seeds)*
- `generation_acc` = **0.0016**  *(stable across seeds)*
- `generation_fails` = [False, False, False, False, False, False]
- `reasoning_acc` = **0.0544**  *(stable across seeds)*
- `reasoning_transfers` = [True, True, True, True, True, True]

**Notes:** Reasoning gap=0.040 (threshold <0.15: True). Generation gap=0.092 (threshold >0.20: False).

---
#### exp_7_4  ~ INCONCLUSIVE
**Hypothesis:** Meaningful memory management behavior requires at minimum two layers of non-linearity in the controller network.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 32s

**Metrics (mean ± std across seeds):**

- `acc_per_depth.0` = **0.1220**  *(stable across seeds)*
- `acc_per_depth.1` = **0.1158**  *(stable across seeds)*
- `acc_per_depth.2` = **0.1212**  *(stable across seeds)*
- `acc_per_depth.3` = **0.1119**  *(stable across seeds)*
- `meaningful_threshold_per_depth.0` = [False, False, False, False, False, False]
- `meaningful_threshold_per_depth.1` = [False, False, False, False, False, False]
- `meaningful_threshold_per_depth.2` = [False, False, False, False, False, False]
- `meaningful_threshold_per_depth.3` = [False, False, False, False, False, False]
- `write_rate_per_depth.0` = **0.5188**  *(stable across seeds)*
- `write_rate_per_depth.1` = **0.3554**  *(stable across seeds)*
- `write_rate_per_depth.2` = **0.4763**  *(stable across seeds)*
- `write_rate_per_depth.3` = **0.5391**  *(stable across seeds)*
- `write_std_per_depth.0` = **0.0014**  *(stable across seeds)*
- `write_std_per_depth.1` = **0.0005**  *(stable across seeds)*
- `write_std_per_depth.2` = **0.0002**  *(stable across seeds)*
- `write_std_per_depth.3` = **0.0000**  *(stable across seeds)*

**Notes:** Min depth for meaningful behavior: None. depth=0 meaningful: False. depth=1 meaningful: False. depth=2 meaningful: False.

---
#### exp_7_5  ✗ REFUTED
**Hypothesis:** A controller's learned policy trained at small scale does not transfer directly to a larger model without additional fine-tuning.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 34s

**Metrics (mean ± std across seeds):**

- `finetuned_transfer_acc` = **0.1013**  *(stable across seeds)*
- `fresh_large_acc` = **0.1211**  *(stable across seeds)*
- `small_acc` = **0.0916**  *(stable across seeds)*
- `transfer_gap` = **0.0281**  *(stable across seeds)*
- `zero_shot_transfer_acc` = **0.0930**  *(stable across seeds)*

**Notes:** Transfer gap (fresh - zero_shot): 0.028. Supported threshold: gap > 0.10. Refuted threshold: gap <= 0.05.

---
#### exp_7_6  ✗ REFUTED
**Hypothesis:** The memory controller is measurably vulnerable to inputs designed to maximize write activity, and this vulnerability does not self-correct during training.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `adversarial_ratio` = **1.0146**  *(stable across seeds)*
- `adversarial_write_rate` = **0.5819**  *(stable across seeds)*
- `normal_write_rate` = **0.5735**  *(stable across seeds)*
- `self_corrects` = [False, False, False, False, False, False]
- `write_rate_at_1000` = **0.5887**  *(stable across seeds)*
- `write_rate_at_1500` = **0.5819**  *(stable across seeds)*
- `write_rate_at_500` = **0.5882**  *(stable across seeds)*

**Notes:** Adversarial ratio: 1.015 (threshold >1.5: False). Self-corrects: False. Checkpoint rates: [0.5881532311439515, 0.5887086486816406, 0.5818839371204376].

---
#### exp_7_7  ✗ REFUTED
**Hypothesis:** Write quality (not read quality, compression ratio, or eviction policy) is the first performance bottleneck encountered during controller training.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `bottleneck_at_20pct_training` = ['read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy']
- `bottleneck_at_50pct_training` = ['read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy', 'read_accuracy']
- `bottleneck_order` = [['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'], ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity']]
- `early_averages.compression_fidelity` = **0.5196**  *(stable across seeds)*
- `early_averages.eviction_correctness` = **0.2120**  *(stable across seeds)*
- `early_averages.read_accuracy` = **0.0309**  *(stable across seeds)*
- `early_averages.write_quality` = **0.5196**  *(stable across seeds)*
- `final_compression_fidelity` = **0.8068**  *(stable across seeds)*
- `final_eviction_correctness` = **0.3703**  *(stable across seeds)*
- `final_read_accuracy` = **0.0594**  *(stable across seeds)*
- `final_write_quality` = **0.8068**  *(stable across seeds)*

**Notes:** Bottleneck at 20%: read_accuracy. Bottleneck at 50%: read_accuracy. Order (lowest first): ['read_accuracy', 'eviction_correctness', 'write_quality', 'compression_fidelity'].

---
#### exp_7_8  ~ INCONCLUSIVE
**Hypothesis:** Curriculum training (one controller component at a time) produces more stable controller behavior than joint training from the start.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 18s

**Metrics (mean ± std across seeds):**

- `curriculum_acc` = **0.0750**  *(stable across seeds)*
- `curriculum_gate_collapse` = [False, False, False, False, False, False]
- `curriculum_loss_variance` = **0.0014**  *(stable across seeds)*
- `curriculum_read_collapsed` = [False, False, False, False, False, False]
- `curriculum_write_collapsed` = [False, False, False, False, False, False]
- `joint_acc` = **0.1095**  *(stable across seeds)*
- `joint_gate_collapse` = [False, False, False, False, False, False]
- `joint_loss_variance` = **0.0058**  *(stable across seeds)*
- `joint_read_collapsed` = [False, False, False, False, False, False]
- `joint_write_collapsed` = [False, False, False, False, False, False]

**Notes:** Curriculum acc >= joint acc: False. Curriculum loss_var < joint loss_var: True. Joint acc=0.110, Curriculum acc=0.075. Joint var=0.00581, Curriculum var=0.00137.

---
#### exp_7_9  ✓ SUPPORTED
**Hypothesis:** The controller's write and read decisions are interpretable (non-random, correlating with human-meaningful features) in their simplest form before any task-specific training.

**Runs:** 6 (seeds: [123, 123, 42, 42, 777, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `heatmaps_saved_to` = ['/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability', '/Users/wscholl/drex/research/results/interpretability']
- `interpretability_score_trained` = **0.0595**  *(stable across seeds)*
- `interpretability_score_untrained` = **0.0549**  *(stable across seeds)*
- `trained.corr_numeric_vs_gate` = **0.0757**  *(stable across seeds)*
- `trained.corr_position_vs_gate` = **-0.0039**  *(stable across seeds)*
- `trained.corr_punct_vs_gate` = **-0.0474**  *(stable across seeds)*
- `trained.corr_rare_vs_gate` = **0.0553**  *(stable across seeds)*
- `trained.gate_mean` = **0.5085**  *(stable across seeds)*
- `trained.gate_nonrandom` = [True, True, True, True, True, True]
- `trained.gate_std` = **0.1505**  *(stable across seeds)*
- `untrained.corr_numeric_vs_gate` = **0.0622**  *(stable across seeds)*
- `untrained.corr_position_vs_gate` = **-0.0077**  *(stable across seeds)*
- `untrained.corr_punct_vs_gate` = **-0.0429**  *(stable across seeds)*
- `untrained.corr_rare_vs_gate` = **0.0596**  *(stable across seeds)*
- `untrained.gate_mean` = **0.5099**  *(stable across seeds)*
- `untrained.gate_nonrandom` = [True, True, True, True, True, True]
- `untrained.gate_std` = **0.1457**  *(stable across seeds)*

**Notes:** Interp score: trained=0.059 untrained=0.055. Gate is non-random. Heatmaps saved to results/interpretability/.

---

### Category 8 — Mechanistic Investigations (Phase 2)
*0 supported / 2 refuted / 2 inconclusive / 0 error*

#### exp_8_1  ✗ REFUTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'REFUTED', 'REFUTED']
**Hypothesis:** The attention-importance anti-correlation (r=-0.503 from exp_1_1) is caused by softmax normalization forcing zero-sum redistribution, not a semantic mismatch — removing normalization will produce positive or near-zero correlation.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 7s

**Metrics (mean ± std across seeds):**

- `pearson_r_entropic` = **-0.0772** ± 0.4460  *(runs: 0.220, 0.139, -0.590)*
- `pearson_r_raw` = **-0.2887** ± 0.3413  *(runs: -0.077, -0.106, -0.682)*
- `pearson_r_softmax` = **-0.0333** ± 0.4882  *(runs: 0.229, 0.268, -0.597)*

**Notes:** Softmax r=0.2286, raw dot-product r=-0.0775, entropy-normalized r=0.2198. Hypothesis: raw should be > 0.05 while softmax < -0.10.

---
#### exp_8_2  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE']
**Hypothesis:** The natural ~16-20% gate activity equilibrium is not fixed by architecture but scales with task memory demand — harder tasks requiring more KV pairs will drive equilibrium write rates upward.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 69s

**Metrics (mean ± std across seeds):**

- `accuracies` = [[0.75, 0.0938, 0.1191, 0.0469], [0.6582, 0.3398, 0.2227, 0.0801], [0.377, 0.4043, 0.2656, 0.0957]]
- `kv_levels` = [[1, 2, 4, 6], [1, 2, 4, 6], [1, 2, 4, 6]]
- `pearson_r_difficulty_vs_rate` = **0.3976** ± 0.4053  *(runs: 0.302, 0.842, 0.049)*
- `write_rate_variance` = **0.0225** ± 0.0086  *(runs: 0.030, 0.013, 0.025)*
- `write_rates` = [[0.2829, 0.6731, 0.5961, 0.4431], [0.3564, 0.4239, 0.3966, 0.6134], [0.4159, 0.2534, 0.6117, 0.3162]]

**Notes:** KV levels [1, 2, 4, 6]: write rates [0.2829, 0.6731, 0.5961, 0.4431]. Pearson r=0.3017, variance=0.029855.

---
#### exp_8_3  ✗ REFUTED
**Hypothesis:** The write-evict correlation collapse (r=0.990 in exp_6_7) is gradient aliasing — both gates receive identical gradient from shared loss. Oracle pre-training of each gate on independent labels breaks this, yielding write-evict correlation < 0.5.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 23s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2125** ± 0.0515  *(runs: 0.272, 0.184, 0.181)*
- `acc_B` = **0.2239** ± 0.0144  *(runs: 0.241, 0.216, 0.216)*
- `acc_C` = **0.1198** ± 0.0407  *(runs: 0.078, 0.122, 0.159)*
- `write_evict_corr_A` = **0.2876** ± 0.2877  *(runs: 0.176, 0.073, 0.615)*
- `write_evict_corr_B` = **0.8816** ± 0.0618  *(runs: 0.814, 0.896, 0.935)*
- `write_evict_corr_C` = **-0.0930** ± 0.3026  *(runs: 0.250, -0.207, -0.322)*

**Notes:** Condition A (joint): corr=0.1756, acc=0.272. Condition B (oracle pretrain): corr=0.8137, acc=0.241. Condition C (grad isolation): corr=0.2500, acc=0.078.

---
#### exp_8_4  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** The learned write gate's small advantage over random selection (exp_1_5 +0.003) comes from exploiting token position as proxy for importance, not from detecting semantic content.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 49s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1498** ± 0.1103  *(runs: 0.216, 0.211, 0.022)*
- `acc_B` = **0.1031** ± 0.0581  *(runs: 0.036, 0.141, 0.133)*
- `acc_C` = **0.0577** ± 0.0426  *(runs: 0.033, 0.107, 0.033)*
- `content_corr_A` = **0.1418** ± 0.5253  *(runs: 0.384, 0.502, -0.461)*
- `content_corr_B` = **-0.2957** ± 0.0891  *(runs: -0.391, -0.214, -0.282)*
- `content_corr_C` = **0.0646** ± 0.1687  *(runs: -0.010, -0.054, 0.258)*
- `pos_corr_A` = **-0.0324** ± 0.3364  *(runs: -0.192, -0.259, 0.354)*
- `pos_corr_B` = **0.1271** ± 0.0788  *(runs: 0.195, 0.146, 0.041)*
- `pos_corr_C` = **0.0039** ± 0.1693  *(runs: -0.024, 0.185, -0.149)*

**Notes:** A (full): acc=0.216, pos_r=-0.1920. B (pos-blind): acc=0.036. C (pos-only): acc=0.033. acc_A - acc_B = 0.1800.

---

### Category 9 — Inconclusive Redesigns (Phase 2)
*3 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_9_1  ✗ REFUTED
**Hypothesis:** At 64x compression with 100-way gallery discrimination, retrieval-objective compressor achieves at least 15% higher Acc@1 than reconstruction-objective compressor, because 64x bottleneck forces genuine tradeoffs between fidelity and discriminability.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 62s

**Metrics (mean ± std across seeds):**

- `acc_at1_autoencoder` = **1.0000**  *(stable across seeds)*
- `acc_at1_contrastive` = **1.0000**  *(stable across seeds)*
- `recon_cosim_ae` = **0.1877** ± 0.0017  *(runs: 0.187, 0.187, 0.190)*
- `recon_cosim_cl` = **-0.0001** ± 0.0002  *(runs: -0.000, 0.000, -0.000)*
- `retrieval_gap` = **0.0000**  *(stable across seeds)*

**Notes:** AE Acc@1=1.0000, CL Acc@1=1.0000, gap=0.0000. Recon cosim: AE=0.1867, CL(dummy dec)=-0.0000.

---
#### exp_9_2  ✓ SUPPORTED
**Hypothesis:** With a 50/50 null-to-retrieval query distribution (fixing exp_4_7's degenerate 80% null), a learned read gate achieves null precision > 0.65 and retrieval recall > 0.65, outperforming always-null and always-retrieve baselines on harmonic F1.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 81s

**Metrics (mean ± std across seeds):**

- `is_degenerate_A` = [False, False, False]
- `is_degenerate_B` = [True, True, True]
- `is_degenerate_C` = [True, True, True]
- `is_degenerate_D` = [False, False, False]
- `null_f1_A` = **0.8120** ± 0.0113  *(runs: 0.800, 0.823, 0.813)*
- `null_f1_B` = **0.0000**  *(stable across seeds)*
- `null_f1_C` = **0.6640** ± 0.0046  *(runs: 0.664, 0.669, 0.659)*
- `null_f1_D` = **0.8579** ± 0.0037  *(runs: 0.857, 0.862, 0.855)*
- `null_precision_A` = **0.8106** ± 0.0189  *(runs: 0.830, 0.810, 0.792)*
- `null_precision_B` = **0.0000**  *(stable across seeds)*
- `null_precision_C` = **0.4970** ± 0.0052  *(runs: 0.497, 0.502, 0.492)*
- `null_precision_D` = **0.9386** ± 0.0054  *(runs: 0.933, 0.944, 0.939)*
- `null_recall_A` = **0.8146** ± 0.0361  *(runs: 0.773, 0.837, 0.834)*
- `null_recall_B` = **0.0000**  *(stable across seeds)*
- `null_recall_C` = **1.0000**  *(stable across seeds)*
- `null_recall_D` = **0.7900** ± 0.0043  *(runs: 0.792, 0.793, 0.785)*
- `retrieval_recall_A` = **0.8054** ± 0.0200  *(runs: 0.825, 0.807, 0.785)*
- `retrieval_recall_B` = **0.9437** ± 0.0016  *(runs: 0.945, 0.944, 0.942)*
- `retrieval_recall_C` = **0.0000**  *(stable across seeds)*
- `retrieval_recall_D` = **0.7824** ± 0.0121  *(runs: 0.771, 0.795, 0.781)*

**Notes:** Cond A (learned, p=0.5): null_f1=0.800, retr_recall=0.825, degenerate=False. Cond B (always-retrieve): null_f1=0.000. Cond D (p=0.2 control): degenerate=False.

---
#### exp_9_3  ~ INCONCLUSIVE
**Hypothesis:** When the memory controller achieves >70% domain A accuracy before domain B training, EWC with lambda=5.0 reduces catastrophic forgetting to <50% of standard fine-tuning's forgetting.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `acc_a_after_ewc` = **0.0871** ± 0.0664  *(runs: 0.014, 0.104, 0.143)*
- `acc_a_after_std` = **0.0696** ± 0.0504  *(runs: 0.019, 0.071, 0.119)*
- `acc_a_before` = **0.1148** ± 0.0850  *(runs: 0.031, 0.113, 0.201)*
- `forgetting_ewc` = **0.0277** ± 0.0261  *(runs: 0.017, 0.009, 0.058)*
- `forgetting_ratio` = **0.7782** ± 0.6107  *(runs: 1.421, 0.206, 0.708)*
- `forgetting_std` = **0.0452** ± 0.0348  *(runs: 0.012, 0.043, 0.081)*
- `phase1_steps` = **1000.0000**  *(stable across seeds)*

**Notes:** Precondition failed: acc_A_before=0.031 < 0.70.

---
#### exp_9_4  ✓ SUPPORTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED']
**Hypothesis:** With MEMORY_SLOTS=12 and K extended to 0-10, an interior optimum exists at K=3-6 — fewer protected slots are insufficient to cover all 3 critical items, more wastes capacity on non-critical entries, creating a U-shaped accuracy curve.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 27s

**Metrics (mean ± std across seeds):**

- `acc_range` = **0.1490** ± 0.0030  *(runs: 0.150, 0.146, 0.151)*
- `accuracies` = [[0.1519, 0.1437, 0.0225, 0.0469, 0.105, 0.1725], [0.0381, 0.1575, 0.1031, 0.1837, 0.1062, 0.1638], [0.0294, 0.0306, 0.0912, 0.0725, 0.1806, 0.1469]]
- `critical_accuracies` = [[0.0328, 0.2215, 0.02, 0.0583, 0.0959, 0.1686], [0.0279, 0.1462, 0.1161, 0.19, 0.1144, 0.1644], [0.0267, 0.036, 0.112, 0.0818, 0.1821, 0.1317]]
- `interior_peak_exists` = [False, True, True]
- `is_flat` = [False, False, False]
- `is_monotone` = [False, False, False]
- `k_opt` = **4.0000** ± 1.0000  *(runs: 5.000, 3.000, 4.000)*
- `max_acc` = **0.1790** ± 0.0058  *(runs: 0.172, 0.184, 0.181)*
- `min_acc` = **0.0300** ± 0.0078  *(runs: 0.022, 0.038, 0.029)*
- `noncritical_accuracies` = [[0.3345, 0.0189, 0.0239, 0.0314, 0.1157, 0.1692], [0.0571, 0.1701, 0.0844, 0.1779, 0.0963, 0.1652], [0.0325, 0.0237, 0.0591, 0.0562, 0.1758, 0.1691]]

**Notes:** K values [0, 2, 4, 6, 8, 10], accs [0.152, 0.144, 0.022, 0.047, 0.105, 0.172]. Optimal K=5, max_acc=0.172. Interior peak: False.

---
#### exp_9_5  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** With all KV pairs concentrated in block 1 (positions 0-7 of 32), an adaptive write budget allocator learns non-uniform allocation (Gini > 0.5, block-1 fraction > 0.60), outperforming uniform allocation by > 5%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 40s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.0179** ± 0.1392  *(runs: 0.068, 0.125, -0.139)*
- `adaptive_acc` = **0.1725** ± 0.1240  *(runs: 0.227, 0.260, 0.031)*
- `block1_allocation_frac` = **0.6578** ± 0.5698  *(runs: 0.974, 1.000, 0.000)*
- `gini_coefficient` = **0.6617** ± 0.1410  *(runs: 0.736, 0.750, 0.499)*
- `mean_block_weights` = [[0.9738, 0.0004, 0.0004, 0.0255], [0.9997, 0.0, 0.0, 0.0003], [0.0, 0.4994, 0.4994, 0.0012]]
- `uniform_acc` = **0.1546** ± 0.0179  *(runs: 0.159, 0.135, 0.170)*

**Notes:** Uniform acc=0.159, adaptive acc=0.227, delta=0.068. Block-1 fraction=0.974, Gini=0.736. Block weights: [0.974, 0.0, 0.0, 0.026].

---

### Category 10 — Retroactive Writing Mechanism (Phase 2)
*0 supported / 1 refuted / 2 inconclusive / 0 error*

#### exp_10_1  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** The retroactive writing benefit decays to <5% accuracy gain when the revision gate's lookahead window is fewer than 6 tokens.

**Runs:** 5 (seeds: [123, 42, 42, 777, 777])  |  **Avg duration:** 213s

**Metrics (mean ± std across seeds):**

- `acc_w0` = **0.2591** ± 0.0068  *(runs: 0.264, 0.252, 0.264, 0.252, 0.264)*
- `acc_w2` = **0.1922** ± 0.0385  *(runs: 0.164, 0.234, 0.164, 0.234, 0.164)*
- `acc_w24` = **0.2172** ± 0.0171  *(runs: 0.205, 0.236, 0.205, 0.236, 0.205)*
- `acc_w4` = **0.2003** ± 0.0496  *(runs: 0.164, 0.255, 0.164, 0.255, 0.164)*
- `acc_w6` = **0.2185** ± 0.0188  *(runs: 0.205, 0.239, 0.205, 0.239, 0.205)*
- `acc_w8` = **0.2237** ± 0.0154  *(runs: 0.212, 0.241, 0.212, 0.241, 0.212)*
- `gap_at_24` = **-0.0419** ± 0.0240  *(runs: -0.059, -0.016, -0.059, -0.016, -0.059)*
- `gap_at_4` = **-0.0588** ± 0.0565  *(runs: -0.100, 0.003, -0.100, 0.003, -0.100)*
- `gap_w0` = **0.0000**  *(stable across seeds)*
- `gap_w2` = **-0.0669** ± 0.0454  *(runs: -0.100, -0.017, -0.100, -0.017, -0.100)*
- `gap_w24` = **-0.0419** ± 0.0240  *(runs: -0.059, -0.016, -0.059, -0.016, -0.059)*
- `gap_w4` = **-0.0588** ± 0.0565  *(runs: -0.100, 0.003, -0.100, 0.003, -0.100)*
- `gap_w6` = **-0.0406** ± 0.0257  *(runs: -0.059, -0.013, -0.059, -0.013, -0.059)*
- `gap_w8` = **-0.0353** ± 0.0223  *(runs: -0.052, -0.011, -0.052, -0.011, -0.052)*
- `max_gap_minus_min_gap` = **0.0681** ± 0.0437  *(runs: 0.100, 0.020, 0.100, 0.020, 0.100)*

**Notes:** gap@24=-0.059, gap@4=-0.100 — pattern inconclusive.

---
#### exp_10_2  ✗ REFUTED
**Hypothesis:** The retroactive writing benefit comes primarily (>80%) from adding new entries never written in the forward pass, not from re-encoding existing forward-pass entries with full context.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 62s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2297**  *(stable across seeds)*
- `acc_B` = **0.1984**  *(stable across seeds)*
- `acc_C` = **0.2313**  *(stable across seeds)*
- `acc_D` = **0.2562**  *(stable across seeds)*
- `gain_B_over_A` = **-0.0312**  *(stable across seeds)*
- `gain_C_over_A` = **0.0016**  *(stable across seeds)*
- `gain_D_over_A` = **0.0266**  *(stable across seeds)*
- `new_write_fraction` = **-1.1765**  *(stable across seeds)*

**Notes:** Overwrite gain (0.002) exceeds new-write gain (-0.031) by >0.03.

---
#### exp_10_3  ~ INCONCLUSIVE
**Hypothesis:** The retroactive writing accuracy gain scales with sequence length (Pearson r > 0.8 across seq_len 24, 32, 48, 64).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 86s

**Metrics (mean ± std across seeds):**

- `acc_fwd_24` = **0.1875**  *(stable across seeds)*
- `acc_fwd_32` = **0.1656**  *(stable across seeds)*
- `acc_fwd_48` = **0.0422**  *(stable across seeds)*
- `acc_fwd_64` = **0.0297**  *(stable across seeds)*
- `acc_retro_24` = **0.1344**  *(stable across seeds)*
- `acc_retro_32` = **0.0422**  *(stable across seeds)*
- `acc_retro_48` = **0.0328**  *(stable across seeds)*
- `acc_retro_64` = **0.0484**  *(stable across seeds)*
- `gap_24` = **-0.0531**  *(stable across seeds)*
- `gap_32` = **-0.1234**  *(stable across seeds)*
- `gap_48` = **-0.0094**  *(stable across seeds)*
- `gap_64` = **0.0188**  *(stable across seeds)*
- `pearson_r` = **0.7726**  *(stable across seeds)*

**Notes:** Pearson r=0.773 — scaling relationship inconclusive.

---

### Category 11 — Read Bottleneck Interventions (Phase 2)
*1 supported / 2 refuted / 0 inconclusive / 0 error*

#### exp_11_1  ✗ REFUTED
**Hypothesis:** A two-step query former (linear -> cross-attention over last 4 hidden states -> linear) shifts the bottleneck away from read accuracy (identified in exp_7_7).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `normal_acc_A` = **0.2109**  *(stable across seeds)*
- `normal_acc_B` = **0.1656**  *(stable across seeds)*
- `normal_acc_C` = **0.2000**  *(stable across seeds)*
- `oracle_acc_A_final` = **0.2922**  *(stable across seeds)*
- `oracle_acc_A_step0.2` = **0.1203**  *(stable across seeds)*
- `oracle_acc_A_step0.5` = **0.2516**  *(stable across seeds)*
- `oracle_acc_A_step1.0` = **0.2922**  *(stable across seeds)*
- `oracle_acc_B_final` = **0.1750**  *(stable across seeds)*
- `oracle_acc_B_minus_A` = **-0.1172**  *(stable across seeds)*
- `oracle_acc_B_step0.2` = **0.1062**  *(stable across seeds)*
- `oracle_acc_B_step0.5` = **0.1203**  *(stable across seeds)*
- `oracle_acc_B_step1.0` = **0.1750**  *(stable across seeds)*
- `oracle_acc_C_final` = **0.1953**  *(stable across seeds)*
- `oracle_acc_C_step0.2` = **0.1328**  *(stable across seeds)*
- `oracle_acc_C_step0.5` = **0.1625**  *(stable across seeds)*
- `oracle_acc_C_step1.0` = **0.1953**  *(stable across seeds)*

**Notes:** oracle_read_acc_B=0.175 not > oracle_read_acc_A=0.292 by 0.02.

---
#### exp_11_2  ✗ REFUTED
**Hypothesis:** Read-before-write duplicate suppression (skip write if cosine similarity to any existing memory slot > 0.8) improves retrieval F1 by >3% without reducing recall by >5%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 52s

**Metrics (mean ± std across seeds):**

- `f1_A` = **0.3432**  *(stable across seeds)*
- `f1_B` = **0.3088**  *(stable across seeds)*
- `f1_gain` = **-0.0343**  *(stable across seeds)*
- `precision_A` = **0.2256**  *(stable across seeds)*
- `precision_B` = **0.1856**  *(stable across seeds)*
- `recall_A` = **0.7163**  *(stable across seeds)*
- `recall_B` = **0.9187**  *(stable across seeds)*
- `recall_drop` = **-0.2025**  *(stable across seeds)*
- `write_rate_A` = **0.3810**  *(stable across seeds)*
- `write_rate_B` = **0.3714**  *(stable across seeds)*

**Notes:** F1_B=0.309 not > F1_A=0.343 by 0.01.

---
#### exp_11_3  ✓ SUPPORTED
**Hypothesis:** The optimal read suppression threshold T varies systematically by task type — factual QA, pattern matching, and sequence completion each have different optimal thresholds (differ by >0.15).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 55s

**Metrics (mean ± std across seeds):**

- `acc_copy_T0.2` = **0.0609**  *(stable across seeds)*
- `acc_copy_T0.4` = **0.0563**  *(stable across seeds)*
- `acc_copy_T0.6` = **0.0266**  *(stable across seeds)*
- `acc_copy_T0.8` = **0.0266**  *(stable across seeds)*
- `acc_copy_T0.95` = **0.0594**  *(stable across seeds)*
- `acc_factual_T0.2` = **0.2313**  *(stable across seeds)*
- `acc_factual_T0.4` = **0.2016**  *(stable across seeds)*
- `acc_factual_T0.6` = **0.1437**  *(stable across seeds)*
- `acc_factual_T0.8` = **0.1375**  *(stable across seeds)*
- `acc_factual_T0.95` = **0.1609**  *(stable across seeds)*
- `acc_pattern_T0.2` = **0.3781**  *(stable across seeds)*
- `acc_pattern_T0.4` = **0.7781**  *(stable across seeds)*
- `acc_pattern_T0.6` = **0.6109**  *(stable across seeds)*
- `acc_pattern_T0.8` = **0.5344**  *(stable across seeds)*
- `acc_pattern_T0.95` = **0.7969**  *(stable across seeds)*
- `max_pairwise_optimal_T_diff` = **0.7500**  *(stable across seeds)*
- `optimal_T_copy` = **0.2000**  *(stable across seeds)*
- `optimal_T_factual` = **0.2000**  *(stable across seeds)*
- `optimal_T_pattern` = **0.9500**  *(stable across seeds)*
- `supp_copy_T0.2` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.4` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.6` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.8` = **0.0000**  *(stable across seeds)*
- `supp_copy_T0.95` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.2` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.4` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.6` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.8` = **0.0000**  *(stable across seeds)*
- `supp_factual_T0.95` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.2` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.4` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.6` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.8` = **0.0000**  *(stable across seeds)*
- `supp_pattern_T0.95` = **0.0000**  *(stable across seeds)*

**Notes:** Max pairwise T difference=0.750>0.15. Optimal T: factual=0.2, pattern=0.95, copy=0.2.

---

### Category 12 — Compression Hard Regimes (Phase 2)
*0 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_12_1  ✗ REFUTED
**Hypothesis:** At 64x compression with 100-way gallery discrimination, retrieval-objective compressor achieves >=15% higher Acc@1 than reconstruction-objective compressor.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 15s

**Metrics (mean ± std across seeds):**

- `acc1_A` = **1.0000**  *(stable across seeds)*
- `acc1_B` = **1.0000**  *(stable across seeds)*
- `recon_cosim_A` = **0.1205** ± 0.0046  *(runs: 0.124, 0.122, 0.115)*
- `recon_cosim_B` = **0.0044** ± 0.0009  *(runs: 0.004, 0.004, 0.005)*
- `retrieval_gap` = **0.0000**  *(stable across seeds)*

**Notes:** Both models >80% Acc@1 or gap <5%.

---
#### exp_12_2  ~ INCONCLUSIVE
**Hypothesis:** The 2x-8x compression training failure is gradient starvation from wide bottleneck, not structural — LR warmup restores training.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 3s

**Metrics (mean ± std across seeds):**

- `a_fails_at_2x_4x` = **0.0000**  *(stable across seeds)*
- `b_still_bad_count` = **0.0000**  *(stable across seeds)*
- `ratio16_schedA_cosim_100` = **0.2010** ± 0.0017  *(runs: 0.201, 0.203, 0.200)*
- `ratio16_schedA_cosim_final` = **0.2235** ± 0.0017  *(runs: 0.223, 0.226, 0.222)*
- `ratio16_schedB_cosim_100` = **0.0170** ± 0.0062  *(runs: 0.011, 0.024, 0.016)*
- `ratio16_schedB_cosim_final` = **0.2033** ± 0.0055  *(runs: 0.209, 0.198, 0.203)*
- `ratio16_schedC_cosim_100` = **0.0033** ± 0.0016  *(runs: 0.005, 0.003, 0.002)*
- `ratio16_schedC_cosim_final` = **0.0229** ± 0.0120  *(runs: 0.017, 0.015, 0.037)*
- `ratio2_schedA_cosim_100` = **0.3955** ± 0.0100  *(runs: 0.391, 0.407, 0.389)*
- `ratio2_schedA_cosim_final` = **0.5940** ± 0.0126  *(runs: 0.599, 0.603, 0.580)*
- `ratio2_schedB_cosim_100` = **0.0267** ± 0.0042  *(runs: 0.023, 0.031, 0.026)*
- `ratio2_schedB_cosim_final` = **0.3891** ± 0.0059  *(runs: 0.388, 0.384, 0.396)*
- `ratio2_schedC_cosim_100` = **0.0054** ± 0.0153  *(runs: 0.011, 0.017, -0.012)*
- `ratio2_schedC_cosim_final` = **0.0193** ± 0.0148  *(runs: 0.036, 0.011, 0.010)*
- `ratio4_schedA_cosim_100` = **0.3354** ± 0.0041  *(runs: 0.338, 0.338, 0.331)*
- `ratio4_schedA_cosim_final` = **0.4580** ± 0.0050  *(runs: 0.463, 0.454, 0.457)*
- `ratio4_schedB_cosim_100` = **0.0304** ± 0.0047  *(runs: 0.027, 0.028, 0.036)*
- `ratio4_schedB_cosim_final` = **0.3181** ± 0.0128  *(runs: 0.304, 0.329, 0.322)*
- `ratio4_schedC_cosim_100` = **0.0114** ± 0.0135  *(runs: 0.011, 0.025, -0.002)*
- `ratio4_schedC_cosim_final` = **0.0285** ± 0.0128  *(runs: 0.017, 0.026, 0.042)*
- `ratio8_schedA_cosim_100` = **0.2569** ± 0.0049  *(runs: 0.257, 0.262, 0.252)*
- `ratio8_schedA_cosim_final` = **0.3317** ± 0.0028  *(runs: 0.332, 0.334, 0.329)*
- `ratio8_schedB_cosim_100` = **0.0221** ± 0.0062  *(runs: 0.017, 0.020, 0.029)*
- `ratio8_schedB_cosim_final` = **0.2627** ± 0.0078  *(runs: 0.272, 0.258, 0.259)*
- `ratio8_schedC_cosim_100` = **0.0096** ± 0.0088  *(runs: 0.003, 0.020, 0.006)*
- `ratio8_schedC_cosim_final` = **0.0197** ± 0.0091  *(runs: 0.010, 0.028, 0.022)*
- `warmup_restores_at_2x_4x` = **0.0000**  *(stable across seeds)*

**Notes:** A does not fail at 2x-4x; hypothesis conditions not triggered.

---

### Category 13 — Compositional Retrieval at Scale (Phase 2)
*2 supported / 0 refuted / 0 inconclusive / 0 error*

#### exp_13_1  ✓ SUPPORTED
**Hypothesis:** The two-hop retrieval regularization effect (exp_4_9) persists at 64-entity KB with 40% near-duplicate interference.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1s

**Metrics (mean ± std across seeds):**

- `interference_gap` = **0.1200**  *(stable across seeds)*
- `interference_sh_acc` = **0.0800**  *(stable across seeds)*
- `interference_th_acc` = **0.2000**  *(stable across seeds)*
- `single_hop_acc` = **0.0625**  *(stable across seeds)*
- `two_hop_acc` = **0.1250**  *(stable across seeds)*
- `two_hop_vs_single_gap` = **0.0625**  *(stable across seeds)*

**Notes:** Two-hop acc within 5% of single-hop on interference subset (gap=0.120).

---
#### exp_13_2  ✓ SUPPORTED
**Hypothesis:** Three-hop compositional retrieval retains >50% of two-hop accuracy (degradation_ratio > 0.50) at hidden_dim=64.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1s

**Metrics (mean ± std across seeds):**

- `acc_single` = **0.0416** ± 0.0181  *(runs: 0.062, 0.031, 0.031)*
- `acc_three` = **0.1146** ± 0.0180  *(runs: 0.125, 0.094, 0.125)*
- `acc_two` = **0.0312**  *(stable across seeds)*
- `degradation_ratio` = **3.6667** ± 0.5774  *(runs: 4.000, 3.000, 4.000)*

**Notes:** Three-hop retains 4.00 of two-hop accuracy (>0.50).

---

### Category 14 — System Integration (Phase 2)
*0 supported / 3 refuted / 0 inconclusive / 0 error*

#### exp_14_1  ✗ REFUTED
**Hypothesis:** Combining retroactive write and read confidence suppression yields super-additive accuracy gains.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2109**  *(stable across seeds)*
- `acc_B` = **0.2047**  *(stable across seeds)*
- `acc_C` = **0.1359**  *(stable across seeds)*
- `acc_D` = **0.0734**  *(stable across seeds)*
- `gap_B` = **-0.0063**  *(stable across seeds)*
- `gap_C` = **-0.0750**  *(stable across seeds)*
- `gap_D` = **-0.1375**  *(stable across seeds)*
- `super_additive` = [False, False, False]

**Notes:** gap_D=-0.138 < max(gap_B,gap_C)+0.005=-0.001.

---
#### exp_14_2  ✗ REFUTED
**Hypothesis:** Write-first curriculum (train write gate before enabling reads) outperforms joint training from step 0.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 7s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2094**  *(stable across seeds)*
- `acc_B` = **0.1812**  *(stable across seeds)*
- `acc_diff_B_minus_A` = **-0.0281**  *(stable across seeds)*
- `wq_diff_B_minus_A` = **0.0000**  *(stable across seeds)*
- `write_quality_at_1000_A` = **1.0000**  *(stable across seeds)*
- `write_quality_at_1000_B` = **1.0000**  *(stable across seeds)*

**Notes:** Joint training A >= curriculum B - 0.01: acc_A=0.209, acc_B=0.181.

---
#### exp_14_3  ✗ REFUTED
**Hypothesis:** Cosine-annealed Gumbel temperature (1.0->0.1) produces higher final accuracy than constant temperature 0.5 by >2%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1906**  *(stable across seeds)*
- `acc_B` = **0.1750**  *(stable across seeds)*
- `acc_C` = **0.1531**  *(stable across seeds)*
- `diff_C_vs_A` = **-0.0375**  *(stable across seeds)*

**Notes:** Constant A >= cosine C - 0.01: acc_A=0.191, acc_C=0.153.

---

### Category 15 — Delta Rule / Associative Matrix Writes (Phase 3)
*1 supported / 2 refuted / 1 inconclusive / 0 error*

#### exp_15_1  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** A delta rule associative matrix write outperforms standard slot write by >5% due to built-in interference correction.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.1056** ± 0.0174  *(runs: 0.109, 0.087, 0.121)*
- `acc_gap` = **0.0258** ± 0.0427  *(runs: 0.004, -0.001, 0.075)*
- `acc_slot` = **0.0798** ± 0.0303  *(runs: 0.105, 0.088, 0.046)*
- `interference_delta` = **0.9525** ± 0.0299  *(runs: 0.967, 0.918, 0.973)*
- `interference_gap` = **-0.4549** ± 0.0925  *(runs: -0.449, -0.550, -0.365)*
- `interference_slot` = **0.4976** ± 0.1208  *(runs: 0.518, 0.368, 0.607)*

**Notes:** Slot memory acc=0.105 >= delta acc=0.109 - 0.02.

---
#### exp_15_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** The correction term in the delta rule is essential — Hebbian M += v*k^T degrades by >10% on key-interference tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 29s

**Metrics (mean ± std across seeds):**

- `delta_acc_clean` = **0.1096** ± 0.0091  *(runs: 0.099, 0.113, 0.117)*
- `delta_acc_interfered` = **0.1162** ± 0.0277  *(runs: 0.106, 0.095, 0.147)*
- `delta_vs_hebbian_gap_on_interference` = **0.0154** ± 0.0359  *(runs: 0.010, -0.018, 0.054)*
- `hebbian_acc_clean` = **0.1017** ± 0.0164  *(runs: 0.083, 0.114, 0.107)*
- `hebbian_acc_interfered` = **0.1009** ± 0.0102  *(runs: 0.096, 0.113, 0.094)*
- `norm_hebbian_acc_clean` = **0.1058** ± 0.0058  *(runs: 0.107, 0.111, 0.099)*
- `norm_hebbian_acc_interfered` = **0.1059** ± 0.0083  *(runs: 0.110, 0.111, 0.096)*

**Notes:** Hebbian acc_int=0.096 within 0.03 of delta acc_int=0.106.

---
#### exp_15_3  ✓ SUPPORTED
**Hypothesis:** Energy-gated delta rule (write only when delta_E < 0) achieves >90% accuracy of continuous write at <70% write rate.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 26s

**Metrics (mean ± std across seeds):**

- `acc_A_continuous` = **0.1310** ± 0.0293  *(runs: 0.138, 0.099, 0.156)*
- `acc_B_energy_gated` = **0.1477** ± 0.0219  *(runs: 0.127, 0.171, 0.146)*
- `acc_C_learned_gate` = **0.1727** ± 0.0185  *(runs: 0.154, 0.174, 0.191)*
- `acc_ratio_B` = **1.1928** ± 0.4634  *(runs: 0.919, 1.728, 0.932)*
- `acc_ratio_C` = **1.3642** ± 0.3465  *(runs: 1.113, 1.760, 1.220)*
- `write_rate_A` = **1.0000**  *(stable across seeds)*
- `write_rate_B` = **0.5159** ± 0.0029  *(runs: 0.519, 0.515, 0.514)*
- `write_rate_C` = **0.2650** ± 0.3384  *(runs: 0.070, 0.656, 0.070)*

**Notes:** Energy gate: acc_ratio=0.919>0.90, write_rate=0.519<0.70. Hypothesis confirmed.

---
#### exp_15_4  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'INCONCLUSIVE']
**Hypothesis:** Delta rule outperforms Larimar outer-product write on overwrite tasks (same key, updated value).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 13s

**Metrics (mean ± std across seeds):**

- `delta_acc_normal` = **0.1663** ± 0.0159  *(runs: 0.166, 0.151, 0.182)*
- `delta_acc_overall` = **0.1700** ± 0.0196  *(runs: 0.168, 0.152, 0.191)*
- `delta_acc_update` = **0.1735** ± 0.0234  *(runs: 0.169, 0.153, 0.199)*
- `larimar_acc_normal` = **0.1319** ± 0.0432  *(runs: 0.182, 0.106, 0.107)*
- `larimar_acc_overall` = **0.1338** ± 0.0400  *(runs: 0.180, 0.106, 0.116)*
- `larimar_acc_update` = **0.1357** ± 0.0373  *(runs: 0.177, 0.106, 0.124)*
- `update_gap_delta_minus_larimar` = **0.0379** ± 0.0424  *(runs: -0.008, 0.047, 0.075)*

**Notes:** Larimar acc_update=0.177 >= delta acc_update=0.169 - 0.02.

---

### Category 16 — Online Gradient Descent Memory / Titans-Style (Phase 3)
*1 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_16_1  ✗ REFUTED
**Hypothesis:** Parametric MLP memory (1 gradient step per token) outperforms fixed-slot memory at matched parameter count.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 6s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **-0.0117** ± 0.0113  *(runs: 0.000, -0.013, -0.022)*
- `acc_parametric` = **0.0342** ± 0.0128  *(runs: 0.045, 0.037, 0.020)*
- `acc_slot` = **0.0458** ± 0.0038  *(runs: 0.045, 0.050, 0.043)*
- `param_count_mlp` = **552.0000**  *(stable across seeds)*
- `param_count_slot_storage` = **256.0000**  *(stable across seeds)*

**Notes:** Slot acc=0.045 >= parametric acc=0.045 - 0.02.

---
#### exp_16_2  ~ INCONCLUSIVE
**Hypothesis:** Skipping MLP memory updates for low-surprise tokens achieves same accuracy with >40% fewer update steps.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 18s

**Metrics (mean ± std across seeds):**

- `baseline_acc` = **0.0417** ± 0.0038  *(runs: 0.043, 0.037, 0.045)*
- `baseline_write_rate` = **1.0000**  *(stable across seeds)*
- `pareto_found` = [False, False, False]
- `per_threshold.0.0.acc` = **0.0417** ± 0.0038  *(runs: 0.043, 0.037, 0.045)*
- `per_threshold.0.0.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.0.3.acc` = **0.0267** ± 0.0038  *(runs: 0.022, 0.028, 0.030)*
- `per_threshold.0.3.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.0.6.acc` = **0.0325** ± 0.0198  *(runs: 0.018, 0.025, 0.055)*
- `per_threshold.0.6.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.1.0.acc` = **0.0308** ± 0.0095  *(runs: 0.020, 0.035, 0.037)*
- `per_threshold.1.0.write_rate` = **1.0000**  *(stable across seeds)*
- `per_threshold.1.5.acc` = **0.0333** ± 0.0063  *(runs: 0.040, 0.028, 0.033)*
- `per_threshold.1.5.write_rate` = **1.0000**  *(stable across seeds)*

**Notes:** Smooth degradation curve; no clear pareto knee found.

---
#### exp_16_3  ✓ SUPPORTED
**Hypothesis:** Parametric memory scales more gracefully with seq_len than slot memory (higher accuracy retention at 4x length).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 14s

**Metrics (mean ± std across seeds):**

- `acc_parametric.24` = **0.0354** ± 0.0110  *(runs: 0.025, 0.047, 0.034)*
- `acc_parametric.48` = **0.0291** ± 0.0130  *(runs: 0.025, 0.019, 0.044)*
- `acc_slot.24` = **0.0844** ± 0.0109  *(runs: 0.078, 0.078, 0.097)*
- `acc_slot.48` = **0.0323** ± 0.0148  *(runs: 0.044, 0.016, 0.037)*
- `retention_diff` = **0.5076** ± 0.3471  *(runs: 0.441, 0.199, 0.883)*
- `retention_parametric` = **0.8897** ± 0.4462  *(runs: 1.000, 0.399, 1.270)*
- `retention_slot` = **0.3821** ± 0.1800  *(runs: 0.559, 0.200, 0.387)*

**Notes:** Parametric retention=1.000 vs slot retention=0.560; diff=0.440 > 0.15. Scales more gracefully.

---

### Category 17 — Prospective / Query-Conditioned Writing (Phase 3)
*0 supported / 3 refuted / 1 inconclusive / 0 error*

#### exp_17_1  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** A write gate conditioned on predicted future query type outperforms context-only gate by >5% on tasks with 4 different query types.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1420** ± 0.0818  *(runs: 0.222, 0.059, 0.145)*
- `acc_B` = **0.0823** ± 0.0406  *(runs: 0.057, 0.129, 0.060)*
- `gap` = **-0.0597** ± 0.1196  *(runs: -0.165, 0.070, -0.084)*
- `query_pred_acc` = **0.7168** ± 0.1323  *(runs: 0.583, 0.720, 0.847)*

**Notes:** Context-only gate matches or beats query-conditioned (gap=-0.165)

---
#### exp_17_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** K-token lookahead async write gate outperforms same-time write by >3% at some K in {2, 4, 6}.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 24s

**Metrics (mean ± std across seeds):**

- `acc_K0` = **0.1303** ± 0.0680  *(runs: 0.208, 0.080, 0.103)*
- `acc_K2` = **0.0813** ± 0.0851  *(runs: 0.032, 0.179, 0.032)*
- `acc_K4` = **0.1263** ± 0.0586  *(runs: 0.194, 0.092, 0.093)*
- `acc_K6` = **0.0376** ± 0.0088  *(runs: 0.032, 0.034, 0.048)*
- `best_gap` = **0.0252** ± 0.0641  *(runs: -0.014, 0.099, -0.010)*
- `best_k` = **3.3333** ± 1.1547  *(runs: 4.000, 2.000, 4.000)*

**Notes:** All lookahead K within 0.02 of K=0 (best gap=-0.014)

---
#### exp_17_3  ~ INCONCLUSIVE
**Hypothesis:** Prospective and retroactive writing are redundant — their combination yields <1.5x the gain of either alone.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1471** ± 0.0726  *(runs: 0.219, 0.074, 0.149)*
- `acc_B` = **0.1428** ± 0.0529  *(runs: 0.086, 0.191, 0.151)*
- `acc_C` = **0.0719** ± 0.0190  *(runs: 0.092, 0.069, 0.054)*
- `acc_D` = **0.1581** ± 0.0254  *(runs: 0.185, 0.135, 0.154)*
- `gap_B` = **-0.0044** ± 0.1251  *(runs: -0.133, 0.117, 0.002)*
- `gap_C` = **-0.0752** ± 0.0633  *(runs: -0.127, -0.004, -0.095)*
- `gap_D` = **0.0109** ± 0.0477  *(runs: -0.034, 0.061, 0.005)*
- `multiplier` = **-10.2538** ± 20.2328  *(runs: -33.594, 0.522, 2.310)*

**Notes:** gap_B=-0.133 or gap_C=-0.127 too small for valid comparison

---
#### exp_17_4  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Query-conditioned write gain scales linearly with query predictability (Pearson r > 0.85).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 38s

**Metrics (mean ± std across seeds):**

- `acc_A_p000` = **0.1480** ± 0.0764  *(runs: 0.221, 0.069, 0.154)*
- `acc_A_p033` = **0.1025** ± 0.0690  *(runs: 0.110, 0.030, 0.168)*
- `acc_A_p066` = **0.0601** ± 0.0431  *(runs: 0.040, 0.110, 0.031)*
- `acc_A_p100` = **0.1468** ± 0.0597  *(runs: 0.082, 0.199, 0.160)*
- `acc_B_p000` = **0.1544** ± 0.0444  *(runs: 0.195, 0.162, 0.107)*
- `acc_B_p033` = **0.1081** ± 0.0793  *(runs: 0.104, 0.189, 0.031)*
- `acc_B_p066` = **0.0462** ± 0.0259  *(runs: 0.031, 0.076, 0.031)*
- `acc_B_p100` = **0.0427** ± 0.0151  *(runs: 0.060, 0.034, 0.034)*
- `gap_p000` = **0.0064** ± 0.0757  *(runs: -0.026, 0.093, -0.048)*
- `gap_p033` = **0.0057** ± 0.1482  *(runs: -0.005, 0.159, -0.137)*
- `gap_p066` = **-0.0139** ± 0.0174  *(runs: -0.008, -0.034, 0.000)*
- `gap_p100` = **-0.1041** ± 0.0741  *(runs: -0.021, -0.165, -0.126)*
- `gap_variance` = **0.0062** ± 0.0081  *(runs: 0.000, 0.015, 0.003)*
- `pearson_r` = **-0.3135** ± 0.5119  *(runs: 0.132, -0.873, -0.200)*

**Notes:** Gap is effectively constant (variance=0.000077)

---

### Category 18 — Tiered Memory Architecture (Phase 3)
*0 supported / 1 refuted / 3 inconclusive / 0 error*

#### exp_18_1  ✗ REFUTED
**Hypothesis:** Two-tier memory (16-slot fast + 64-slot slow with learned demotion) outperforms flat 64-slot memory by >5% on long-context tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 192s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0449** ± 0.0117  *(runs: 0.032, 0.047, 0.055)*
- `acc_B` = **0.0408** ± 0.0190  *(runs: 0.029, 0.063, 0.030)*
- `acc_C` = **0.0328** ± 0.0004  *(runs: 0.032, 0.033, 0.033)*
- `gap_B` = **-0.0041** ± 0.0206  *(runs: -0.003, 0.016, -0.025)*
- `gap_C` = **-0.0120** ± 0.0113  *(runs: 0.000, -0.014, -0.022)*
- `slow_coverage_B` = **0.3631** ± 0.0476  *(runs: 0.409, 0.367, 0.314)*
- `slow_coverage_C` = **0.3951** ± 0.0295  *(runs: 0.425, 0.366, 0.394)*

**Notes:** Flat memory matches tiered (gap=-0.003)

---
#### exp_18_2  ~ INCONCLUSIVE
**Hypothesis:** Learned demotion controller discovers frequency-not-recency policy (corr_access > 0.15, corr_recency < -0.10).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 48s

**Metrics (mean ± std across seeds):**

- `corr_access` = **0.2119** ± 0.0239  *(runs: 0.237, 0.189, 0.210)*
- `corr_content_norm` = **0.2629** ± 0.1190  *(runs: 0.399, 0.211, 0.179)*
- `corr_recency` = **0.2354** ± 0.0684  *(runs: 0.175, 0.309, 0.222)*
- `n_demotion_events` = **3263.6667** ± 807.1557  *(runs: 2488.000, 4099.000, 3204.000)*

**Notes:** corr_access=0.237 > 0.15 but corr_recency=0.175 > -0.10

---
#### exp_18_3  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE']
**Hypothesis:** Tiered memory has a capacity crossover point — flat is better below it, tiered above it.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 390s

**Metrics (mean ± std across seeds):**

- `acc_flat_128` = **0.0471** ± 0.0244  *(runs: 0.032, 0.075, 0.035)*
- `acc_flat_16` = **0.0549** ± 0.0029  *(runs: 0.052, 0.057, 0.056)*
- `acc_flat_32` = **0.0479** ± 0.0154  *(runs: 0.030, 0.060, 0.053)*
- `acc_flat_64` = **0.0550** ± 0.0228  *(runs: 0.077, 0.032, 0.056)*
- `acc_flat_8` = **0.0396** ± 0.0133  *(runs: 0.033, 0.055, 0.030)*
- `acc_tiered_128` = **0.0399** ± 0.0123  *(runs: 0.054, 0.033, 0.033)*
- `acc_tiered_16` = **0.0320** ± 0.0022  *(runs: 0.030, 0.035, 0.031)*
- `acc_tiered_32` = **0.0306** ± 0.0023  *(runs: 0.030, 0.033, 0.029)*
- `acc_tiered_64` = **0.0328** ± 0.0015  *(runs: 0.034, 0.033, 0.031)*
- `acc_tiered_8` = **0.0543** ± 0.0038  *(runs: 0.055, 0.050, 0.058)*
- `crossover_capacity` = **26.6667** ± 32.3316  *(runs: 8.000, 64.000, 8.000)*

**Notes:** Crossover at 8 outside expected range 16-64

---
#### exp_18_4  ~ INCONCLUSIVE
**Hypothesis:** Simultaneous cross-tier retrieval outperforms cascaded sequential retrieval by >3%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 200s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0322** ± 0.0006  *(runs: 0.032, 0.033, 0.032)*
- `acc_B` = **0.0652** ± 0.0102  *(runs: 0.058, 0.061, 0.077)*
- `acc_C` = **0.0717** ± 0.0226  *(runs: 0.098, 0.056, 0.062)*
- `gap_A_vs_B` = **-0.0330** ± 0.0105  *(runs: -0.026, -0.028, -0.045)*
- `gap_A_vs_C` = **-0.0395** ± 0.0230  *(runs: -0.066, -0.023, -0.030)*
- `gap_A_vs_best_seq` = **-0.0462** ± 0.0190  *(runs: -0.066, -0.028, -0.045)*
- `slow_access_rate_B` = **0.3806** ± 0.0565  *(runs: 0.339, 0.358, 0.445)*

**Notes:** Simultaneous best but gap=-0.066 < 0.03

---

### Category 19 — Sparse Hopfield Addressing (Phase 3)
*0 supported / 3 refuted / 0 inconclusive / 0 error*

#### exp_19_1  ✗ REFUTED
**Hypothesis:** Sparse Hopfield retrieval (sparsemax top-k=2) outperforms standard softmax attention by >5% precision@1 on 40% interference tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 15s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0228**  *(stable across seeds)*
- `acc_B` = **0.0175**  *(stable across seeds)*
- `acc_C` = **0.0316**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **-0.0053**  *(stable across seeds)*
- `precision_A` = **0.0000**  *(stable across seeds)*
- `precision_B` = **0.0000**  *(stable across seeds)*
- `precision_C` = **0.0316**  *(stable across seeds)*
- `precision_gap_B_minus_A` = **0.0000**  *(stable across seeds)*

**Notes:** Soft attention matches or beats sparse on interference tasks.

---
#### exp_19_2  ✗ REFUTED
**Hypothesis:** Hopfield energy write criterion (write only if ΔE < 0) produces <35% write rate with >3% accuracy improvement.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 46s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1291**  *(stable across seeds)*
- `acc_B` = **0.0297**  *(stable across seeds)*
- `acc_C` = **0.1228**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **-0.0994**  *(stable across seeds)*
- `write_rate_B` = **0.8214**  *(stable across seeds)*
- `write_rate_C` = **0.1367**  *(stable across seeds)*

**Notes:** Energy writes too frequent (0.8214) or acc dropped vs learned gate.

---
#### exp_19_3  ✗ REFUTED
**Hypothesis:** Sparse Hopfield sustains accuracy 2+ patterns longer than dense before capacity cliff.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 65s

**Metrics (mean ± std across seeds):**

- `acc_soft_12` = **0.0200**  *(stable across seeds)*
- `acc_soft_16` = **0.0206**  *(stable across seeds)*
- `acc_soft_20` = **0.0147**  *(stable across seeds)*
- `acc_soft_24` = **0.0144**  *(stable across seeds)*
- `acc_soft_4` = **0.1212**  *(stable across seeds)*
- `acc_soft_8` = **0.0256**  *(stable across seeds)*
- `acc_sparse_12` = **0.0244**  *(stable across seeds)*
- `acc_sparse_16` = **0.0194**  *(stable across seeds)*
- `acc_sparse_20` = **0.0147**  *(stable across seeds)*
- `acc_sparse_24` = **0.0144**  *(stable across seeds)*
- `acc_sparse_4` = **0.1253**  *(stable across seeds)*
- `acc_sparse_8` = **0.0344**  *(stable across seeds)*
- `capacity_cliff_soft` = **28.0000**  *(stable across seeds)*
- `capacity_cliff_sparse` = **28.0000**  *(stable across seeds)*
- `cliff_diff_sparse_minus_soft` = **0.0000**  *(stable across seeds)*

**Notes:** Capacity cliff differs by only 0 (< 1).

---

### Category 20 — Three-Gate Coordinated Controller (Phase 3)
*0 supported / 2 refuted / 2 inconclusive / 0 error*

#### exp_20_1  ~ INCONCLUSIVE
**Hypothesis:** L1 auxiliary loss on write gate (targeting ~15% activity) improves accuracy and avoids degenerate modes.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1672**  *(stable across seeds)*
- `acc_B` = **0.1891**  *(stable across seeds)*
- `acc_C` = **0.1806**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **0.0219**  *(stable across seeds)*
- `is_collapsed_A` = [False, False, False]
- `is_collapsed_B` = [False, False, False]
- `is_collapsed_C` = [True, True, True]
- `write_rate_A` = **0.4111**  *(stable across seeds)*
- `write_rate_B` = **0.0357**  *(stable across seeds)*
- `write_rate_C` = **0.0159**  *(stable across seeds)*

**Notes:** Auxiliary adjusts write rate but accuracy gain (0.0219) below threshold or write_rate out of range.

---
#### exp_20_2  ✗ REFUTED
**Hypothesis:** Explicit read accuracy auxiliary loss reduces the read bottleneck more effectively than implicit task loss.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1672**  *(stable across seeds)*
- `acc_B` = **0.1691**  *(stable across seeds)*
- `acc_C` = **0.1762**  *(stable across seeds)*
- `acc_gap_B_minus_A` = **0.0019**  *(stable across seeds)*
- `oracle_gap_B_minus_A` = **0.0050**  *(stable across seeds)*
- `oracle_read_acc_A` = **0.1578**  *(stable across seeds)*
- `oracle_read_acc_B` = **0.1628**  *(stable across seeds)*
- `oracle_read_acc_C` = **0.1744**  *(stable across seeds)*

**Notes:** Task acc unchanged (gap=0.0019 <= 0.005).

---
#### exp_20_3  ✗ REFUTED
**Hypothesis:** Three-gate controller with all auxiliary losses combined outperforms any single-auxiliary system.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2353**  *(stable across seeds)*
- `acc_B` = **0.2506**  *(stable across seeds)*
- `acc_C` = **0.2425**  *(stable across seeds)*
- `acc_D` = **0.2334**  *(stable across seeds)*
- `acc_E` = **0.2431**  *(stable across seeds)*
- `best_single_aux` = **0.2506**  *(stable across seeds)*
- `gap_E_minus_A` = **0.0078**  *(stable across seeds)*
- `gap_E_minus_best_single` = **-0.0075**  *(stable across seeds)*

**Notes:** No-auxiliary baseline (acc_A=0.2353) nearly matches full system (acc_E=0.2431).

---
#### exp_20_4  ~ INCONCLUSIVE
**Hypothesis:** Optimal write sparsity auxiliary weight is in [0.01, 0.1] — outside this range gate collapses or task signal drowns.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 18s

**Metrics (mean ± std across seeds):**

- `acc_lam_0_0` = **0.1663**  *(stable across seeds)*
- `acc_lam_0_01` = **0.2018**  *(stable across seeds)*
- `acc_lam_0_1` = **0.1985**  *(stable across seeds)*
- `acc_lam_1_0` = **0.2062**  *(stable across seeds)*
- `below_left_of_range` = [False, False, False]
- `below_right_of_range` = [False, False, False]
- `collapsed_lam_0_0` = [False, False, False]
- `collapsed_lam_0_01` = [False, False, False]
- `collapsed_lam_0_1` = [False, False, False]
- `collapsed_lam_1_0` = [True, True, True]
- `monotone_decay` = [False, False, False]
- `peak_acc` = **0.2018**  *(stable across seeds)*
- `peak_in_range` = [True, True, True]
- `peak_lambda` = **0.0100**  *(stable across seeds)*
- `wr_lam_0_0` = **0.4075**  *(stable across seeds)*
- `wr_lam_0_01` = **0.0350**  *(stable across seeds)*
- `wr_lam_0_1` = **0.0247**  *(stable across seeds)*
- `wr_lam_1_0` = **0.0078**  *(stable across seeds)*

**Notes:** Peak at lambda=0.01 (in_range=True); boundary conditions not fully met.

---

### Category 21 — Feedforward Controller + Hindsight Distillation (Phase 3)
*0 supported / 4 refuted / 0 inconclusive / 0 error*

#### exp_21_1  ✗ REFUTED
**Hypothesis:** A feedforward-only memory controller achieves higher external memory utilization than an LSTM controller at equal parameter count.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 8s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0313** ± 0.0040  *(runs: 0.027, 0.035, 0.032)*
- `acc_B` = **0.0338** ± 0.0041  *(runs: 0.038, 0.031, 0.033)*
- `acc_gap` = **0.0026** ± 0.0081  *(runs: 0.012, -0.004, 0.000)*
- `params_A` = **10545.0000**  *(stable across seeds)*
- `params_B` = **9473.0000**  *(stable across seeds)*
- `util_A` = **0.0840** ± 0.0622  *(runs: 0.154, 0.036, 0.062)*
- `util_B` = **0.0209** ± 0.0044  *(runs: 0.018, 0.026, 0.019)*
- `util_gap` = **-0.0631** ± 0.0658  *(runs: -0.137, -0.010, -0.042)*

**Notes:** LSTM: acc=0.0269 util=0.1542. FF: acc=0.0384 util=0.0175. acc_gap=0.0116 util_gap=-0.1368.

---
#### exp_21_2  ✗ REFUTED
**Hypothesis:** Hindsight oracle labels (which writes were causally relevant) provide a stronger training signal than task loss alone (+3% accuracy).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 13s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0321** ± 0.0040  *(runs: 0.028, 0.034, 0.034)*
- `acc_B` = **0.0316** ± 0.0013  *(runs: 0.033, 0.031, 0.031)*
- `acc_gap` = **-0.0004** ± 0.0052  *(runs: 0.006, -0.003, -0.004)*
- `gate_quality_r` = **0.0781** ± 0.1465  *(runs: 0.077, -0.068, 0.225)*

**Notes:** Task-only: acc=0.0275. Oracle-augmented: acc=0.0331. gap=0.0056, gate_quality_r=0.0773.

---
#### exp_21_3  ✗ REFUTED
**Hypothesis:** Write gate distilled from oracle labels (trained primarily on oracle supervision) achieves higher accuracy than end-to-end learned gate.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 24s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0321** ± 0.0040  *(runs: 0.028, 0.034, 0.034)*
- `acc_B` = **0.0294** ± 0.0053  *(runs: 0.024, 0.035, 0.029)*
- `acc_C` = **0.0367** ± 0.0058  *(runs: 0.041, 0.039, 0.030)*
- `acc_gap_BA` = **-0.0027** ± 0.0031  *(runs: -0.003, 0.001, -0.006)*
- `gate_quality_A` = **0.0019** ± 0.0181  *(runs: -0.011, -0.006, 0.023)*
- `gate_quality_B` = **0.0270** ± 0.0294  *(runs: 0.059, 0.019, 0.002)*
- `gate_quality_C` = **0.1177** ± 0.0567  *(runs: 0.179, 0.108, 0.067)*
- `quality_gap_BA` = **0.0251** ± 0.0453  *(runs: 0.070, 0.025, -0.020)*

**Notes:** A(e2e): acc=0.0275 gq=-0.0108. B(distilled): acc=0.0244 gq=0.0595. C(mixed): acc=0.0406 gq=0.1786. acc_gap(B-A)=-0.0031 quality_gap=0.0703.

---
#### exp_21_4  ✗ REFUTED
**Hypothesis:** Feedforward controller + hindsight distillation is the strongest write policy (outperforms either alone and LSTM baseline).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 43s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.0348** ± 0.0051  *(runs: 0.033, 0.041, 0.031)*
- `acc_B` = **0.0338** ± 0.0050  *(runs: 0.039, 0.032, 0.030)*
- `acc_C` = **0.0325** ± 0.0017  *(runs: 0.032, 0.031, 0.034)*
- `acc_D` = **0.0315** ± 0.0069  *(runs: 0.024, 0.038, 0.032)*
- `acc_E` = **0.0298** ± 0.0019  *(runs: 0.029, 0.028, 0.032)*
- `combination_is_best` = [False, False, False]
- `gap_over_best_other` = **-0.0083** ± 0.0052  *(runs: -0.010, -0.013, -0.003)*
- `gap_over_lstm_baseline` = **-0.0050** ± 0.0068  *(runs: -0.003, -0.013, 0.001)*

**Notes:** A(LSTM+task)=0.0325  B(FF+task)=0.0394  C(FF+oracle)=0.0319  D(LSTM+oracle)=0.0244  E(FF+hindsight)=0.0294. gap_over_best=-0.0100  combination_is_best=False.

---

### Category 22 — Read Architecture Redesigns (Phase 4)
*0 supported / 4 refuted / 1 inconclusive / 0 error*

#### exp_22_1  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Slot-conditioned read (soft attention over slots -> query refinement) reduces read error by >5% vs fixed linear-projection query on multi-fact retrieval (4 KV pairs, random baseline <10%).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 41s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2811** ± 0.0243  *(runs: 0.294, 0.296, 0.253)*
- `acc_B` = **0.2633** ± 0.0085  *(runs: 0.258, 0.259, 0.273)*
- `gap` = **-0.0177** ± 0.0327  *(runs: -0.036, -0.037, 0.020)*

**Notes:** Standard read outperforms slot-conditioned by 0.036.

---
#### exp_22_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Iterative message-passing read (2 rounds of slot→query→slot attention refinement) outperforms single-pass read by >3% on multi-fact retrieval.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 41s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2811** ± 0.0243  *(runs: 0.294, 0.296, 0.253)*
- `acc_B` = **0.2583** ± 0.0129  *(runs: 0.246, 0.257, 0.272)*
- `gap` = **-0.0227** ± 0.0361  *(runs: -0.048, -0.039, 0.019)*

**Notes:** Single-pass outperforms iterative by 0.048.

---
#### exp_22_3  ~ INCONCLUSIVE
**Hypothesis:** Orthogonal slot initialization via Gram-Schmidt plus orthogonality regularization prevents slot collapse and improves read accuracy by >5% without changing the read mechanism.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 35s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2811** ± 0.0243  *(runs: 0.294, 0.296, 0.253)*
- `acc_B` = **0.2882** ± 0.0098  *(runs: 0.299, 0.286, 0.279)*
- `collapse_A` = **0.2582** ± 0.0860  *(runs: 0.199, 0.218, 0.357)*
- `collapse_B` = **0.1459** ± 0.0119  *(runs: 0.135, 0.158, 0.145)*
- `collapse_reduction` = **0.1123** ± 0.0863  *(runs: 0.065, 0.060, 0.212)*
- `gap` = **0.0071** ± 0.0182  *(runs: 0.005, -0.010, 0.026)*

**Notes:** Gap=0.005, collapse_reduction=0.065.

---
#### exp_22_4  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Read gating (suppress low-confidence reads via entropy threshold) transfers from simple associative recall to multi-pair retrieval without retuning: accuracy gap is <2% on simple and >3% on hard task.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 52s

**Metrics (mean ± std across seeds):**

- `acc_gate_easy` = **0.2721** ± 0.1857  *(runs: 0.058, 0.367, 0.391)*
- `acc_gate_hard` = **0.1190** ± 0.0903  *(runs: 0.144, 0.019, 0.194)*
- `acc_std_easy` = **0.4866** ± 0.0710  *(runs: 0.534, 0.521, 0.405)*
- `acc_std_hard` = **0.1962** ± 0.0027  *(runs: 0.198, 0.193, 0.198)*
- `entropy_threshold` = **1.7675**  *(stable across seeds)*
- `gap_easy` = **0.2146** ± 0.2365  *(runs: 0.476, 0.154, 0.014)*
- `gap_hard` = **-0.0773** ± 0.0875  *(runs: -0.053, -0.174, -0.004)*

**Notes:** Easy gap=0.476 too large or hard benefit negative (-0.053).

---
#### exp_22_5  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Contrastive slot training (InfoNCE-style loss pushing slot embeddings apart) improves retrieval accuracy by >5% on high-interference tasks (many similar keys).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 20s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2854** ± 0.0100  *(runs: 0.286, 0.295, 0.275)*
- `acc_B` = **0.1775** ± 0.1235  *(runs: 0.037, 0.227, 0.269)*
- `gap` = **-0.1079** ± 0.1264  *(runs: -0.249, -0.068, -0.006)*
- `random_baseline` = **0.0312**  *(stable across seeds)*

**Notes:** Standard outperforms contrastive by 0.249.

---

### Category 23 — Retroactive Re-Encoding Variants (Phase 4)
*2 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_23_1  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE']
**Hypothesis:** Multi-head re-encoding (MHA num_heads=4 over slots) outperforms single-head cross-attention re-encoding by >3% due to richer slot-context interaction.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 36s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2627** ± 0.0144  *(runs: 0.276, 0.247, 0.264)*
- `acc_B` = **0.2877** ± 0.0035  *(runs: 0.291, 0.284, 0.287)*
- `gap` = **0.0250** ± 0.0111  *(runs: 0.015, 0.037, 0.023)*

**Notes:** Gap=0.015, between -0.02 and +0.03.

---
#### exp_23_2  ✓ SUPPORTED ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'SUPPORTED']
**Hypothesis:** Two-pass re-encoding yields diminishing returns: the second re-encoding pass provides <20% of the accuracy gain from the first pass.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 52s

**Metrics (mean ± std across seeds):**

- `acc_0pass` = **0.2008** ± 0.0441  *(runs: 0.233, 0.151, 0.219)*
- `acc_1pass` = **0.2729** ± 0.0192  *(runs: 0.263, 0.261, 0.295)*
- `acc_2pass` = **0.3004** ± 0.0040  *(runs: 0.305, 0.299, 0.297)*
- `diminishment_ratio` = **0.5917** ± 0.7138  *(runs: 1.396, 0.347, 0.033)*
- `gain_pass1` = **0.0721** ± 0.0402  *(runs: 0.030, 0.110, 0.076)*
- `gain_pass2` = **0.0275** ± 0.0217  *(runs: 0.042, 0.038, 0.003)*

**Notes:** No diminishing returns: ratio=1.396>0.80.

---
#### exp_23_3  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'SUPPORTED', 'INCONCLUSIVE']
**Hypothesis:** Selective re-encoding (re-encode only slots with cosine distance > T from context) achieves >90% of full re-encoding accuracy at <60% re-encode rate.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 34s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.2008** ± 0.0441  *(runs: 0.233, 0.151, 0.219)*
- `acc_B` = **0.1565** ± 0.1192  *(runs: 0.028, 0.179, 0.263)*
- `acc_C` = **0.2502** ± 0.0190  *(runs: 0.271, 0.247, 0.233)*
- `acc_ratio` = **1.0881** ± 0.2558  *(runs: 1.000, 1.376, 0.888)*
- `reenc_rate` = **0.0000**  *(stable across seeds)*

**Notes:** Selective: acc_ratio=1.000>0.90 at reenc_rate=0.000<0.60.

---
#### exp_23_4  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'SUPPORTED']
**Hypothesis:** Re-encoding gain is task-type specific: factual recall tasks show >2x the accuracy benefit of pattern-completion tasks from cross-attention re-encoding.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 37s

**Metrics (mean ± std across seeds):**

- `acc_factual_base` = **0.2008** ± 0.0441  *(runs: 0.233, 0.151, 0.219)*
- `acc_factual_reenc` = **0.1565** ± 0.1192  *(runs: 0.028, 0.179, 0.263)*
- `acc_pattern_base` = **0.3550** ± 0.0066  *(runs: 0.349, 0.354, 0.362)*
- `acc_pattern_reenc` = **0.3925** ± 0.0299  *(runs: 0.373, 0.427, 0.378)*
- `gain_factual` = **-0.0444** ± 0.1398  *(runs: -0.206, 0.029, 0.044)*
- `gain_pattern` = **0.0375** ± 0.0306  *(runs: 0.024, 0.072, 0.016)*
- `specificity_ratio` = **-1.7464** ± 5.9166  *(runs: -8.436, 0.397, 2.800)*

**Notes:** No task-type specificity: ratio=-8.436<0.5.

---

### Category 24 — Scale and Length Generalization (Phase 4)
*1 supported / 2 refuted / 1 inconclusive / 0 error*

#### exp_24_1  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Parametric memory retains >80% accuracy at 4x training sequence length, while slot memory drops below 40%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 46s

**Metrics (mean ± std across seeds):**

- `acc_parametric.24` = **0.0354** ± 0.0110  *(runs: 0.025, 0.047, 0.034)*
- `acc_parametric.48` = **0.0291** ± 0.0130  *(runs: 0.025, 0.019, 0.044)*
- `acc_parametric.96` = **0.0406** ± 0.0083  *(runs: 0.047, 0.044, 0.031)*
- `acc_slot.24` = **0.0844** ± 0.0109  *(runs: 0.078, 0.078, 0.097)*
- `acc_slot.48` = **0.0323** ± 0.0148  *(runs: 0.044, 0.016, 0.037)*
- `acc_slot.96` = **0.0312** ± 0.0113  *(runs: 0.019, 0.034, 0.041)*
- `retention_param_4x` = **1.2383** ± 0.5524  *(runs: 1.876, 0.932, 0.907)*
- `retention_slot_4x` = **0.3663** ± 0.1104  *(runs: 0.239, 0.441, 0.419)*

**Notes:** Param retention_4x=1.876, slot acc_96=0.019. Threshold not fully met.

---
#### exp_24_2  ✓ SUPPORTED
**Hypothesis:** Compositional two-hop retrieval sustains >70% accuracy under 60% entity interference, up from 40% interference tested in exp_13_1.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 8s

**Metrics (mean ± std across seeds):**

- `retention_at_60pct_interf` = **0.7887**  *(stable across seeds)*
- `two_hop_acc_by_interference.0.0` = **1.0000**  *(stable across seeds)*
- `two_hop_acc_by_interference.0.4` = **0.8538**  *(stable across seeds)*
- `two_hop_acc_by_interference.0.6` = **0.7887**  *(stable across seeds)*

**Notes:** Two-hop acc at 60% interference=0.789>0.70.

---
#### exp_24_3  ✗ REFUTED
**Hypothesis:** Energy-gated delta rule achieves the same accuracy-to-write-rate ratio at HIDDEN_DIM=128 as at HIDDEN_DIM=32 (within 5%) — the mechanism is dimension-robust.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 88s

**Metrics (mean ± std across seeds):**

- `max_ratio` = **0.4132** ± 0.0087  *(runs: 0.418, 0.403, 0.418)*
- `min_ratio` = **0.2504** ± 0.0467  *(runs: 0.257, 0.201, 0.294)*
- `ratio_spread` = **0.3952** ± 0.1024  *(runs: 0.387, 0.501, 0.297)*
- `results_by_dim.128.acc` = **0.2112** ± 0.0043  *(runs: 0.214, 0.206, 0.214)*
- `results_by_dim.128.acc_wr_ratio` = **0.4132** ± 0.0087  *(runs: 0.418, 0.403, 0.418)*
- `results_by_dim.128.write_rate` = **0.5113** ± 0.0003  *(runs: 0.511, 0.512, 0.511)*
- `results_by_dim.32.acc` = **0.1292** ± 0.0235  *(runs: 0.132, 0.104, 0.151)*
- `results_by_dim.32.acc_wr_ratio` = **0.2504** ± 0.0467  *(runs: 0.257, 0.201, 0.294)*
- `results_by_dim.32.write_rate` = **0.5161** ± 0.0028  *(runs: 0.514, 0.519, 0.515)*
- `results_by_dim.64.acc` = **0.1981** ± 0.0166  *(runs: 0.204, 0.179, 0.211)*
- `results_by_dim.64.acc_wr_ratio` = **0.3871** ± 0.0329  *(runs: 0.397, 0.350, 0.414)*
- `results_by_dim.64.write_rate` = **0.5119** ± 0.0012  *(runs: 0.513, 0.512, 0.511)*

**Notes:** Ratio spread=0.387>0.15. Mechanism is dim-sensitive.

---
#### exp_24_4  ✗ REFUTED
**Hypothesis:** Four-hop compositional chains are infeasible at HIDDEN_DIM=64: accuracy drops >50% vs two-hop even with hop-by-hop training curriculum.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 11s

**Metrics (mean ± std across seeds):**

- `acc_2hop` = **1.0000**  *(stable across seeds)*
- `acc_4hop` = **0.8131**  *(stable across seeds)*
- `acc_drop_fraction` = **0.1869**  *(stable across seeds)*
- `random_baseline` = **0.0625**  *(stable across seeds)*

**Notes:** 4-hop achievable: drop=18.7%<20% vs 2-hop.

---

### Category 25 — Hard Benchmarks (Phase 4)
*1 supported / 2 refuted / 0 inconclusive / 0 error*

#### exp_25_1  ✓ SUPPORTED
**Hypothesis:** Multi-domain retrieval benchmark (facts + patterns + temporal chains in same sequence): any memory architecture achieves <70% joint accuracy without domain-specific slots.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 60s

**Metrics (mean ± std across seeds):**

- `acc_domain_specific` = **0.0635** ± 0.0256  *(runs: 0.048, 0.093, 0.049)*
- `acc_generic` = **0.1433** ± 0.0058  *(runs: 0.145, 0.148, 0.137)*
- `generic_below_70pct` = [True, True, True]
- `random_baseline` = **0.0312**  *(stable across seeds)*

**Notes:** Generic acc=0.145<0.70. Domain-specific: 0.048.

---
#### exp_25_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** Noisy-key retrieval (Gaussian noise added to query at test time): slot memory degrades >20% while parametric memory degrades <10%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 9s

**Metrics (mean ± std across seeds):**

- `param_accs_by_sigma.0.0` = **0.0333** ± 0.0095  *(runs: 0.031, 0.025, 0.044)*
- `param_accs_by_sigma.0.1` = **0.0302** ± 0.0090  *(runs: 0.041, 0.025, 0.025)*
- `param_accs_by_sigma.0.2` = **0.0250** ± 0.0031  *(runs: 0.022, 0.028, 0.025)*
- `param_deg_at_sigma01` = **0.0422** ± 0.3664  *(runs: -0.301, 0.000, 0.428)*
- `slot_accs_by_sigma.0.0` = **0.0427** ± 0.0119  *(runs: 0.037, 0.056, 0.034)*
- `slot_accs_by_sigma.0.1` = **0.0364** ± 0.0208  *(runs: 0.059, 0.019, 0.031)*
- `slot_accs_by_sigma.0.2` = **0.0385** ± 0.0243  *(runs: 0.066, 0.019, 0.031)*
- `slot_deg_at_sigma01` = **0.0590** ± 0.6266  *(runs: -0.584, 0.668, 0.093)*

**Notes:** Slot is noise-robust (deg=-0.584) — no architecture gap.

---
#### exp_25_3  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Temporal ordering task (retrieve k-th event in temporal sequence): accuracy drops monotonically with k, revealing the read bottleneck for ordered memory access.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 22s

**Metrics (mean ± std across seeds):**

- `acc_by_k.1` = **0.1313** ± 0.1120  *(runs: 0.143, 0.237, 0.014)*
- `acc_by_k.2` = **0.1032** ± 0.0424  *(runs: 0.139, 0.114, 0.056)*
- `acc_by_k.3` = **0.1171** ± 0.0882  *(runs: 0.183, 0.151, 0.017)*
- `acc_by_k.4` = **0.1171** ± 0.0362  *(runs: 0.158, 0.106, 0.087)*
- `acc_by_k.5` = **0.1306** ± 0.1103  *(runs: 0.156, 0.226, 0.010)*
- `acc_by_k.6` = **0.1192** ± 0.0646  *(runs: 0.193, 0.071, 0.094)*
- `first_to_last_drop` = **0.0121** ± 0.1344  *(runs: -0.049, 0.166, -0.081)*
- `k1_acc` = **0.1313** ± 0.1120  *(runs: 0.143, 0.237, 0.014)*
- `k_max_acc` = **0.1192** ± 0.0646  *(runs: 0.193, 0.071, 0.094)*
- `monotone_decreasing` = [False, False, False]

**Notes:** Temporal ordering is nearly flat (drop=-0.049<0.05).

---

### Category 26 — Seed Stability Validation (Phase 4)
*1 supported / 2 refuted / 0 inconclusive / 0 error*

#### exp_26_1  ✓ SUPPORTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED']
**Hypothesis:** Protected slot interior optimum (exp_9_4) is seed-stable: 3 additional seeds confirm an interior peak at K=3-6 with MEMORY_SLOTS=12.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 44s

**Metrics (mean ± std across seeds):**

- `acc_range` = **0.1490** ± 0.0030  *(runs: 0.150, 0.146, 0.151)*
- `accuracies` = [[0.1519, 0.1437, 0.0225, 0.0469, 0.105, 0.1725], [0.0381, 0.1575, 0.1031, 0.1837, 0.1062, 0.1638], [0.0294, 0.0306, 0.0912, 0.0725, 0.1806, 0.1469]]
- `critical_accuracies` = [[0.0328, 0.2215, 0.02, 0.0583, 0.0959, 0.1686], [0.0279, 0.1462, 0.1161, 0.19, 0.1144, 0.1644], [0.0267, 0.036, 0.112, 0.0818, 0.1821, 0.1317]]
- `interior_peak_exists` = [False, True, True]
- `k_opt` = **4.0000** ± 1.0000  *(runs: 5.000, 3.000, 4.000)*
- `max_acc` = **0.1790** ± 0.0058  *(runs: 0.172, 0.184, 0.181)*
- `min_acc` = **0.0300** ± 0.0078  *(runs: 0.022, 0.038, 0.029)*
- `noncritical_accuracies` = [[0.3345, 0.0189, 0.0239, 0.0314, 0.1157, 0.1692], [0.0571, 0.1701, 0.0844, 0.1779, 0.0963, 0.1652], [0.0325, 0.0237, 0.0591, 0.0562, 0.1758, 0.1691]]

**Notes:** K values [0, 2, 4, 6, 8, 10], accs [0.152, 0.144, 0.022, 0.047, 0.105, 0.172]. Optimal K=5, max_acc=0.172. Interior peak: False.

---
#### exp_26_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Write budget with positionally biased task (exp_9_5) is seed-stable: additional seeds confirm non-uniform (oracle) budget outperforms uniform.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 58s

**Metrics (mean ± std across seeds):**

- `acc_learned` = **0.1742** ± 0.0940  *(runs: 0.231, 0.066, 0.226)*
- `acc_oracle` = **0.2485** ± 0.0296  *(runs: 0.255, 0.274, 0.216)*
- `acc_uniform` = **0.2596** ± 0.0043  *(runs: 0.256, 0.264, 0.258)*
- `uniform_to_learned_gap` = **-0.0854** ± 0.0982  *(runs: -0.026, -0.199, -0.032)*
- `uniform_to_oracle_gap` = **-0.0110** ± 0.0273  *(runs: -0.001, 0.010, -0.042)*

**Notes:** Uniform=0.256, Oracle=0.255, Learned=0.231. Oracle gap=-0.001, Learned gap=-0.026.

---
#### exp_26_3  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Query-conditioned write gate (exp_17_1) improvement over context-only gate is seed-stable: additional seeds confirm >5% accuracy improvement on multi-query-type tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 9s

**Metrics (mean ± std across seeds):**

- `acc_A` = **0.1398** ± 0.0812  *(runs: 0.221, 0.059, 0.140)*
- `acc_B` = **0.0762** ± 0.0774  *(runs: 0.033, 0.030, 0.166)*
- `gap` = **-0.0636** ± 0.1111  *(runs: -0.188, -0.028, 0.026)*
- `query_pred_acc` = **0.7209** ± 0.1728  *(runs: 0.857, 0.526, 0.780)*

**Notes:** Context-only=0.221, QueryCond=0.033, gap=-0.188, qtype_pred=0.857.

---

### Category 27 — Parametric-Delta Hybrid (Phase 4)
*0 supported / 2 refuted / 1 inconclusive / 0 error*

#### exp_27_1  ✗ REFUTED
**Hypothesis:** Hybrid memory (delta-rule matrix + parametric MLP) outperforms either alone when each component is pre-trained independently for 100 steps before joint fine-tuning.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 32s

**Metrics (mean ± std across seeds):**

- `acc_cold_joint` = **0.1208** ± 0.0273  *(runs: 0.117, 0.150, 0.096)*
- `acc_delta_only` = **0.0819** ± 0.0064  *(runs: 0.075, 0.083, 0.087)*
- `acc_param_only` = **0.0236** ± 0.0134  *(runs: 0.008, 0.033, 0.029)*
- `acc_pretrained_hybrid` = **0.0444** ± 0.0105  *(runs: 0.033, 0.046, 0.054)*
- `superadditive_ratio` = **0.5378** ± 0.0879  *(runs: 0.444, 0.550, 0.619)*

**Notes:** Delta=0.075, Param=0.008, Cold=0.117, PretrainedHybrid=0.033, ratio=0.444.

---
#### exp_27_2  ~ INCONCLUSIVE
**Hypothesis:** In the hybrid model, the delta-rule component specializes in short-range retrieval while the parametric component specializes in long-range retrieval: delta_short > param_short + 5% AND param_long > delta_long + 5%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 12s

**Metrics (mean ± std across seeds):**

- `delta_long` = **0.1062** ± 0.0165  *(runs: 0.087, 0.113, 0.119)*
- `delta_short` = **0.1000** ± 0.0108  *(runs: 0.087, 0.106, 0.106)*
- `long_gap` = **-0.0812** ± 0.0109  *(runs: -0.069, -0.087, -0.087)*
- `param_long` = **0.0250** ± 0.0063  *(runs: 0.019, 0.025, 0.031)*
- `param_short` = **0.0271** ± 0.0130  *(runs: 0.031, 0.013, 0.037)*
- `short_gap` = **0.0729** ± 0.0191  *(runs: 0.056, 0.094, 0.069)*

**Notes:** Delta: short=0.087, long=0.087. Param: short=0.031, long=0.019. short_gap=0.056, long_gap=-0.069.

---
#### exp_27_3  ✗ REFUTED
**Hypothesis:** Sequential pre-training isolation (delta pre-train → parametric pre-train → joint fine-tune) achieves super-additive accuracy >1.1× best single component AND outperforms cold joint training.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 32s

**Metrics (mean ± std across seeds):**

- `acc_cold_joint` = **0.1208** ± 0.0273  *(runs: 0.117, 0.150, 0.096)*
- `acc_delta_only` = **0.0819** ± 0.0064  *(runs: 0.075, 0.083, 0.087)*
- `acc_param_only` = **0.0236** ± 0.0134  *(runs: 0.008, 0.033, 0.029)*
- `acc_pretrain_sequential` = **0.0431** ± 0.0064  *(runs: 0.037, 0.042, 0.050)*
- `loss_var_cold` = **0.0953** ± 0.0169  *(runs: 0.111, 0.077, 0.098)*
- `loss_var_sequential` = **0.1050** ± 0.0065  *(runs: 0.102, 0.101, 0.112)*
- `stability_ratio` = **0.9083** ± 0.1628  *(runs: 1.087, 0.768, 0.870)*
- `superadditive_ratio` = **0.5238** ± 0.0412  *(runs: 0.500, 0.500, 0.571)*

**Notes:** A=0.075, B=0.008, C(cold)=0.117, D(seq)=0.037, superadd=0.500, stability_ratio=1.087.

---

### Category 28 — Explicit Scaling Laws (Phase 5)
*0 supported / 1 refuted / 4 inconclusive / 0 error*

#### exp_28_1  ~ INCONCLUSIVE
**Hypothesis:** Parametric memory retains >90% accuracy at SEQ_LEN=192 while slot memory drops below 30%, confirming a length-scaling crossover.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 531s

**Metrics (mean ± std across seeds):**

- `acc_delta_len192` = **0.1805** ± 0.0262  *(runs: 0.177, 0.208, 0.156)*
- `acc_delta_len24` = **0.1556** ± 0.0136  *(runs: 0.156, 0.142, 0.169)*
- `acc_delta_len48` = **0.1514** ± 0.0288  *(runs: 0.163, 0.119, 0.173)*
- `acc_delta_len96` = **0.1500** ± 0.0211  *(runs: 0.173, 0.131, 0.146)*
- `acc_param_len192` = **0.0333** ± 0.0036  *(runs: 0.029, 0.035, 0.035)*
- `acc_param_len24` = **0.0347** ± 0.0084  *(runs: 0.040, 0.040, 0.025)*
- `acc_param_len48` = **0.0319** ± 0.0073  *(runs: 0.031, 0.025, 0.040)*
- `acc_param_len96` = **0.0305** ± 0.0052  *(runs: 0.031, 0.035, 0.025)*
- `acc_slot_len192` = **0.0792** ± 0.0758  *(runs: 0.035, 0.167, 0.035)*
- `acc_slot_len24` = **0.0701** ± 0.0766  *(runs: 0.021, 0.158, 0.031)*
- `acc_slot_len48` = **0.0993** ± 0.0576  *(runs: 0.140, 0.033, 0.125)*
- `acc_slot_len96` = **0.0840** ± 0.1058  *(runs: 0.021, 0.206, 0.025)*
- `param_retention_8x` = **1.0158** ± 0.3553  *(runs: 0.737, 0.894, 1.416)*
- `slot_retention_8x` = **1.2965** ± 0.3534  *(runs: 1.702, 1.053, 1.135)*

**Notes:** Slot retention 8×=1.702, param retention 8×=0.737. Slot@192=0.035, Param@192=0.029.

---
#### exp_28_2  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'INCONCLUSIVE']
**Hypothesis:** Accuracy scales as dim^α with α > 0.3 for delta rule, R² > 0.95 log-log fit over HIDDEN_DIM ∈ {32,64,128,256}.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 381s

**Metrics (mean ± std across seeds):**

- `acc_delta_H128` = **0.2175** ± 0.0145  *(runs: 0.202, 0.219, 0.231)*
- `acc_delta_H256` = **0.2125** ± 0.0109  *(runs: 0.205, 0.207, 0.225)*
- `acc_delta_H32` = **0.1323** ± 0.0142  *(runs: 0.116, 0.144, 0.137)*
- `acc_delta_H64` = **0.2089** ± 0.0199  *(runs: 0.229, 0.189, 0.209)*
- `acc_param_H128` = **0.0294** ± 0.0012  *(runs: 0.030, 0.030, 0.028)*
- `acc_param_H256` = **0.0310** ± 0.0052  *(runs: 0.030, 0.027, 0.037)*
- `acc_param_H32` = **0.0292** ± 0.0059  *(runs: 0.027, 0.025, 0.036)*
- `acc_param_H64` = **0.0310** ± 0.0023  *(runs: 0.034, 0.030, 0.030)*
- `acc_slot_H128` = **0.0320** ± 0.0051  *(runs: 0.037, 0.031, 0.027)*
- `acc_slot_H256` = **0.1036** ± 0.1193  *(runs: 0.037, 0.241, 0.032)*
- `acc_slot_H32` = **0.1279** ± 0.0885  *(runs: 0.166, 0.191, 0.027)*
- `acc_slot_H64` = **0.0739** ± 0.0533  *(runs: 0.031, 0.134, 0.057)*
- `alpha_delta` = **0.2126** ± 0.0291  *(runs: 0.228, 0.179, 0.231)*
- `alpha_param` = **0.0207** ± 0.0170  *(runs: 0.034, 0.027, 0.002)*
- `alpha_slot` = **-0.2507** ± 0.3210  *(runs: -0.618, -0.108, -0.026)*
- `r2_delta` = **0.6318** ± 0.1585  *(runs: 0.450, 0.737, 0.709)*
- `r2_param` = **0.0594** ± 0.0523  *(runs: 0.099, 0.079, 0.000)*
- `r2_slot` = **0.1727** ± 0.2857  *(runs: 0.503, 0.011, 0.004)*

**Notes:** Delta: α=0.228, R²=0.450. Slot: α=-0.618, R²=0.503. Param: α=0.034, R²=0.099.

---
#### exp_28_3  ~ INCONCLUSIVE
**Hypothesis:** Parametric memory has the steepest per-step accuracy gain across STEPS ∈ {200,400,800,1600,3200}, confirming highest sample efficiency.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 607s

**Metrics (mean ± std across seeds):**

- `acc_delta_s1600` = **0.2295** ± 0.0026  *(runs: 0.229, 0.232, 0.227)*
- `acc_delta_s200` = **0.1694** ± 0.0310  *(runs: 0.205, 0.152, 0.151)*
- `acc_delta_s3200` = **0.2305** ± 0.0147  *(runs: 0.244, 0.233, 0.215)*
- `acc_delta_s400` = **0.2087** ± 0.0176  *(runs: 0.228, 0.204, 0.194)*
- `acc_delta_s800` = **0.2198** ± 0.0075  *(runs: 0.218, 0.213, 0.228)*
- `acc_param_s1600` = **0.0305** ± 0.0026  *(runs: 0.030, 0.028, 0.033)*
- `acc_param_s200` = **0.0403** ± 0.0090  *(runs: 0.050, 0.032, 0.038)*
- `acc_param_s3200` = **0.0326** ± 0.0051  *(runs: 0.030, 0.029, 0.038)*
- `acc_param_s400` = **0.0337** ± 0.0032  *(runs: 0.036, 0.030, 0.034)*
- `acc_param_s800` = **0.0368** ± 0.0069  *(runs: 0.045, 0.033, 0.032)*
- `acc_slot_s1600` = **0.0472** ± 0.0233  *(runs: 0.031, 0.074, 0.036)*
- `acc_slot_s200` = **0.0864** ± 0.0879  *(runs: 0.028, 0.188, 0.044)*
- `acc_slot_s3200` = **0.0347** ± 0.0217  *(runs: 0.030, 0.058, 0.016)*
- `acc_slot_s400` = **0.0837** ± 0.0899  *(runs: 0.033, 0.188, 0.030)*
- `acc_slot_s800` = **0.0969** ± 0.1092  *(runs: 0.035, 0.223, 0.032)*
- `slope_delta` = **0.0145** ± 0.0049  *(runs: 0.010, 0.020, 0.014)*
- `slope_param` = **-0.0019** ± 0.0032  *(runs: -0.005, -0.001, 0.001)*
- `slope_slot` = **-0.0198** ± 0.0281  *(runs: -0.000, -0.052, -0.007)*

**Notes:** Slopes (×10³/step): slot=-0.0004, delta=0.0100, param=-0.0054.

---
#### exp_28_4  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Slot accuracy peaks at NUM_SLOTS=1.5–2×NUM_PAIRS (9–12 slots), then degrades with excess slots due to slot collapse.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 48s

**Metrics (mean ± std across seeds):**

- `acc_k12` = **0.1302** ± 0.0402  *(runs: 0.087, 0.136, 0.167)*
- `acc_k16` = **0.1625** ± 0.0131  *(runs: 0.174, 0.165, 0.148)*
- `acc_k2` = **0.0682** ± 0.0439  *(runs: 0.039, 0.047, 0.119)*
- `acc_k24` = **0.1573** ± 0.0102  *(runs: 0.148, 0.156, 0.168)*
- `acc_k24_vs_peak` = **-0.0148** ± 0.0135  *(runs: -0.026, -0.018, 0.000)*
- `acc_k4` = **0.1185** ± 0.0469  *(runs: 0.165, 0.071, 0.119)*
- `acc_k6` = **0.1200** ± 0.0832  *(runs: 0.024, 0.174, 0.162)*
- `acc_k8` = **0.0984** ± 0.0597  *(runs: 0.106, 0.035, 0.154)*
- `acc_peak` = **0.1721** ± 0.0036  *(runs: 0.174, 0.174, 0.168)*
- `collapse_k12` = **0.6323** ± 0.0959  *(runs: 0.741, 0.558, 0.599)*
- `collapse_k16` = **0.6238** ± 0.0751  *(runs: 0.624, 0.699, 0.548)*
- `collapse_k2` = **0.8207** ± 0.1102  *(runs: 0.938, 0.806, 0.719)*
- `collapse_k24` = **0.4390** ± 0.0588  *(runs: 0.433, 0.500, 0.383)*
- `collapse_k4` = **0.6444** ± 0.0760  *(runs: 0.561, 0.709, 0.664)*
- `collapse_k6` = **0.7082** ± 0.2581  *(runs: 0.999, 0.506, 0.619)*
- `collapse_k8` = **0.7613** ± 0.2424  *(runs: 0.770, 0.999, 0.515)*
- `peak_k` = **15.3333** ± 9.0185  *(runs: 16.000, 6.000, 24.000)*

**Notes:** Peak at k=16 (acc=0.174). k=24 acc=0.148 (drop=0.026).

---
#### exp_28_5  ✗ REFUTED
**Hypothesis:** Delta rule accuracy variance across VOCAB_SIZE ∈ {32,64,128,256} is <2% at each NUM_PAIRS level, confirming vocab-independence of capacity.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 279s

**Metrics (mean ± std across seeds):**

- `acc_mean_n2` = **0.4804** ± 0.0089  *(runs: 0.482, 0.471, 0.489)*
- `acc_mean_n4` = **0.2219** ± 0.0022  *(runs: 0.224, 0.223, 0.220)*
- `acc_mean_n8` = **0.0843** ± 0.0108  *(runs: 0.092, 0.088, 0.072)*
- `acc_n2_v128` = **0.4849** ± 0.0159  *(runs: 0.474, 0.503, 0.477)*
- `acc_n2_v256` = **0.3792** ± 0.0246  *(runs: 0.393, 0.351, 0.394)*
- `acc_n2_v32` = **0.5380** ± 0.0134  *(runs: 0.536, 0.526, 0.552)*
- `acc_n2_v64` = **0.5195** ± 0.0141  *(runs: 0.523, 0.504, 0.531)*
- `acc_n4_v128` = **0.2224** ± 0.0133  *(runs: 0.237, 0.212, 0.217)*
- `acc_n4_v256` = **0.1263** ± 0.0162  *(runs: 0.113, 0.121, 0.144)*
- `acc_n4_v32` = **0.2794** ± 0.0053  *(runs: 0.284, 0.281, 0.273)*
- `acc_n4_v64` = **0.2596** ± 0.0160  *(runs: 0.261, 0.275, 0.243)*
- `acc_n8_v128` = **0.0607** ± 0.0167  *(runs: 0.071, 0.070, 0.041)*
- `acc_n8_v256` = **0.0180** ± 0.0051  *(runs: 0.013, 0.023, 0.019)*
- `acc_n8_v32` = **0.1497** ± 0.0097  *(runs: 0.161, 0.144, 0.144)*
- `acc_n8_v64` = **0.1089** ± 0.0215  *(runs: 0.125, 0.117, 0.084)*
- `acc_variance_n2` = **0.1588** ± 0.0161  *(runs: 0.143, 0.175, 0.159)*
- `acc_variance_n4` = **0.1531** ± 0.0216  *(runs: 0.170, 0.160, 0.129)*
- `acc_variance_n8` = **0.1317** ± 0.0145  *(runs: 0.148, 0.122, 0.125)*
- `max_variance` = **0.1680** ± 0.0084  *(runs: 0.170, 0.175, 0.159)*

**Notes:** Max variance across vocab sizes = 0.1703 (>0.02). Vocab-independence NOT confirmed.

---

### Category 29 — TTT / Titans-Inspired Memory (Phase 5)
*2 supported / 0 refuted / 2 inconclusive / 0 error*

#### exp_29_1  ✓ SUPPORTED
**Hypothesis:** Outer-product linear associative memory (no test-time SGD) matches slot memory within 2% at matched parameter count.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 79s

**Metrics (mean ± std across seeds):**

- `acc_linear` = **0.2533** ± 0.0176  *(runs: 0.261, 0.233, 0.266)*
- `acc_slot` = **0.1369** ± 0.1018  *(runs: 0.028, 0.154, 0.229)*
- `gap` = **0.1164** ± 0.1037  *(runs: 0.234, 0.079, 0.037)*

**Notes:** Linear acc=0.261, slot acc=0.028, gap=+0.234.

---
#### exp_29_2  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Adam-at-inference for parametric memory MLP improves accuracy by >5% over SGD at the same number of inner steps (INFERENCE_STEPS=3).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 212s

**Metrics (mean ± std across seeds):**

- `acc_adam` = **0.0757** ± 0.0198  *(runs: 0.096, 0.056, 0.075)*
- `acc_sgd` = **0.0722** ± 0.0105  *(runs: 0.062, 0.083, 0.071)*
- `gap` = **0.0035** ± 0.0302  *(runs: 0.033, -0.027, 0.004)*

**Notes:** Adam acc=0.096, SGD acc=0.062, gap=+0.033.

---
#### exp_29_3  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** Surprise-gated TTT (update when gradient-norm ratio > 1.5) achieves >90% of full-update accuracy at <50% update rate.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 69s

**Metrics (mean ± std across seeds):**

- `acc_full` = **0.0312** ± 0.0150  *(runs: 0.019, 0.027, 0.048)*
- `acc_no_ttt` = **0.0306** ± 0.0043  *(runs: 0.029, 0.035, 0.027)*
- `acc_ratio` = **1.0709** ± 0.6743  *(runs: 1.778, 1.000, 0.435)*
- `acc_surprise` = **0.0271** ± 0.0063  *(runs: 0.033, 0.027, 0.021)*
- `write_rate` = **0.0947** ± 0.0132  *(runs: 0.102, 0.080, 0.102)*

**Notes:** Full acc=0.019, Surprise acc=0.033, No-TTT acc=0.029. acc_ratio=1.778, write_rate=0.102.

---
#### exp_29_4  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Weight decay at inference prevents saturation: at SEQ_LEN=192, accuracy with weight_decay=0.01 is >20% higher than without.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 468s

**Metrics (mean ± std across seeds):**

- `acc_gain_wd_192` = **-0.0021** ± 0.0065  *(runs: 0.003, 0.000, -0.009)*
- `acc_plain_len192` = **0.0312** ± 0.0032  *(runs: 0.031, 0.028, 0.034)*
- `acc_plain_len24` = **0.0385** ± 0.0047  *(runs: 0.037, 0.034, 0.044)*
- `acc_plain_len96` = **0.0281** ± 0.0094  *(runs: 0.019, 0.037, 0.028)*
- `acc_wd_len192` = **0.0292** ± 0.0048  *(runs: 0.034, 0.028, 0.025)*
- `acc_wd_len24` = **0.0354** ± 0.0101  *(runs: 0.028, 0.031, 0.047)*
- `acc_wd_len96` = **0.0385** ± 0.0079  *(runs: 0.037, 0.031, 0.047)*
- `gap_len192` = **-0.0021** ± 0.0065  *(runs: 0.003, 0.000, -0.009)*
- `gap_len24` = **-0.0031** ± 0.0063  *(runs: -0.009, -0.003, 0.003)*
- `gap_len96` = **0.0104** ± 0.0145  *(runs: 0.019, -0.006, 0.019)*

**Notes:** Gain at SEQ_LEN=192: +0.003. Gaps: len24=-0.009, len96=+0.019, len192=+0.003.

---

### Category 30 — Multi-Head & Extended Delta Rule (Phase 5)
*1 supported / 2 refuted / 1 inconclusive / 0 error*

#### exp_30_1  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'INCONCLUSIVE', 'SUPPORTED']
**Hypothesis:** Multi-head delta rule (4 heads x H/4 dims) outperforms single-head by >5% on 8-pair associative recall.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 419s

**Metrics (mean ± std across seeds):**

- `acc_multi` = **0.2577** ± 0.0972  *(runs: 0.366, 0.179, 0.228)*
- `acc_single` = **0.1465** ± 0.0060  *(runs: 0.152, 0.140, 0.147)*
- `gap` = **0.1112** ± 0.0918  *(runs: 0.214, 0.039, 0.081)*

**Notes:** Multi-head acc=0.366, Single-head acc=0.152, gap=0.214. Params: single=113KB, multi=162KB.

---
#### exp_30_2  ✗ REFUTED
**Hypothesis:** Momentum delta (β=0.9) matches energy-gated accuracy within 5% while achieving lower final-100-step loss variance (ratio < 0.8).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 192s

**Metrics (mean ± std across seeds):**

- `acc_energy` = **0.1835** ± 0.0163  *(runs: 0.171, 0.178, 0.202)*
- `acc_momentum` = **0.2117** ± 0.0153  *(runs: 0.228, 0.197, 0.211)*
- `gap_acc` = **0.0281** ± 0.0254  *(runs: 0.057, 0.019, 0.009)*
- `loss_var_energy` = **0.0255** ± 0.0017  *(runs: 0.025, 0.024, 0.027)*
- `loss_var_momentum` = **0.0302** ± 0.0007  *(runs: 0.030, 0.030, 0.031)*
- `var_ratio` = **1.1900** ± 0.0537  *(runs: 1.202, 1.236, 1.131)*
- `write_rate_energy` = **0.5984** ± 0.0017  *(runs: 0.600, 0.598, 0.597)*
- `write_rate_momentum` = **1.0000**  *(stable across seeds)*

**Notes:** Momentum acc=0.228, Energy acc=0.171, gap=0.057. Var ratio=1.202. Write rates: energy=0.600, momentum=1.000.

---
#### exp_30_3  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'REFUTED']
**Hypothesis:** Bidirectional delta rule improves accuracy by >8% on late-query tasks vs unidirectional, without harming early-query tasks by more than 2%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 731s

**Metrics (mean ± std across seeds):**

- `acc_bidir_early` = **0.1823** ± 0.0067  *(runs: 0.175, 0.184, 0.188)*
- `acc_bidir_late` = **0.1838** ± 0.0130  *(runs: 0.169, 0.193, 0.190)*
- `acc_unidir_early` = **0.1846** ± 0.0072  *(runs: 0.177, 0.192, 0.184)*
- `acc_unidir_late` = **0.1929** ± 0.0063  *(runs: 0.188, 0.191, 0.200)*
- `early_change` = **-0.0023** ± 0.0060  *(runs: -0.003, -0.008, 0.004)*
- `late_improvement` = **-0.0091** ± 0.0106  *(runs: -0.019, 0.002, -0.010)*

**Notes:** Late: bidir=0.169, unidir=0.188, improvement=-0.019. Early: bidir=0.175, unidir=0.177, change=-0.003.

---
#### exp_30_4  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'INCONCLUSIVE', 'SUPPORTED']
**Hypothesis:** Energy-gated delta rule Pareto knee lies at 40-60% write rate for all tested hidden dimensions (32, 64, 128).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1217s

**Metrics (mean ± std across seeds):**

- `all_knees_in_range` = **0.3333** ± 0.5774  *(runs: 0.000, 0.000, 1.000)*
- `knee_wr_H128` = **0.5903** ± 0.0022  *(runs: 0.593, 0.588, 0.590)*
- `knee_wr_H32` = **0.5448** ± 0.0587  *(runs: 0.612, 0.521, 0.501)*
- `knee_wr_H64` = **0.5972** ± 0.0074  *(runs: 0.599, 0.604, 0.589)*
- `pareto_acc_H128_th0` = **0.1815** ± 0.0036  *(runs: 0.185, 0.181, 0.178)*
- `pareto_acc_H128_th1` = **0.1875** ± 0.0075  *(runs: 0.191, 0.179, 0.193)*
- `pareto_acc_H128_th2` = **0.1810** ± 0.0072  *(runs: 0.186, 0.173, 0.184)*
- `pareto_acc_H128_th3` = **0.1831** ± 0.0087  *(runs: 0.177, 0.193, 0.180)*
- `pareto_acc_H128_th4` = **0.1870** ± 0.0065  *(runs: 0.189, 0.192, 0.180)*
- `pareto_acc_H128_th5` = **0.0271** ± 0.0032  *(runs: 0.029, 0.029, 0.023)*
- `pareto_acc_H32_th0` = **0.1487** ± 0.0030  *(runs: 0.145, 0.151, 0.150)*
- `pareto_acc_H32_th1` = **0.1643** ± 0.0073  *(runs: 0.156, 0.166, 0.170)*
- `pareto_acc_H32_th2` = **0.1612** ± 0.0099  *(runs: 0.150, 0.165, 0.169)*
- `pareto_acc_H32_th3` = **0.1542** ± 0.0246  *(runs: 0.126, 0.169, 0.168)*
- `pareto_acc_H32_th4` = **0.1617** ± 0.0195  *(runs: 0.140, 0.168, 0.177)*
- `pareto_acc_H32_th5` = **0.0294** ± 0.0056  *(runs: 0.030, 0.034, 0.023)*
- `pareto_acc_H64_th0` = **0.1857** ± 0.0135  *(runs: 0.198, 0.171, 0.188)*
- `pareto_acc_H64_th1` = **0.1813** ± 0.0094  *(runs: 0.173, 0.180, 0.191)*
- `pareto_acc_H64_th2` = **0.1964** ± 0.0151  *(runs: 0.205, 0.205, 0.179)*
- `pareto_acc_H64_th3` = **0.1831** ± 0.0111  *(runs: 0.170, 0.191, 0.188)*
- `pareto_acc_H64_th4` = **0.1831** ± 0.0090  *(runs: 0.189, 0.188, 0.173)*
- `pareto_acc_H64_th5` = **0.0365** ± 0.0059  *(runs: 0.030, 0.042, 0.037)*
- `pareto_wr_H128_th0` = **1.0000**  *(stable across seeds)*
- `pareto_wr_H128_th1` = **0.6250** ± 0.0025  *(runs: 0.622, 0.627, 0.625)*
- `pareto_wr_H128_th2` = **0.5915** ± 0.0016  *(runs: 0.593, 0.592, 0.590)*
- `pareto_wr_H128_th3` = **0.5891** ± 0.0008  *(runs: 0.588, 0.590, 0.589)*
- `pareto_wr_H128_th4` = **0.5889** ± 0.0006  *(runs: 0.590, 0.588, 0.589)*
- `pareto_wr_H128_th5` = **0.0000**  *(stable across seeds)*
- `pareto_wr_H32_th0` = **1.0000**  *(stable across seeds)*
- `pareto_wr_H32_th1` = **0.6429** ± 0.0016  *(runs: 0.642, 0.645, 0.642)*
- `pareto_wr_H32_th2` = **0.6121** ± 0.0006  *(runs: 0.612, 0.613, 0.612)*
- `pareto_wr_H32_th3` = **0.5873** ± 0.0022  *(runs: 0.587, 0.589, 0.585)*
- `pareto_wr_H32_th4` = **0.5056** ± 0.0142  *(runs: 0.494, 0.521, 0.501)*
- `pareto_wr_H32_th5` = **0.0000**  *(stable across seeds)*
- `pareto_wr_H64_th0` = **1.0000**  *(stable across seeds)*
- `pareto_wr_H64_th1` = **0.6342** ± 0.0008  *(runs: 0.635, 0.634, 0.634)*
- `pareto_wr_H64_th2` = **0.5997** ± 0.0036  *(runs: 0.599, 0.604, 0.597)*
- `pareto_wr_H64_th3` = **0.5895** ± 0.0004  *(runs: 0.590, 0.590, 0.589)*
- `pareto_wr_H64_th4` = **0.5873** ± 0.0032  *(runs: 0.588, 0.590, 0.584)*
- `pareto_wr_H64_th5` = **0.0000**  *(stable across seeds)*

**Notes:** Knee write rates: H32=0.612, H64=0.599, H128=0.593. All in [0.40,0.60]: False.

---

### Category 31 — Top Mechanism Integration (Phase 5)
*0 supported / 2 refuted / 2 inconclusive / 0 error*

#### exp_31_1  ✗ REFUTED
**Hypothesis:** Combined model (retroactive re-encoding + two-hop retrieval) outperforms both in isolation by >5% using pre-training isolation.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 34s

**Metrics (mean ± std across seeds):**

- `acc_combined` = **0.1625** ± 0.0600  *(runs: 0.135, 0.121, 0.231)*
- `acc_retro` = **0.5854** ± 0.0868  *(runs: 0.647, 0.486, 0.623)*
- `acc_twohop` = **0.6408** ± 0.0687  *(runs: 0.718, 0.585, 0.620)*
- `gain_vs_retro` = **-0.4229** ± 0.0787  *(runs: -0.512, -0.365, -0.391)*
- `gain_vs_twohop` = **-0.4783** ± 0.0977  *(runs: -0.583, -0.464, -0.389)*

**Notes:** Retro=0.647, TwoHop=0.718, Combined=0.135. Gain vs retro: -0.512, gain vs twohop: -0.583.

---
#### exp_31_2  ✗ REFUTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'REFUTED', 'REFUTED']
**Hypothesis:** Retroactive re-encoding gap persists above +0.08 at SEQ_LEN=192 (8× baseline).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1211s

**Metrics (mean ± std across seeds):**

- `acc_forward_len192` = **0.1292** ± 0.0154  *(runs: 0.120, 0.147, 0.120)*
- `acc_forward_len24` = **0.1334** ± 0.0160  *(runs: 0.122, 0.127, 0.152)*
- `acc_forward_len48` = **0.1339** ± 0.0104  *(runs: 0.131, 0.125, 0.145)*
- `acc_forward_len96` = **0.1229** ± 0.0198  *(runs: 0.127, 0.102, 0.141)*
- `acc_retro_len192` = **0.1265** ± 0.0098  *(runs: 0.138, 0.123, 0.119)*
- `acc_retro_len24` = **0.1130** ± 0.0156  *(runs: 0.095, 0.125, 0.119)*
- `acc_retro_len48` = **0.1161** ± 0.0166  *(runs: 0.098, 0.119, 0.131)*
- `acc_retro_len96` = **0.1089** ± 0.0113  *(runs: 0.103, 0.122, 0.102)*
- `gap_len192` = **-0.0026** ± 0.0204  *(runs: 0.017, -0.024, -0.002)*
- `gap_len24` = **-0.0204** ± 0.0166  *(runs: -0.027, -0.002, -0.033)*
- `gap_len48` = **-0.0177** ± 0.0137  *(runs: -0.033, -0.006, -0.014)*
- `gap_len96` = **-0.0141** ± 0.0308  *(runs: -0.024, 0.020, -0.039)*

**Notes:** Gaps: len24=-0.027, len48=-0.033, len96=-0.023, len192=0.017. 192 gap > 0.08: False

---
#### exp_31_3  ~ INCONCLUSIVE
**Hypothesis:** DeltaRetroModel outperforms delta-only by >8% and outperforms retro-only by >2%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 69s

**Metrics (mean ± std across seeds):**

- `acc_delta` = **0.1596** ± 0.0118  *(runs: 0.154, 0.173, 0.151)*
- `acc_delta_retro` = **0.1823** ± 0.0109  *(runs: 0.179, 0.194, 0.173)*
- `acc_retro` = **0.1025** ± 0.0251  *(runs: 0.129, 0.079, 0.100)*
- `gain_vs_delta` = **0.0227** ± 0.0020  *(runs: 0.025, 0.021, 0.022)*
- `gain_vs_retro` = **0.0798** ± 0.0330  *(runs: 0.051, 0.116, 0.073)*

**Notes:** Delta=0.154, Retro=0.129, DeltaRetro=0.179. Gain vs delta: +0.025, gain vs retro: +0.051.

---
#### exp_31_4  ~ INCONCLUSIVE
**Hypothesis:** Learned importance-based eviction achieves >10% better accuracy than FIFO eviction when NUM_PAIRS (12) exceeds CAPACITY_LIMIT (8).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 280s

**Metrics (mean ± std across seeds):**

- `acc_fifo` = **0.2212** ± 0.0554  *(runs: 0.214, 0.280, 0.170)*
- `acc_learned` = **0.2650** ± 0.0380  *(runs: 0.241, 0.309, 0.245)*
- `capacity_limit` = **8.0000**  *(stable across seeds)*
- `gap` = **0.0438** ± 0.0271  *(runs: 0.028, 0.029, 0.075)*
- `num_pairs` = **12.0000**  *(stable across seeds)*

**Notes:** FIFO=0.214, Learned=0.241, Gap=+0.028. Capacity=8, NumPairs=12.

---

### Category 32 — Deep Seed Validation (Phase 5)
*4 supported / 0 refuted / 0 inconclusive / 0 error*

#### exp_32_1  ✓ SUPPORTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'SUPPORTED', 'INCONCLUSIVE', 'INCONCLUSIVE', 'REFUTED', 'REFUTED', 'SUPPORTED', 'SUPPORTED']
**Hypothesis:** Deep seed validation of retroactive writing (exp_3_6): gap > 0.09 (stricter criterion for robust confirmation).

**Runs:** 9 (seeds: [0, 123, 13, 1, 2, 42, 777, 7, 99])  |  **Avg duration:** 50s

**Metrics (mean ± std across seeds):**

- `acc_gap` = **0.0431** ± 0.1182  *(runs: -0.072, 0.172, 0.183, 0.014, 0.027, -0.044, -0.153, 0.130, 0.131)*
- `forward_acc` = **0.1269** ± 0.0721  *(runs: 0.189, 0.031, 0.069, 0.167, 0.181, 0.208, 0.188, 0.053, 0.056)*
- `retroactive_write_rate` = **0.0833**  *(stable across seeds)*
- `seed` = **118.2222** ± 251.2115  *(runs: 0.000, 123.000, 13.000, 1.000, 2.000, 42.000, 777.000, 7.000, 99.000)*
- `two_pass_acc` = **0.1700** ± 0.0622  *(runs: 0.117, 0.203, 0.252, 0.181, 0.208, 0.164, 0.034, 0.183, 0.188)*

**Notes:** Seed=0. Two-pass vs forward gap=-0.072. Retroactive write rate=0.083. Required gap > 0.09.

---
#### exp_32_2  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'REFUTED', 'SUPPORTED', 'REFUTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** Deep seed validation of energy-gated delta rule (exp_15_3): acc_ratio > 0.90 AND write_rate < 0.70.

**Runs:** 9 (seeds: [0, 123, 13, 1, 2, 42, 777, 7, 99])  |  **Avg duration:** 28s

**Metrics (mean ± std across seeds):**

- `acc_A_continuous` = **0.1395** ± 0.0162  *(runs: 0.138, 0.114, 0.151, 0.116, 0.154, 0.137, 0.160, 0.136, 0.149)*
- `acc_B_energy_gated` = **0.1392** ± 0.0146  *(runs: 0.124, 0.171, 0.141, 0.132, 0.138, 0.146, 0.142, 0.139, 0.119)*
- `acc_ratio_B` = **1.0158** ± 0.2064  *(runs: 0.904, 1.492, 0.934, 1.141, 0.895, 1.064, 0.891, 1.023, 0.799)*
- `seed` = **118.2222** ± 251.2115  *(runs: 0.000, 123.000, 13.000, 1.000, 2.000, 42.000, 777.000, 7.000, 99.000)*
- `write_rate_A` = **1.0000**  *(stable across seeds)*
- `write_rate_B` = **0.5154** ± 0.0024  *(runs: 0.515, 0.516, 0.520, 0.517, 0.513, 0.514, 0.517, 0.514, 0.513)*

**Notes:** Seed=0. acc_ratio=0.905, write_rate=0.515. acc_A=0.138, acc_B=0.124. Required: ratio>0.90 AND wr<0.70.

---
#### exp_32_3  ✓ SUPPORTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'SUPPORTED', 'INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED']
**Hypothesis:** Deep seed validation of three-hop chain (exp_13_2): degradation_ratio > 2.0 (three-hop exceeds two-hop by 2×).

**Runs:** 9 (seeds: [0, 123, 13, 1, 2, 42, 777, 7, 99])  |  **Avg duration:** 2s

**Metrics (mean ± std across seeds):**

- `acc_single` = **0.0625** ± 0.0350  *(runs: 0.094, 0.062, 0.094, 0.125, 0.031, 0.031, 0.031, 0.062, 0.031)*
- `acc_three` = **0.0972** ± 0.0244  *(runs: 0.062, 0.125, 0.062, 0.094, 0.094, 0.094, 0.125, 0.125, 0.094)*
- `acc_two` = **0.0312**  *(stable across seeds)*
- `degradation_ratio` = **3.1111** ± 0.7817  *(runs: 2.000, 4.000, 2.000, 3.000, 3.000, 3.000, 4.000, 4.000, 3.000)*
- `seed` = **118.2222** ± 251.2115  *(runs: 0.000, 123.000, 13.000, 1.000, 2.000, 42.000, 777.000, 7.000, 99.000)*

**Notes:** Seed=0. degradation_ratio=2.000. Accs: 1-hop=0.094, 2-hop=0.031, 3-hop=0.062. Required ratio > 2.0 to confirm three-hop beats two-hop finding.

---
#### exp_32_4  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'REFUTED', 'INCONCLUSIVE', 'SUPPORTED', 'SUPPORTED']
**Hypothesis:** Deep seed validation of parametric seq scaling (exp_16_3): param retention gap > 0.35 (stricter criterion).

**Runs:** 9 (seeds: [0, 123, 13, 1, 2, 42, 777, 7, 99])  |  **Avg duration:** 28s

**Metrics (mean ± std across seeds):**

- `acc_param_len24` = **0.0291** ± 0.0121  *(runs: 0.044, 0.044, 0.016, 0.025, 0.025, 0.028, 0.031, 0.041, 0.009)*
- `acc_param_len48` = **0.0281** ± 0.0084  *(runs: 0.034, 0.028, 0.025, 0.034, 0.028, 0.016, 0.016, 0.041, 0.031)*
- `acc_slot_len24` = **0.1076** ± 0.0344  *(runs: 0.141, 0.081, 0.138, 0.128, 0.084, 0.113, 0.109, 0.138, 0.037)*
- `acc_slot_len48` = **0.0361** ± 0.0144  *(runs: 0.041, 0.031, 0.019, 0.037, 0.047, 0.053, 0.019, 0.022, 0.056)*
- `retention_diff` = **0.7719** ± 0.5839  *(runs: 0.498, 0.259, 1.467, 1.083, 0.568, 0.083, 0.329, 0.841, 1.818)*
- `retention_parametric` = **1.2119** ± 0.8743  *(runs: 0.787, 0.643, 1.603, 1.376, 1.124, 0.555, 0.500, 1.000, 3.319)*
- `retention_slot` = **0.4401** ± 0.4231  *(runs: 0.289, 0.384, 0.136, 0.293, 0.556, 0.472, 0.171, 0.159, 1.501)*
- `seed` = **118.2222** ± 251.2115  *(runs: 0.000, 123.000, 13.000, 1.000, 2.000, 42.000, 777.000, 7.000, 99.000)*

**Notes:** Seed=0. Param retention gap=0.498. Required gap > 0.35. Slot retention=0.289, Param retention=0.787.

---

### Category 33 — Capacity Physics / Interference Density Law (Phase 5)
*1 supported / 2 refuted / 1 inconclusive / 0 error*

#### exp_33_1  ✗ REFUTED
**Hypothesis:** Slot memory accuracy follows acc ~ ρ^(-γ) with R² > 0.90 across ρ ∈ {0.031, 0.063, 0.125, 0.25, 0.5, 1.0} (N_pairs/hidden_dim).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 58s

**Metrics (mean ± std across seeds):**

- `acc_n16` = **0.0089** ± 0.0043  *(runs: 0.005, 0.009, 0.013)*
- `acc_n2` = **0.0070** ± 0.0008  *(runs: 0.008, 0.007, 0.006)*
- `acc_n32` = **0.0091** ± 0.0024  *(runs: 0.009, 0.007, 0.012)*
- `acc_n4` = **0.0058** ± 0.0005  *(runs: 0.005, 0.005, 0.006)*
- `acc_n64` = **0.0078** ± 0.0008  *(runs: 0.007, 0.008, 0.009)*
- `acc_n8` = **0.0120** ± 0.0052  *(runs: 0.009, 0.009, 0.018)*
- `gamma_slot` = **-0.0616** ± 0.0624  *(runs: -0.004, -0.052, -0.128)*
- `intercept` = **-4.7250** ± 0.2974  *(runs: -4.959, -4.825, -4.390)*
- `r_squared` = **0.1051** ± 0.0906  *(runs: 0.001, 0.161, 0.153)*
- `rho_n16` = **0.2500**  *(stable across seeds)*
- `rho_n2` = **0.0312**  *(stable across seeds)*
- `rho_n32` = **0.5000**  *(stable across seeds)*
- `rho_n4` = **0.0625**  *(stable across seeds)*
- `rho_n64` = **1.0000**  *(stable across seeds)*
- `rho_n8` = **0.1250**  *(stable across seeds)*

**Notes:** Slot memory γ=-0.004, R²=0.000. Accs: [0.0078, 0.0055, 0.0094, 0.0047, 0.0086, 0.007].

---
#### exp_33_2  ~ INCONCLUSIVE ⚠ inconsistent across seeds ['INCONCLUSIVE', 'INCONCLUSIVE', 'SUPPORTED']
**Hypothesis:** Different architectures have distinct interference exponents: γ_parametric < γ_slot < γ_delta with spread > 0.3.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 988s

**Metrics (mean ± std across seeds):**

- `acc_delta_n16` = **0.0130** ± 0.0009  *(runs: 0.013, 0.014, 0.013)*
- `acc_delta_n2` = **0.3911** ± 0.0173  *(runs: 0.407, 0.373, 0.394)*
- `acc_delta_n32` = **0.0050** ± 0.0012  *(runs: 0.006, 0.004, 0.005)*
- `acc_delta_n4` = **0.1136** ± 0.0285  *(runs: 0.146, 0.093, 0.102)*
- `acc_delta_n64` = **0.0063** ± 0.0034  *(runs: 0.005, 0.004, 0.010)*
- `acc_delta_n8` = **0.0187** ± 0.0061  *(runs: 0.026, 0.015, 0.016)*
- `acc_param_n16` = **0.0094** ± 0.0016  *(runs: 0.008, 0.009, 0.011)*
- `acc_param_n2` = **0.0091** ± 0.0032  *(runs: 0.005, 0.010, 0.012)*
- `acc_param_n32` = **0.0060** ± 0.0012  *(runs: 0.005, 0.007, 0.006)*
- `acc_param_n4` = **0.0070** ± 0.0021  *(runs: 0.005, 0.009, 0.008)*
- `acc_param_n64` = **0.0091** ± 0.0037  *(runs: 0.008, 0.006, 0.013)*
- `acc_param_n8` = **0.0076** ± 0.0044  *(runs: 0.006, 0.004, 0.013)*
- `acc_slot_n16` = **0.0107** ± 0.0047  *(runs: 0.010, 0.016, 0.006)*
- `acc_slot_n2` = **0.0063** ± 0.0016  *(runs: 0.005, 0.006, 0.008)*
- `acc_slot_n32` = **0.0078** ± 0.0021  *(runs: 0.009, 0.005, 0.009)*
- `acc_slot_n4` = **0.0565** ± 0.0885  *(runs: 0.008, 0.003, 0.159)*
- `acc_slot_n64` = **0.0070** ± 0.0036  *(runs: 0.011, 0.006, 0.004)*
- `acc_slot_n8` = **0.0102**  *(stable across seeds)*
- `gamma_delta` = **1.2711** ± 0.1119  *(runs: 1.338, 1.334, 1.142)*
- `gamma_param` = **0.0045** ± 0.0847  *(runs: -0.081, 0.088, 0.006)*
- `gamma_slot` = **0.0793** ± 0.3878  *(runs: -0.196, -0.088, 0.523)*
- `gamma_spread` = **1.3641** ± 0.2052  *(runs: 1.534, 1.422, 1.136)*
- `ordering_ok` = **0.3333** ± 0.5774  *(runs: 0.000, 0.000, 1.000)*
- `r2_delta` = **0.8825** ± 0.0793  *(runs: 0.944, 0.911, 0.793)*
- `r2_param` = **0.1043** ± 0.1026  *(runs: 0.206, 0.107, 0.001)*
- `r2_slot` = **0.3207** ± 0.3090  *(runs: 0.654, 0.043, 0.265)*

**Notes:** γ ordering: param=-0.081 < slot=-0.196 < delta=1.338, spread=1.534, R²: slot=0.654 delta=0.944 param=0.206. Ordering correct: False.

---
#### exp_33_3  ✗ REFUTED
**Hypothesis:** Interference exponent γ is independent of hidden dimension: γ values at H=32, H=64, H=128 are within ±0.1 for each architecture.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 1557s

**Metrics (mean ± std across seeds):**

- `gamma_delta_H128` = **0.8019** ± 0.1649  *(runs: 0.921, 0.870, 0.614)*
- `gamma_delta_H32` = **1.0263** ± 0.0924  *(runs: 0.949, 1.001, 1.129)*
- `gamma_delta_H64` = **1.0701** ± 0.0702  *(runs: 1.007, 1.057, 1.146)*
- `gamma_slot_H128` = **0.1796** ± 0.1821  *(runs: 0.152, 0.013, 0.374)*
- `gamma_slot_H32` = **0.0664** ± 0.2485  *(runs: 0.350, -0.113, -0.038)*
- `gamma_slot_H64` = **0.0382** ± 0.1342  *(runs: -0.115, 0.134, 0.096)*
- `max_gamma_spread_delta` = **0.2682** ± 0.2340  *(runs: 0.086, 0.187, 0.532)*
- `max_gamma_spread_slot` = **0.3747** ± 0.1140  *(runs: 0.465, 0.247, 0.412)*
- `max_spread_overall` = **0.4146** ± 0.1493  *(runs: 0.465, 0.247, 0.532)*
- `r2_delta_H128` = **0.7319** ± 0.0561  *(runs: 0.792, 0.681, 0.722)*
- `r2_delta_H32` = **0.8277** ± 0.0882  *(runs: 0.727, 0.891, 0.865)*
- `r2_delta_H64` = **0.8362** ± 0.0771  *(runs: 0.770, 0.818, 0.921)*
- `r2_slot_H128` = **0.1197** ± 0.1337  *(runs: 0.093, 0.001, 0.265)*
- `r2_slot_H32` = **0.2617** ± 0.1639  *(runs: 0.446, 0.132, 0.207)*
- `r2_slot_H64` = **0.0912** ± 0.0737  *(runs: 0.170, 0.081, 0.023)*

**Notes:** Slot γ spread=0.465, Delta γ spread=0.086. Max of both=0.465. Slot γ: H32=0.350 H64=-0.115 H128=0.152. Delta γ: H32=0.949 H64=1.007 H128=0.921.

---
#### exp_33_4  ✓ SUPPORTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'SUPPORTED']
**Hypothesis:** Tripling training steps (400→1200) at ρ=1.0 recovers >50% of accuracy lost vs ρ=0.5 for at least one architecture.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 791s

**Metrics (mean ± std across seeds):**

- `acc_delta_1200` = **0.0065** ± 0.0024  *(runs: 0.009, 0.004, 0.007)*
- `acc_delta_400` = **0.0073** ± 0.0012  *(runs: 0.009, 0.006, 0.007)*
- `acc_delta_800` = **0.0107** ± 0.0030  *(runs: 0.014, 0.009, 0.009)*
- `acc_delta_rho05` = **0.0096** ± 0.0020  *(runs: 0.012, 0.009, 0.008)*
- `acc_slot_1200` = **0.0094** ± 0.0041  *(runs: 0.005, 0.013, 0.011)*
- `acc_slot_400` = **0.0063** ± 0.0027  *(runs: 0.005, 0.009, 0.005)*
- `acc_slot_800` = **0.0057** ± 0.0028  *(runs: 0.003, 0.005, 0.009)*
- `acc_slot_rho05` = **0.0083** ± 0.0017  *(runs: 0.008, 0.010, 0.007)*
- `recovery_delta` = **-0.2581** ± 0.4470  *(runs: 0.000, -0.774, 0.000)*
- `recovery_slot` = **2.1902** ± 1.9863  *(runs: 0.000, 3.875, 2.696)*

**Notes:** Slot recovery=0.000, Delta recovery=0.000. ρ=0.5 ref: slot=0.008, delta=0.012. ρ=1.0@400: slot=0.005, delta=0.009. ρ=1.0@1200: slot=0.005, delta=0.009.

---

### Category 34 — Training Dynamics (Phase 6)
*1 supported / 5 refuted / 3 inconclusive / 0 error*

#### exp_34_1  ✗ REFUTED
**Hypothesis:** Delta rule memory shows a sharper phase transition (≥30% accuracy gain within 200 steps) compared to slot memory.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 48s

**Metrics (mean ± std across seeds):**

- `delta_chk1000` = **0.2854** ± 0.0280  *(runs: 0.300, 0.303, 0.253)*
- `delta_chk1250` = **0.2864** ± 0.0161  *(runs: 0.291, 0.300, 0.269)*
- `delta_chk150` = **0.2062** ± 0.0174  *(runs: 0.191, 0.203, 0.225)*
- `delta_chk1500` = **0.2802** ± 0.0222  *(runs: 0.284, 0.256, 0.300)*
- `delta_chk300` = **0.2417** ± 0.0213  *(runs: 0.225, 0.234, 0.266)*
- `delta_chk50` = **0.0781** ± 0.0421  *(runs: 0.113, 0.031, 0.091)*
- `delta_chk500` = **0.2823** ± 0.0424  *(runs: 0.331, 0.256, 0.259)*
- `delta_chk750` = **0.2292** ± 0.0095  *(runs: 0.219, 0.237, 0.231)*
- `delta_final` = **0.2802** ± 0.0222  *(runs: 0.284, 0.256, 0.300)*
- `delta_has_transition` = **0.0000**  *(stable across seeds)*
- `delta_transition_gain` = **0.0000**  *(stable across seeds)*
- `delta_transition_window` = ['none', 'none', 'none']
- `slot_chk1000` = **0.0302** ± 0.0201  *(runs: 0.022, 0.053, 0.016)*
- `slot_chk1250` = **0.0291** ± 0.0018  *(runs: 0.028, 0.031, 0.028)*
- `slot_chk150` = **0.0521** ± 0.0266  *(runs: 0.025, 0.053, 0.078)*
- `slot_chk1500` = **0.0313** ± 0.0054  *(runs: 0.025, 0.034, 0.034)*
- `slot_chk300` = **0.0375** ± 0.0217  *(runs: 0.025, 0.025, 0.062)*
- `slot_chk50` = **0.0458** ± 0.0072  *(runs: 0.050, 0.050, 0.037)*
- `slot_chk500` = **0.0229** ± 0.0078  *(runs: 0.031, 0.022, 0.016)*
- `slot_chk750` = **0.0302** ± 0.0018  *(runs: 0.031, 0.028, 0.031)*
- `slot_final` = **0.0313** ± 0.0054  *(runs: 0.025, 0.034, 0.034)*
- `slot_has_transition` = **0.0000**  *(stable across seeds)*
- `slot_transition_gain` = **0.0000**  *(stable across seeds)*

**Notes:** Delta transition: False (gain=0.000, window -1--1). Slot transition: False (gain=0.000). Final: delta=0.284 slot=0.025.

---
#### exp_34_2  ~ INCONCLUSIVE
**Hypothesis:** Memory projection parameters (k/v/q) receive larger gradient than encoder early in training (ratio > 2.0 at step 100) and converge to similar magnitudes later (ratio < 1.5 at step 1000).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 27s

**Metrics (mean ± std across seeds):**

- `enc_gnorm_s100` = **2.1306** ± 0.2412  *(runs: 2.339, 2.186, 1.866)*
- `enc_gnorm_s1000` = **4.6440** ± 0.6143  *(runs: 4.843, 3.955, 5.135)*
- `enc_gnorm_s300` = **3.5942** ± 0.3176  *(runs: 3.260, 3.893, 3.630)*
- `enc_gnorm_s600` = **5.4258** ± 0.8753  *(runs: 4.681, 6.390, 5.206)*
- `mem_gnorm_s100` = **4.8566** ± 0.3218  *(runs: 5.040, 5.045, 4.485)*
- `mem_gnorm_s1000` = **9.8536** ± 1.1597  *(runs: 9.339, 9.040, 11.182)*
- `mem_gnorm_s300` = **7.2527** ± 0.6407  *(runs: 6.664, 7.935, 7.159)*
- `mem_gnorm_s600` = **10.8033** ± 2.3946  *(runs: 8.260, 13.015, 11.134)*
- `ratio_s100` = **2.2884** ± 0.1252  *(runs: 2.155, 2.307, 2.403)*
- `ratio_s1000` = **2.1307** ± 0.1831  *(runs: 1.929, 2.286, 2.178)*
- `ratio_s300` = **2.0182** ± 0.0400  *(runs: 2.044, 2.039, 1.972)*
- `ratio_s600` = **1.9800** ± 0.1934  *(runs: 1.765, 2.037, 2.139)*

**Notes:** Gradient ratio (mem/enc): early(s100)=2.155, late(s1000)=1.929. Hypothesis: ratio>2.0 early and <1.5 late.

---
#### exp_34_3  ~ INCONCLUSIVE
**Hypothesis:** Energy-gated delta write rate decreases naturally during training: ≥0.70 at step 100 and ≤0.40 at step 1500.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 41s

**Metrics (mean ± std across seeds):**

- `write_rate_s100` = **0.5345** ± 0.0029  *(runs: 0.534, 0.532, 0.538)*
- `write_rate_s1200` = **0.5414** ± 0.0004  *(runs: 0.542, 0.541, 0.541)*
- `write_rate_s1500` = **0.5414** ± 0.0038  *(runs: 0.543, 0.544, 0.537)*
- `write_rate_s300` = **0.5385** ± 0.0023  *(runs: 0.541, 0.537, 0.537)*
- `write_rate_s600` = **0.5394** ± 0.0028  *(runs: 0.541, 0.541, 0.536)*
- `write_rate_s900` = **0.5388** ± 0.0028  *(runs: 0.537, 0.542, 0.537)*

**Notes:** Write rate: early(s100)=0.534, late(s1500)=0.543. Full trajectory: {'write_rate_s100': 0.5339, 'write_rate_s300': 0.5411, 'write_rate_s600': 0.5411, 'write_rate_s900': 0.5375, 'write_rate_s1200': 0.5418, 'write_rate_s1500': 0.5431}. Hypothesis: ≥0.70 early → ≤0.40 late.

---
#### exp_34_4  ~ INCONCLUSIVE
**Hypothesis:** Easy-first curriculum (2→8 pairs) improves final accuracy vs random mixed training by >0.08 on a 4-pair evaluation task.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 99s

**Metrics (mean ± std across seeds):**

- `acc_easy_first` = **0.2778** ± 0.0312  *(runs: 0.269, 0.312, 0.252)*
- `acc_hard_first` = **0.2722** ± 0.0209  *(runs: 0.271, 0.294, 0.252)*
- `acc_random` = **0.2555** ± 0.0265  *(runs: 0.225, 0.273, 0.269)*
- `gap_easy_vs_hard` = **0.0055** ± 0.0115  *(runs: -0.002, 0.019, 0.000)*
- `gap_easy_vs_random` = **0.0222** ± 0.0337  *(runs: 0.044, 0.040, -0.017)*

**Notes:** Random=0.225, Easy-first=0.269, Hard-first=0.271. Gap(easy-random)=+0.044.

---
#### exp_34_5  ✗ REFUTED
**Hypothesis:** Gradual memory warmup (write scale 0→1 over 200 of 600 steps) improves final accuracy by >3% over full-memory training from step 0, because the backbone learns representations before the memory becomes active.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 93s

**Metrics (mean ± std across seeds):**

- `acc_full_immediate` = **0.2092** ± 0.0081  *(runs: 0.201, 0.217, 0.209)*
- `acc_warmup` = **0.2225** ± 0.0085  *(runs: 0.214, 0.231, 0.223)*
- `total_steps` = **600.0000**  *(stable across seeds)*
- `warmup_gain` = **0.0133** ± 0.0010  *(runs: 0.013, 0.013, 0.014)*
- `warmup_steps` = **200.0000**  *(stable across seeds)*

**Notes:** No significant warmup benefit: gain=0.013 (threshold >0.03). full=0.201, warmup=0.214.

---
#### exp_34_6  ✓ SUPPORTED ⚠ inconsistent across seeds ['INCONCLUSIVE', 'REFUTED', 'SUPPORTED']
**Hypothesis:** Delta rule memory architecture shows strong optimizer preference: best optimizer outperforms worst by >10% accuracy, indicating sensitivity greater than typical for standard architectures.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 146s

**Metrics (mean ± std across seeds):**

- `acc_Adam` = **0.2092** ± 0.0081  *(runs: 0.201, 0.217, 0.209)*
- `acc_AdamW` = **0.2104** ± 0.0159  *(runs: 0.193, 0.224, 0.214)*
- `acc_RMSprop` = **0.2194** ± 0.0147  *(runs: 0.209, 0.212, 0.236)*
- `acc_SGD` = **0.1571** ± 0.1362  *(runs: 0.242, 0.229, 0.000)*
- `acc_SGD_momentum` = **0.2152** ± 0.0100  *(runs: 0.220, 0.204, 0.222)*
- `best_optimizer` = ['SGD', 'SGD', 'RMSprop']
- `spread_max_min` = **0.1036** ± 0.1155  *(runs: 0.049, 0.025, 0.236)*
- `worst_optimizer` = ['AdamW', 'SGD_momentum', 'SGD']

**Notes:** Moderate spread=0.049. Best=SGD(0.242), Worst=AdamW(0.193).

---
#### exp_34_7  ✗ REFUTED
**Hypothesis:** Delta rule memory has a narrow stable learning rate band spanning <1.5 decades, indicating higher LR sensitivity than standard architectures (which typically have stable bands of 2+ decades).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 180s

**Metrics (mean ± std across seeds):**

- `acc_lr_1e-02` = **0.2448** ± 0.0124  *(runs: 0.248, 0.231, 0.256)*
- `acc_lr_1e-03` = **0.2226** ± 0.0059  *(runs: 0.223, 0.228, 0.216)*
- `acc_lr_1e-04` = **0.1490** ± 0.0176  *(runs: 0.148, 0.167, 0.132)*
- `acc_lr_1e-05` = **0.0308** ± 0.0043  *(runs: 0.035, 0.027, 0.030)*
- `acc_lr_3e-02` = **0.2227** ± 0.0196  *(runs: 0.245, 0.212, 0.210)*
- `acc_lr_3e-03` = **0.2359** ± 0.0136  *(runs: 0.245, 0.220, 0.242)*
- `acc_lr_3e-04` = **0.2128** ± 0.0051  *(runs: 0.212, 0.208, 0.218)*
- `acc_lr_3e-05` = **0.0740** ± 0.0131  *(runs: 0.088, 0.062, 0.071)*
- `max_acc` = **0.2448** ± 0.0124  *(runs: 0.248, 0.231, 0.256)*
- `max_stable_lr` = **0.0300**  *(stable across seeds)*
- `min_stable_lr` = **0.0001**  *(stable across seeds)*
- `n_stable_lrs` = **6.0000**  *(stable across seeds)*
- `stable_band_decades` = **2.4770**  *(stable across seeds)*

**Notes:** Wide stable band: 2.48 decades > 2.0. Low LR sensitivity.

---
#### exp_34_8  ✗ REFUTED
**Hypothesis:** Delta rule memory quality degrades at larger batch sizes independently of gradient noise: accuracy at B=128 is >5% lower than B=8 even when effective LR is scaled proportionally (linear scaling rule).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 164s

**Metrics (mean ± std across seeds):**

- `acc_B128` = **0.2329** ± 0.0174  *(runs: 0.214, 0.247, 0.237)*
- `acc_B16` = **0.2227** ± 0.0090  *(runs: 0.229, 0.212, 0.226)*
- `acc_B32` = **0.2231** ± 0.0152  *(runs: 0.233, 0.206, 0.231)*
- `acc_B4` = **0.1223** ± 0.0149  *(runs: 0.106, 0.127, 0.134)*
- `acc_B64` = **0.2287** ± 0.0125  *(runs: 0.216, 0.241, 0.230)*
- `acc_B8` = **0.1854** ± 0.0179  *(runs: 0.166, 0.188, 0.202)*
- `drop_B8_to_B128` = **-0.0475** ± 0.0119  *(runs: -0.047, -0.059, -0.036)*
- `spearman_batch_vs_acc` = **0.7907** ± 0.2638  *(runs: 0.486, 0.943, 0.943)*

**Notes:** No batch sensitivity: drop=-0.047<0.01. Memory quality scales normally with batch size.

---
#### exp_34_9  ✗ REFUTED
**Hypothesis:** At convergence, more than 40% of learned write gate activations are in the dead zone (<0.05) or saturated zone (>0.95), indicating bimodal gate collapse rather than graded, informative gating.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 32s

**Metrics (mean ± std across seeds):**

- `acc` = **0.2476** ± 0.0047  *(runs: 0.248, 0.243, 0.252)*
- `bimodal_fraction` = **0.0000**  *(stable across seeds)*
- `gate_dead_frac` = **0.0000**  *(stable across seeds)*
- `gate_entropy_normalized` = **0.3315** ± 0.0487  *(runs: 0.303, 0.304, 0.388)*
- `gate_mean` = **0.4178** ± 0.0522  *(runs: 0.376, 0.476, 0.402)*
- `gate_saturated_frac` = **0.0000**  *(stable across seeds)*
- `gate_std` = **0.0371** ± 0.0044  *(runs: 0.036, 0.033, 0.042)*
- `total_gate_samples` = **147200.0000**  *(stable across seeds)*

**Notes:** Gate is well-spread: bimodal_fraction=0.000 < 0.15. Entropy=0.303. No dead/saturation collapse.

---

### Category 35 — Failure Modes (Phase 6)
*2 supported / 1 refuted / 0 inconclusive / 0 error*

#### exp_35_1  ✗ REFUTED
**Hypothesis:** Delta rule memory degrades gracefully under post-hoc noise injection: accuracy at 30% additive noise to M is >60% of clean baseline, not a catastrophic cliff (>70% accuracy drop).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 49s

**Metrics (mean ± std across seeds):**

- `acc_noise_0pct` = **0.2049** ± 0.0112  *(runs: 0.194, 0.216, 0.205)*
- `acc_noise_100pct` = **0.0203** ± 0.0023  *(runs: 0.023, 0.019, 0.019)*
- `acc_noise_10pct` = **0.0217** ± 0.0056  *(runs: 0.019, 0.018, 0.028)*
- `acc_noise_200pct` = **0.0155** ± 0.0015  *(runs: 0.015, 0.015, 0.017)*
- `acc_noise_25pct` = **0.0160** ± 0.0016  *(runs: 0.016, 0.015, 0.018)*
- `acc_noise_50pct` = **0.0165** ± 0.0038  *(runs: 0.015, 0.013, 0.021)*
- `baseline_acc` = **0.2049** ± 0.0112  *(runs: 0.194, 0.216, 0.205)*
- `cliff_at_noise` = **0.1000**  *(stable across seeds)*
- `cliff_detected` = [True, True, True]
- `retention_at_25pct_noise` = **0.0782** ± 0.0097  *(runs: 0.081, 0.068, 0.086)*
- `retention_at_50pct_noise` = **0.0807** ± 0.0197  *(runs: 0.078, 0.062, 0.102)*

**Notes:** Catastrophic degradation: baseline=0.194, @25%noise=0.016 (retention=0.080). Cliff at noise=0.1.

---
#### exp_35_2  ✓ SUPPORTED
**Hypothesis:** Slot memory architecture does NOT hallucinate: querying with keys never presented in context produces accuracy near random chance, not significantly above chance (< random + 5%).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 25s

**Metrics (mean ± std across seeds):**

- `acc_in_distribution` = **0.1919** ± 0.0170  *(runs: 0.185, 0.211, 0.179)*
- `acc_ood_query` = **0.0324** ± 0.0067  *(runs: 0.040, 0.031, 0.027)*
- `hallucination_ratio` = **0.1698** ± 0.0394  *(runs: 0.215, 0.146, 0.148)*
- `ood_above_random` = **0.0012** ± 0.0068  *(runs: 0.009, -0.000, -0.005)*
- `random_baseline` = **0.0312**  *(stable across seeds)*

**Notes:** No hallucination: OOD acc=0.040 ≈ random baseline=0.031 (gap=0.009<0.05). In-dist acc=0.185.

---
#### exp_35_3  ✓ SUPPORTED ⚠ inconsistent across seeds ['SUPPORTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Out-of-distribution inputs cause abnormal write gate behavior: the write rate for OOD tokens deviates from in-distribution write rate by more than 2×, revealing that the gate is not robust to OOD inputs.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 71s

**Metrics (mean ± std across seeds):**

- `acc_ood_0` = **0.2957** ± 0.0034  *(runs: 0.297, 0.292, 0.298)*
- `acc_ood_100` = **0.0367** ± 0.0236  *(runs: 0.063, 0.018, 0.029)*
- `acc_ood_25` = **0.1854** ± 0.0301  *(runs: 0.172, 0.165, 0.220)*
- `acc_ood_50` = **0.1226** ± 0.0173  *(runs: 0.120, 0.107, 0.141)*
- `acc_ood_75` = **0.0846** ± 0.0179  *(runs: 0.088, 0.065, 0.100)*
- `baseline_write_rate` = **0.3120** ± 0.4736  *(runs: 0.000, 0.857, 0.079)*
- `full_ood_write_rate` = **0.3304** ± 0.4503  *(runs: 0.000, 0.843, 0.148)*
- `wr_ood_0` = **0.3120** ± 0.4736  *(runs: 0.000, 0.857, 0.079)*
- `wr_ood_100` = **0.3304** ± 0.4503  *(runs: 0.000, 0.843, 0.148)*
- `wr_ood_25` = **0.3171** ± 0.4679  *(runs: 0.000, 0.855, 0.097)*
- `wr_ood_50` = **0.3208** ± 0.4609  *(runs: 0.000, 0.849, 0.113)*
- `wr_ood_75` = **0.3260** ± 0.4563  *(runs: 0.000, 0.848, 0.131)*
- `wr_ratio_ood_vs_baseline` = **0.9513** ± 0.9353  *(runs: 0.000, 0.984, 1.870)*

**Notes:** Abnormal OOD write rate: baseline=0.000, OOD=0.000, ratio=0.00 (outside [0.5, 2.0]). Gate not robust to OOD inputs.

---

### Category 36 — Biological Analogues (Phase 6)
*1 supported / 1 refuted / 1 inconclusive / 0 error*

#### exp_36_1  ✗ REFUTED
**Hypothesis:** An offline consolidation phase — replaying all written key-value pairs through the memory without new input — improves associative recall accuracy by >3% over single-pass writing, analogous to hippocampal-cortical consolidation.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 43s

**Metrics (mean ± std across seeds):**

- `acc_no_consolidation` = **0.1778** ± 0.0084  *(runs: 0.170, 0.177, 0.186)*
- `acc_with_consolidation` = **0.1696** ± 0.0112  *(runs: 0.175, 0.177, 0.157)*
- `consolidation_gain` = **-0.0082** ± 0.0188  *(runs: 0.005, 0.000, -0.030)*
- `consolidation_passes` = **2.0000**  *(stable across seeds)*

**Notes:** No consolidation benefit: gain=0.005 (threshold >0.03). Offline replay does not improve retrieval.

---
#### exp_36_2  ~ INCONCLUSIVE
**Hypothesis:** Storing prediction residuals (what the model predicted wrong) rather than full token representations produces equivalent or better associative recall accuracy with the same memory capacity — the predictive coding hypothesis.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 70s

**Metrics (mean ± std across seeds):**

- `acc_full_representation` = **0.2049** ± 0.0112  *(runs: 0.194, 0.216, 0.205)*
- `acc_residual_coding` = **0.1977** ± 0.0069  *(runs: 0.190, 0.200, 0.204)*
- `residual_advantage` = **-0.0071** ± 0.0084  *(runs: -0.004, -0.017, -0.001)*

**Notes:** Roughly equivalent: full=0.194, residual=0.190, gap=-0.004. Neither clearly superior.

---
#### exp_36_3  ✓ SUPPORTED
**Hypothesis:** Separating episodic memory (what happened: event order/temporal context) from semantic memory (what things mean: content associations) outperforms a unified memory store by >5% on tasks requiring both types of recall.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 107s

**Metrics (mean ± std across seeds):**

- `acc_split` = **0.2359** ± 0.0189  *(runs: 0.215, 0.243, 0.251)*
- `acc_unified` = **0.1606** ± 0.0148  *(runs: 0.156, 0.177, 0.148)*
- `params_split` = **29120.0000**  *(stable across seeds)*
- `params_unified` = **29120.0000**  *(stable across seeds)*
- `split_advantage` = **0.0753** ± 0.0235  *(runs: 0.058, 0.066, 0.102)*

**Notes:** Split memory wins: unified=0.156, split=0.215, advantage=0.058>0.05.

---

### Category 37 — Robustness (Phase 7)
*1 supported / 0 refuted / 2 inconclusive / 0 error*

#### exp_37_1  ~ INCONCLUSIVE
**Hypothesis:** Training DeltaModel with σ_train=0.05 noise on M keeps acc_ratio(σ=0.10) ≥ 0.50 (vs 0.08 ratio found in exp_35_1 with no augmentation).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 54s

**Metrics (mean ± std across seeds):**

- `aug_improvement` = **0.0067** ± 0.0508  *(runs: 0.037, -0.052, 0.035)*
- `aug_s000` = **0.2156** ± 0.0094  *(runs: 0.207, 0.226, 0.214)*
- `aug_s003` = **0.2154** ± 0.0028  *(runs: 0.212, 0.218, 0.216)*
- `aug_s005` = **0.2292** ± 0.0095  *(runs: 0.223, 0.225, 0.240)*
- `aug_s010` = **0.2115** ± 0.0145  *(runs: 0.202, 0.228, 0.204)*
- `aug_s020` = **0.2160** ± 0.0053  *(runs: 0.218, 0.220, 0.210)*
- `baseline_aug` = **0.2156** ± 0.0094  *(runs: 0.207, 0.226, 0.214)*
- `baseline_std` = **0.2277** ± 0.0007  *(runs: 0.227, 0.228, 0.228)*
- `ratio_aug_at10` = **0.9801** ± 0.0291  *(runs: 0.976, 1.011, 0.953)*
- `ratio_std_at10` = **0.9734** ± 0.0784  *(runs: 0.939, 1.063, 0.918)*
- `std_s000` = **0.2277** ± 0.0007  *(runs: 0.227, 0.228, 0.228)*
- `std_s003` = **0.2225** ± 0.0049  *(runs: 0.219, 0.220, 0.228)*
- `std_s005` = **0.2292** ± 0.0058  *(runs: 0.223, 0.232, 0.233)*
- `std_s010` = **0.2217** ± 0.0181  *(runs: 0.213, 0.242, 0.209)*
- `std_s020` = **0.2256** ± 0.0051  *(runs: 0.227, 0.230, 0.220)*

**Notes:** Partial improvement: ratio_aug=0.976, ratio_std=0.939. aug_improvement=0.037.

---
#### exp_37_2  ~ INCONCLUSIVE
**Hypothesis:** Row-normalizing M after each write bounds M magnitude; acc_ratio at σ=0.10 ≥ 0.50 (vs 0.08 for standard delta).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 60s

**Metrics (mean ± std across seeds):**

- `baseline_norm` = **0.2248** ± 0.0047  *(runs: 0.227, 0.228, 0.219)*
- `baseline_std` = **0.2273** ± 0.0078  *(runs: 0.220, 0.226, 0.236)*
- `improvement` = **0.0220** ± 0.0762  *(runs: -0.064, 0.081, 0.049)*
- `norm_s000` = **0.2248** ± 0.0047  *(runs: 0.227, 0.228, 0.219)*
- `norm_s003` = **0.2154** ± 0.0121  *(runs: 0.219, 0.202, 0.225)*
- `norm_s005` = **0.2202** ± 0.0051  *(runs: 0.214, 0.224, 0.223)*
- `norm_s010` = **0.2150** ± 0.0017  *(runs: 0.214, 0.214, 0.217)*
- `norm_s020` = **0.2185** ± 0.0163  *(runs: 0.235, 0.218, 0.203)*
- `ratio_norm_at10` = **0.9568** ± 0.0276  *(runs: 0.942, 0.940, 0.989)*
- `ratio_std_at10` = **0.9347** ± 0.0736  *(runs: 1.006, 0.859, 0.939)*
- `std_s000` = **0.2273** ± 0.0078  *(runs: 0.220, 0.226, 0.236)*
- `std_s003` = **0.2175** ± 0.0141  *(runs: 0.233, 0.214, 0.206)*
- `std_s005` = **0.2129** ± 0.0104  *(runs: 0.203, 0.224, 0.212)*
- `std_s010` = **0.2123** ± 0.0155  *(runs: 0.221, 0.194, 0.221)*
- `std_s020` = **0.2144** ± 0.0066  *(runs: 0.222, 0.209, 0.212)*

**Notes:** Partial: ratio_norm=0.942, ratio_std=1.006, improvement=-0.064.

---
#### exp_37_3  ✓ SUPPORTED
**Hypothesis:** EMA update (α=0.85) achieves acc_ratio ≥ 0.50 at σ=0.10 while retaining ≥ 95% of clean accuracy.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 100s

**Metrics (mean ± std across seeds):**

- `best_alpha` = **0.9167** ± 0.0577  *(runs: 0.950, 0.850, 0.950)*
- `best_clean` = **0.2340** ± 0.0188  *(runs: 0.222, 0.256, 0.224)*
- `best_ratio` = **1.0476** ± 0.0761  *(runs: 1.079, 0.961, 1.103)*
- `ema070_s000` = **0.2606** ± 0.0047  *(runs: 0.260, 0.266, 0.256)*
- `ema070_s005` = **0.2577** ± 0.0097  *(runs: 0.258, 0.248, 0.268)*
- `ema070_s010` = **0.2460** ± 0.0020  *(runs: 0.244, 0.247, 0.247)*
- `ema070_s020` = **0.2523** ± 0.0087  *(runs: 0.245, 0.262, 0.250)*
- `ema085_s000` = **0.2546** ± 0.0066  *(runs: 0.247, 0.256, 0.261)*
- `ema085_s005` = **0.2513** ± 0.0016  *(runs: 0.253, 0.252, 0.249)*
- `ema085_s010` = **0.2562** ± 0.0133  *(runs: 0.252, 0.246, 0.271)*
- `ema085_s020` = **0.2471** ± 0.0064  *(runs: 0.254, 0.243, 0.244)*
- `ema095_s000` = **0.2296** ± 0.0112  *(runs: 0.222, 0.242, 0.224)*
- `ema095_s005` = **0.2221** ± 0.0104  *(runs: 0.221, 0.233, 0.212)*
- `ema095_s010` = **0.2356** ± 0.0141  *(runs: 0.239, 0.220, 0.247)*
- `ema095_s020` = **0.2208** ± 0.0084  *(runs: 0.224, 0.227, 0.211)*
- `ratio_ema070` = **0.9443** ± 0.0192  *(runs: 0.938, 0.929, 0.966)*
- `ratio_ema085` = **1.0065** ± 0.0411  *(runs: 1.018, 0.961, 1.041)*
- `ratio_ema095` = **1.0297** ± 0.1068  *(runs: 1.079, 0.907, 1.103)*
- `ratio_std` = **0.9938** ± 0.0941  *(runs: 1.063, 0.887, 1.032)*
- `std_s000` = **0.2190** ± 0.0114  *(runs: 0.211, 0.232, 0.214)*
- `std_s005` = **0.2225** ± 0.0051  *(runs: 0.218, 0.221, 0.228)*
- `std_s010` = **0.2169** ± 0.0099  *(runs: 0.224, 0.206, 0.221)*
- `std_s020` = **0.2117** ± 0.0097  *(runs: 0.203, 0.211, 0.222)*

**Notes:** EMA (α=0.95) resilient: ratio=1.079 ≥ 0.50, clean_loss=-0.054 ≤ 0.05. ratio_std=1.063.

---

### Category 38 — Episodic/Semantic Architecture (Phase 7)
*1 supported / 2 refuted / 0 inconclusive / 0 error*

#### exp_38_1  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'REFUTED', 'INCONCLUSIVE']
**Hypothesis:** Learned soft-routing on episodic/semantic split outperforms fixed 50/50 split by >5% accuracy; router learns a non-trivial allocation (not always 0.5).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 101s

**Metrics (mean ± std across seeds):**

- `acc_fixed` = **0.3545** ± 0.1074  *(runs: 0.335, 0.470, 0.258)*
- `acc_router` = **0.2412** ± 0.0154  *(runs: 0.231, 0.233, 0.259)*
- `gap_router_minus_fixed` = **-0.1134** ± 0.1191  *(runs: -0.104, -0.237, 0.001)*

**Notes:** Fixed split wins: fixed=0.335, router=0.231, gap=-0.104<-0.03.

---
#### exp_38_2  ✗ REFUTED ⚠ inconsistent across seeds ['REFUTED', 'SUPPORTED', 'REFUTED']
**Hypothesis:** EPI_FRAC=0.25 (25% episodic, 75% semantic) outperforms 50/50 by >3%, indicating semantic capacity dominates for content-association tasks.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 142s

**Metrics (mean ± std across seeds):**

- `acc_25_epi` = **0.3394** ± 0.0736  *(runs: 0.268, 0.415, 0.335)*
- `acc_75_epi` = **0.2962** ± 0.0395  *(runs: 0.259, 0.291, 0.338)*
- `best_acc` = **0.3767** ± 0.0444  *(runs: 0.328, 0.415, 0.387)*
- `best_split` = ['epi050', 'epi025', 'epi050']
- `epi025` = **0.3394** ± 0.0736  *(runs: 0.268, 0.415, 0.335)*
- `epi050` = **0.3363** ± 0.0471  *(runs: 0.328, 0.294, 0.387)*
- `epi075` = **0.2962** ± 0.0395  *(runs: 0.259, 0.291, 0.338)*
- `gap_best_vs_50` = **0.0404** ± 0.0700  *(runs: 0.000, 0.121, 0.000)*

**Notes:** 50/50 optimal: best=epi050(0.328), 50/50=0.328, gap=0.000 ≤ 0.03.

---
#### exp_38_3  ✓ SUPPORTED ⚠ inconsistent across seeds ['REFUTED', 'INCONCLUSIVE', 'SUPPORTED']
**Hypothesis:** Gated read (learned softmax over [M_sem, M_epi] outputs) outperforms simple concatenation by >5%.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 118s

**Metrics (mean ± std across seeds):**

- `acc_concat` = **0.3545** ± 0.1074  *(runs: 0.335, 0.470, 0.258)*
- `acc_gated` = **0.4247** ± 0.1072  *(runs: 0.302, 0.498, 0.474)*
- `gap_gated_minus_concat` = **0.0701** ± 0.1300  *(runs: -0.033, 0.028, 0.216)*

**Notes:** Concat wins: concat=0.335, gated=0.302, gap=-0.033<-0.03.

---

### Category 39 — Write Controller Adaptation (Phase 7)
*0 supported / 1 refuted / 2 inconclusive / 0 error*

#### exp_39_1  ~ INCONCLUSIVE
**Hypothesis:** Accuracy peaks near write_rate ≈ 0.50; forcing rate to 0.10 or 0.90 degrades accuracy by >10% each (concave accuracy vs write-rate curve).

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 249s

**Metrics (mean ± std across seeds):**

- `acc_ts01` = **0.2319** ± 0.0167  *(runs: 0.216, 0.249, 0.230)*
- `acc_ts03` = **0.2150** ± 0.0201  *(runs: 0.233, 0.219, 0.193)*
- `acc_ts05` = **0.2242** ± 0.0175  *(runs: 0.239, 0.228, 0.205)*
- `acc_ts07` = **0.2142** ± 0.0157  *(runs: 0.201, 0.211, 0.231)*
- `acc_ts09` = **0.2179** ± 0.0060  *(runs: 0.212, 0.224, 0.217)*
- `acc_ts11` = **0.0290** ± 0.0064  *(runs: 0.031, 0.034, 0.022)*
- `acc_ts14` = **0.0319** ± 0.0023  *(runs: 0.034, 0.033, 0.029)*
- `drop_at_high_wr` = **0.0000**  *(stable across seeds)*
- `drop_at_low_wr` = **0.2110** ± 0.0035  *(runs: 0.209, 0.215, 0.209)*
- `peak_acc` = **0.2400** ± 0.0091  *(runs: 0.239, 0.249, 0.231)*
- `peak_scale` = **0.4333** ± 0.3055  *(runs: 0.500, 0.100, 0.700)*
- `peak_wr` = **0.5345** ± 0.0266  *(runs: 0.532, 0.562, 0.509)*
- `wr_ts01` = **0.5624** ± 0.0002  *(runs: 0.563, 0.562, 0.562)*
- `wr_ts03` = **0.5471** ± 0.0049  *(runs: 0.553, 0.543, 0.545)*
- `wr_ts05` = **0.5249** ± 0.0061  *(runs: 0.532, 0.522, 0.521)*
- `wr_ts07` = **0.5095** ± 0.0003  *(runs: 0.510, 0.509, 0.509)*
- `wr_ts09` = **0.3998** ± 0.0150  *(runs: 0.405, 0.412, 0.383)*
- `wr_ts11` = **0.0000**  *(stable across seeds)*
- `wr_ts14` = **0.0000**  *(stable across seeds)*

**Notes:** Asymmetric: peak_wr=0.532, low_drop=0.209, high_drop=0.000. One extreme hurts more than other.

---
#### exp_39_2  ~ INCONCLUSIVE
**Hypothesis:** EnergyGatedDelta write rate at equilibrium increases with ρ: wr(ρ=0.75) > wr(ρ=0.08) by >0.15, showing task-load dependence.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 586s

**Metrics (mean ± std across seeds):**

- `acc_n02` = **0.5264** ± 0.0121  *(runs: 0.533, 0.512, 0.534)*
- `acc_n05` = **0.2344** ± 0.0495  *(runs: 0.226, 0.287, 0.190)*
- `acc_n10` = **0.1250** ± 0.0115  *(runs: 0.113, 0.136, 0.126)*
- `acc_n20` = **0.0757** ± 0.0018  *(runs: 0.074, 0.078, 0.075)*
- `acc_n32` = **0.0653** ± 0.0027  *(runs: 0.065, 0.068, 0.063)*
- `acc_n48` = **0.0451** ± 0.0083  *(runs: 0.037, 0.045, 0.054)*
- `delta_wr_high_minus_low` = **0.0769** ± 0.0076  *(runs: 0.083, 0.069, 0.079)*
- `rhos` = [[0.031, 0.078, 0.156, 0.312, 0.5, 0.75], [0.031, 0.078, 0.156, 0.312, 0.5, 0.75], [0.031, 0.078, 0.156, 0.312, 0.5, 0.75]]
- `wr_high_rho` = **0.7772** ± 0.0053  *(runs: 0.779, 0.771, 0.781)*
- `wr_low_rho` = **0.7004** ± 0.0037  *(runs: 0.696, 0.703, 0.702)*
- `wr_n02` = **0.7004** ± 0.0037  *(runs: 0.696, 0.703, 0.702)*
- `wr_n05` = **0.8258** ± 0.0098  *(runs: 0.828, 0.834, 0.815)*
- `wr_n10` = **0.8697** ± 0.0024  *(runs: 0.869, 0.868, 0.872)*
- `wr_n20` = **0.8824** ± 0.0043  *(runs: 0.886, 0.878, 0.883)*
- `wr_n32` = **0.8348** ± 0.0070  *(runs: 0.832, 0.830, 0.843)*
- `wr_n48` = **0.7772** ± 0.0053  *(runs: 0.779, 0.771, 0.781)*
- `wr_ref_rho` = **0.8258** ± 0.0098  *(runs: 0.828, 0.834, 0.815)*

**Notes:** Moderate variation: low=0.696, high=0.779, Δ=0.083. Non-monotone or boundary effects.

---
#### exp_39_3  ✗ REFUTED
**Hypothesis:** Learnable threshold converges to 0.40–0.55 from any initial value; spread of final thresholds < 0.30 across initializations.

**Runs:** 3 (seeds: [123, 42, 777])  |  **Avg duration:** 233s

**Metrics (mean ± std across seeds):**

- `acc_init005` = **0.2348** ± 0.0157  *(runs: 0.241, 0.246, 0.217)*
- `acc_init020` = **0.2486** ± 0.0425  *(runs: 0.221, 0.227, 0.297)*
- `acc_init050` = **0.2133** ± 0.0088  *(runs: 0.223, 0.206, 0.211)*
- `acc_init080` = **0.2198** ± 0.0110  *(runs: 0.219, 0.231, 0.209)*
- `acc_init120` = **0.2273** ± 0.0112  *(runs: 0.235, 0.233, 0.214)*
- `final_thresh_init005` = **0.0486** ± 0.0009  *(runs: 0.049, 0.048, 0.049)*
- `final_thresh_init020` = **0.1869** ± 0.0024  *(runs: 0.188, 0.184, 0.188)*
- `final_thresh_init050` = **0.4801** ± 0.0150  *(runs: 0.464, 0.482, 0.494)*
- `final_thresh_init080` = **0.8087** ± 0.0047  *(runs: 0.814, 0.804, 0.808)*
- `final_thresh_init120` = **1.0683** ± 0.0032  *(runs: 1.072, 1.067, 1.066)*
- `final_thresholds` = [[0.0494, 0.1884, 0.4642, 0.8137, 1.0718], [0.0477, 0.1841, 0.4823, 0.8044, 1.0673], [0.0488, 0.1881, 0.4939, 0.808, 1.0657]]
- `final_wr_init005` = **0.5638** ± 0.0004  *(runs: 0.564, 0.564, 0.563)*
- `final_wr_init020` = **0.5584** ± 0.0022  *(runs: 0.556, 0.558, 0.561)*
- `final_wr_init050` = **0.5255** ± 0.0017  *(runs: 0.525, 0.527, 0.524)*
- `final_wr_init080` = **0.4869** ± 0.0116  *(runs: 0.481, 0.500, 0.479)*
- `final_wr_init120` = **0.0000**  *(stable across seeds)*
- `frac_in_attractor_40_55` = **0.2000**  *(stable across seeds)*
- `mean_thresh` = **0.5185** ± 0.0021  *(runs: 0.517, 0.517, 0.521)*
- `spread` = **1.0197** ± 0.0027  *(runs: 1.022, 1.020, 1.017)*

**Notes:** No convergence: spread=1.022>0.50. Thresholds: [0.049, 0.188, 0.464, 0.814, 1.072]. Initial value matters; no attractor at 0.54.

---

## Cross-Cutting Observations

**All SUPPORTED experiments:** exp_1_5, exp_2_6, exp_3_1, exp_3_5, exp_3_6, exp_4_1, exp_4_4, exp_4_7, exp_4_9, exp_5_2, exp_5_6, exp_6_1, exp_6_3, exp_7_1, exp_7_2, exp_7_9, exp_9_2, exp_9_4, exp_9_5, exp_11_3, exp_13_1, exp_13_2, exp_15_3, exp_16_3, exp_23_2, exp_23_3, exp_24_2, exp_25_1, exp_26_1, exp_29_1, exp_29_3, exp_30_1, exp_32_1, exp_32_2, exp_32_3, exp_32_4, exp_33_4, exp_34_6, exp_35_2, exp_35_3, exp_36_3, exp_37_3, exp_38_3

**All REFUTED experiments:** exp_1_1, exp_1_2, exp_1_8, exp_2_2, exp_2_4, exp_2_5, exp_2_9, exp_3_2, exp_3_3, exp_3_4, exp_3_7, exp_4_2, exp_5_1, exp_5_4, exp_5_5, exp_5_7, exp_6_7, exp_7_5, exp_7_6, exp_7_7, exp_8_1, exp_8_3, exp_9_1, exp_10_2, exp_11_1, exp_11_2, exp_12_1, exp_14_1, exp_14_2, exp_14_3, exp_15_1, exp_15_2, exp_16_1, exp_17_1, exp_17_2, exp_17_4, exp_18_1, exp_19_1, exp_19_2, exp_19_3, exp_20_2, exp_20_3, exp_21_1, exp_21_2, exp_21_3, exp_21_4, exp_22_1, exp_22_2, exp_22_4, exp_22_5, exp_23_4, exp_24_3, exp_24_4, exp_25_2, exp_25_3, exp_26_2, exp_26_3, exp_27_1, exp_27_3, exp_28_5, exp_30_2, exp_30_3, exp_31_1, exp_31_2, exp_33_1, exp_33_3, exp_34_1, exp_34_5, exp_34_7, exp_34_8, exp_34_9, exp_35_1, exp_36_1, exp_38_1, exp_38_2, exp_39_3

**Inconsistent across seeds (need more investigation):** exp_8_1, exp_8_2, exp_8_4, exp_9_4, exp_9_5, exp_10_1, exp_15_1, exp_15_2, exp_15_4, exp_17_1, exp_17_2, exp_17_4, exp_18_3, exp_22_1, exp_22_2, exp_22_4, exp_22_5, exp_23_1, exp_23_2, exp_23_3, exp_23_4, exp_24_1, exp_25_2, exp_25_3, exp_26_1, exp_26_2, exp_26_3, exp_28_2, exp_28_4, exp_29_2, exp_29_3, exp_29_4, exp_30_1, exp_30_3, exp_30_4, exp_31_2, exp_32_1, exp_32_2, exp_32_3, exp_32_4, exp_33_2, exp_33_4, exp_34_6, exp_35_3, exp_38_1, exp_38_2, exp_38_3

**High-variance metrics (std > 0.05 — seed-sensitive, interpret carefully):**

- exp_8_1.pearson_r_entropic (std=0.446)
- exp_8_1.pearson_r_raw (std=0.341)
- exp_8_1.pearson_r_softmax (std=0.488)
- exp_8_2.pearson_r_difficulty_vs_rate (std=0.405)
- exp_8_3.acc_A (std=0.051)
- exp_8_3.write_evict_corr_A (std=0.288)
- exp_8_3.write_evict_corr_B (std=0.062)
- exp_8_3.write_evict_corr_C (std=0.303)
- exp_8_4.acc_A (std=0.110)
- exp_8_4.acc_B (std=0.058)
- exp_8_4.content_corr_A (std=0.525)
- exp_8_4.content_corr_B (std=0.089)
- exp_8_4.content_corr_C (std=0.169)
- exp_8_4.pos_corr_A (std=0.336)
- exp_8_4.pos_corr_B (std=0.079)
- exp_8_4.pos_corr_C (std=0.169)
- exp_9_3.acc_a_after_ewc (std=0.066)
- exp_9_3.acc_a_after_std (std=0.050)
- exp_9_3.acc_a_before (std=0.085)
- exp_9_3.forgetting_ratio (std=0.611)

---
*Report generated by research/aggregate.py*