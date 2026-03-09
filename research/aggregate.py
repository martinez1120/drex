"""
aggregate.py — Aggregate multi-seed results and generate master report.

Reads all JSON files from results/, groups by experiment_id, computes
mean ± std for numeric metrics across seeds, determines consensus outcome,
and writes MASTER_REPORT.md.

Usage:
    python3 aggregate.py
    python3 aggregate.py --results-dir results/  # explicit path
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESEARCH_DIR = Path(__file__).parent
RESULTS_DIR  = RESEARCH_DIR / "results"
OUTPUT_PATH  = RESEARCH_DIR / "MASTER_REPORT.md"

OUTCOME_ORDER = {"SUPPORTED": 0, "INCONCLUSIVE": 1, "REFUTED": 2, "ERROR": 3}
OUTCOME_EMOJI = {
    "SUPPORTED":    "✓",
    "REFUTED":      "✗",
    "INCONCLUSIVE": "~",
    "ERROR":        "!",
}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all JSON result files, grouped by experiment_id."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            eid = data.get("experiment_id", "unknown")
            groups[eid].append(data)
        except Exception as e:
            print(f"Warning: could not read {path.name}: {e}")
    return dict(groups)


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_metrics(runs: list[dict]) -> dict[str, Any]:
    """
    For each numeric metric, compute mean ± std across runs.
    For non-numeric / mixed, collect unique values.
    Returns aggregated dict.
    """
    if not runs:
        return {}

    # Collect all keys across runs
    all_keys: set[str] = set()
    for run in runs:
        all_keys.update(_flatten_metrics(run.get("metrics", {})).keys())

    agg: dict[str, Any] = {}
    for key in sorted(all_keys):
        values = []
        for run in runs:
            flat = _flatten_metrics(run.get("metrics", {}))
            if key in flat and flat[key] is not None:
                values.append(flat[key])

        if not values:
            continue

        # All numeric?
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric) == len(values):
            mean = statistics.mean(numeric)
            std  = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
            agg[key] = {"mean": round(mean, 4), "std": round(std, 4), "values": numeric}
        else:
            agg[key] = {"values": values}

    return agg


def _flatten_metrics(d: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested metric dicts with dot-notation keys."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten_metrics(v, full_key))
            else:
                out[full_key] = v
    return out


def consensus_outcome(outcomes: list[str]) -> tuple[str, bool]:
    """Return (majority_outcome, is_consistent).
    Consistent = all outcomes are identical."""
    if not outcomes:
        return "ERROR", False
    counts: dict[str, int] = defaultdict(int)
    for o in outcomes:
        counts[o] += 1
    majority = max(counts, key=lambda k: (counts[k], -OUTCOME_ORDER.get(k, 99)))
    consistent = len(set(outcomes)) == 1
    return majority, consistent


def summarise_experiment(eid: str, runs: list[dict]) -> dict[str, Any]:
    outcomes = [r.get("outcome", "ERROR") for r in runs]
    majority, consistent = consensus_outcome(outcomes)
    agg_metrics = aggregate_metrics(runs)
    notes_all = [r.get("notes", "") for r in runs if r.get("notes")]
    hypothesis = runs[0].get("hypothesis", "") if runs else ""
    durations = [r.get("duration_seconds", 0) for r in runs]

    return {
        "experiment_id":  eid,
        "hypothesis":     hypothesis,
        "outcome":        majority,
        "consistent":     consistent,
        "outcomes_per_seed": outcomes,
        "seeds":          [r.get("seed", -1) for r in runs],
        "n_runs":         len(runs),
        "metrics":        agg_metrics,
        "notes_sample":   notes_all[0] if notes_all else "",
        "mean_duration":  round(statistics.mean(durations), 1) if durations else 0,
    }


# ── Markdown report ───────────────────────────────────────────────────────────

CATEGORY_NAMES = {
    "exp_1":  "Category 1 — What To Write",
    "exp_2":  "Category 2 — How To Write (Compression)",
    "exp_3":  "Category 3 — When To Write",
    "exp_4":  "Category 4 — What To Read",
    "exp_5":  "Category 5 — When To Read",
    "exp_6":  "Category 6 — How To Forget",
    "exp_7":  "Category 7 — Cross-Cutting",
    "exp_8":  "Category 8 — Mechanistic Investigations (Phase 2)",
    "exp_9":  "Category 9 — Inconclusive Redesigns (Phase 2)",
    "exp_10": "Category 10 — Retroactive Writing Mechanism (Phase 2)",
    "exp_11": "Category 11 — Read Bottleneck Interventions (Phase 2)",
    "exp_12": "Category 12 — Compression Hard Regimes (Phase 2)",
    "exp_13": "Category 13 — Compositional Retrieval at Scale (Phase 2)",
    "exp_14": "Category 14 — System Integration (Phase 2)",
    "exp_15": "Category 15 — Delta Rule / Associative Matrix Writes (Phase 3)",
    "exp_16": "Category 16 — Online Gradient Descent Memory / Titans-Style (Phase 3)",
    "exp_17": "Category 17 — Prospective / Query-Conditioned Writing (Phase 3)",
    "exp_18": "Category 18 — Tiered Memory Architecture (Phase 3)",
    "exp_19": "Category 19 — Sparse Hopfield Addressing (Phase 3)",
    "exp_20": "Category 20 — Three-Gate Coordinated Controller (Phase 3)",
    "exp_21": "Category 21 — Feedforward Controller + Hindsight Distillation (Phase 3)",
    "exp_22": "Category 22 — Read Architecture Redesigns (Phase 4)",
    "exp_23": "Category 23 — Retroactive Re-Encoding Variants (Phase 4)",
    "exp_24": "Category 24 — Scale and Length Generalization (Phase 4)",
    "exp_25": "Category 25 — Hard Benchmarks (Phase 4)",
    "exp_26": "Category 26 — Seed Stability Validation (Phase 4)",
    "exp_27": "Category 27 — Parametric-Delta Hybrid (Phase 4)",
    "exp_28": "Category 28 — Explicit Scaling Laws (Phase 5)",
    "exp_29": "Category 29 — TTT / Titans-Inspired Memory (Phase 5)",
    "exp_30": "Category 30 — Multi-Head & Extended Delta Rule (Phase 5)",
    "exp_31": "Category 31 — Top Mechanism Integration (Phase 5)",
    "exp_32": "Category 32 — Deep Seed Validation (Phase 5)",
    "exp_33": "Category 33 — Capacity Physics / Interference Density Law (Phase 5)",
    "exp_34": "Category 34 — Training Dynamics (Phase 6)",
    "exp_35": "Category 35 — Failure Modes (Phase 6)",
    "exp_36": "Category 36 — Biological Analogues (Phase 6)",
    "exp_37": "Category 37 — Robustness (Phase 7)",
    "exp_38": "Category 38 — Episodic/Semantic Architecture (Phase 7)",
    "exp_39": "Category 39 — Write Controller Adaptation (Phase 7)",
}


def sort_key(eid: str) -> tuple[int, int]:
    """Sort by category then experiment number."""
    parts = eid.replace("exp_", "").split("_")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 99, 99


def format_metric_row(key: str, val: Any) -> str:
    if isinstance(val, dict) and "mean" in val:
        std = val["std"]
        vals_str = ", ".join(f"{v:.3f}" for v in val.get("values", []))
        if std > 0:
            return f"`{key}` = **{val['mean']:.4f}** ± {std:.4f}  *(runs: {vals_str})*"
        return f"`{key}` = **{val['mean']:.4f}**  *(stable across seeds)*"
    elif isinstance(val, dict) and "values" in val:
        vals = val["values"]
        return f"`{key}` = {vals!r}"
    return f"`{key}` = {val!r}"


def generate_report(summaries: list[dict[str, Any]]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_total = len(summaries)
    n_supported    = sum(1 for s in summaries if s["outcome"] == "SUPPORTED")
    n_refuted      = sum(1 for s in summaries if s["outcome"] == "REFUTED")
    n_inconclusive = sum(1 for s in summaries if s["outcome"] == "INCONCLUSIVE")
    n_error        = sum(1 for s in summaries if s["outcome"] == "ERROR")
    n_inconsistent = sum(1 for s in summaries if not s["consistent"])

    lines: list[str] = []
    lines.append("# drex Research — Master Results Report")
    lines.append(f"\n**Generated:** {now}")
    lines.append(f"**Experiments:** {n_total}  |  "
                 f"**Seeds per experiment:** {', '.join(str(s) for s in sorted({seed for sm in summaries for seed in sm['seeds']}) ) or '—'}")
    lines.append(f"**Total runs evaluated:** {sum(s['n_runs'] for s in summaries)}")
    lines.append("")

    # Overall scoreboard
    lines.append("## Overall Scoreboard\n")
    lines.append(f"| Outcome | Count | % |")
    lines.append(f"|---------|-------|---|")
    lines.append(f"| ✓ SUPPORTED    | {n_supported}    | {100*n_supported/max(n_total,1):.0f}% |")
    lines.append(f"| ~ INCONCLUSIVE | {n_inconclusive} | {100*n_inconclusive/max(n_total,1):.0f}% |")
    lines.append(f"| ✗ REFUTED      | {n_refuted}      | {100*n_refuted/max(n_total,1):.0f}% |")
    lines.append(f"| ! ERROR        | {n_error}        | {100*n_error/max(n_total,1):.0f}% |")
    lines.append(f"\n**Seed consistency:** {n_total - n_inconsistent}/{n_total} experiments gave the same verdict across all seeds. "
                 f"{n_inconsistent} inconsistent.")

    # Summary table
    lines.append("\n## Summary Table\n")
    lines.append("| ID | Outcome | Consistent | Key Metric (mean ± std) | Notes |")
    lines.append("|----|---------|------------|------------------------|-------|")
    for s in summaries:
        icon = OUTCOME_EMOJI.get(s["outcome"], "?")
        cons = "✓" if s["consistent"] else f"⚠ {s['outcomes_per_seed']}"
        # pick one representative numeric metric
        key_metric = ""
        for k, v in s["metrics"].items():
            if isinstance(v, dict) and "mean" in v:
                key_metric = f"{k}={v['mean']:.3f}±{v['std']:.3f}"
                break
        lines.append(
            f"| {s['experiment_id']} | {icon} {s['outcome']} | {cons} "
            f"| {key_metric} | {s['notes_sample'][:60]}… |"
        )

    # Per-category detailed sections
    lines.append("\n---\n")
    lines.append("## Detailed Results by Category")

    # Group by category prefix
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        cat_key = "_".join(s["experiment_id"].split("_")[:2])   # "exp_1"
        by_cat[cat_key].append(s)

    for cat_key in sorted(by_cat.keys(), key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 99):
        cat_label = CATEGORY_NAMES.get(cat_key, cat_key)
        cat_summaries = sorted(by_cat[cat_key], key=lambda s: sort_key(s["experiment_id"]))

        cat_sup = sum(1 for s in cat_summaries if s["outcome"] == "SUPPORTED")
        cat_ref = sum(1 for s in cat_summaries if s["outcome"] == "REFUTED")
        cat_inc = sum(1 for s in cat_summaries if s["outcome"] == "INCONCLUSIVE")
        cat_err = sum(1 for s in cat_summaries if s["outcome"] == "ERROR")

        lines.append(f"\n### {cat_label}")
        lines.append(f"*{cat_sup} supported / {cat_ref} refuted / {cat_inc} inconclusive / {cat_err} error*\n")

        for s in cat_summaries:
            icon = OUTCOME_EMOJI.get(s["outcome"], "?")
            cons_note = "" if s["consistent"] else f" ⚠ inconsistent across seeds {s['outcomes_per_seed']}"
            lines.append(f"#### {s['experiment_id']}  {icon} {s['outcome']}{cons_note}")
            lines.append(f"**Hypothesis:** {s['hypothesis']}\n")
            lines.append(f"**Runs:** {s['n_runs']} (seeds: {s['seeds']})  |  "
                         f"**Avg duration:** {s['mean_duration']:.0f}s\n")

            if s["metrics"]:
                lines.append("**Metrics (mean ± std across seeds):**\n")
                for k, v in s["metrics"].items():
                    lines.append(f"- {format_metric_row(k, v)}")
                lines.append("")

            if s["notes_sample"]:
                lines.append(f"**Notes:** {s['notes_sample']}\n")

            lines.append("---")

    # Cross-cutting observations section (high-level)
    supported_ids = [s["experiment_id"] for s in summaries if s["outcome"] == "SUPPORTED"]
    refuted_ids   = [s["experiment_id"] for s in summaries if s["outcome"] == "REFUTED"]
    incons_ids    = [s["experiment_id"] for s in summaries if not s["consistent"]]

    lines.append("\n## Cross-Cutting Observations\n")
    lines.append(f"**All SUPPORTED experiments:** {', '.join(supported_ids) or 'none'}\n")
    lines.append(f"**All REFUTED experiments:** {', '.join(refuted_ids) or 'none'}\n")
    lines.append(f"**Inconsistent across seeds (need more investigation):** "
                 f"{', '.join(incons_ids) or 'none — all experiments were seed-stable'}\n")

    # Flag high-variance metrics
    high_var = []
    for s in summaries:
        for k, v in s["metrics"].items():
            if isinstance(v, dict) and "std" in v and v["std"] > 0.05 and "mean" in v:
                high_var.append(f"{s['experiment_id']}.{k} (std={v['std']:.3f})")
    if high_var:
        lines.append("**High-variance metrics (std > 0.05 — seed-sensitive, interpret carefully):**\n")
        for h in high_var[:20]:
            lines.append(f"- {h}")
        lines.append("")

    lines.append("---")
    lines.append("*Report generated by research/aggregate.py*")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(results_dir: Path = RESULTS_DIR, output: Path = OUTPUT_PATH) -> None:
    print(f"Loading results from {results_dir}...")
    groups = load_results(results_dir)
    print(f"Found {len(groups)} experiment IDs across "
          f"{sum(len(v) for v in groups.values())} result files.")

    summaries = []
    for eid, runs in sorted(groups.items(), key=lambda kv: sort_key(kv[0])):
        s = summarise_experiment(eid, runs)
        summaries.append(s)
        icon = OUTCOME_EMOJI.get(s["outcome"], "?")
        cons = "stable" if s["consistent"] else f"inconsistent {s['outcomes_per_seed']}"
        print(f"  {eid:20s}  {icon} {s['outcome']:15s}  {cons}  (n={s['n_runs']})")

    report = generate_report(summaries)
    output.write_text(report, encoding="utf-8")
    print(f"\nMaster report written to: {output.relative_to(RESEARCH_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate drex experiment results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    main(results_dir=args.results_dir, output=args.output)
