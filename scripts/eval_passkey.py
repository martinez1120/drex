#!/usr/bin/env python3
"""
scripts/eval_passkey.py — Passkey recall sweep across context lengths.

Measures the fraction of trials where the model correctly recalls a 5-digit
passkey embedded in a long filler sequence. Sweeps from 2k to 32k tokens.

Usage (random-init baseline — expected ~0% accuracy):
    python scripts/eval_passkey.py

Evaluate a trained checkpoint:
    python scripts/eval_passkey.py --checkpoint checkpoints/step_0050000_final.safetensors

Custom sweep:
    python scripts/eval_passkey.py --lengths 2048 4096 8192 --trials 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from drex.eval.passkey import PasskeyBenchmark
from drex.models.transformer import DrexConfig, DrexTransformer
from drex.utils.config import load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(args: argparse.Namespace, device: torch.device) -> DrexTransformer:
    """Build a DrexTransformer, optionally loading weights from a checkpoint."""
    config = DrexConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        vocab_size=256,
        window_size=args.window_size,
        max_seq_len=max(args.lengths) + args.window_size,
        dropout=0.0,
        use_l3=args.use_l3,
        l3_base_path=args.l3_path,
    )
    model = DrexTransformer(config).to(device)

    if args.checkpoint:
        step = load_checkpoint(model, args.checkpoint)
        print(f"Loaded checkpoint from '{args.checkpoint}' (step {step})", flush=True)
    else:
        print("No checkpoint supplied — evaluating randomly-initialised model.", flush=True)

    model.eval()
    return model


def _print_table(results: dict[int, dict[int, float]]) -> None:
    """Print a markdown-style table of accuracy by context length and layer count."""
    lengths = sorted(next(iter(results.values())).keys())
    header = f"{'Config':<30}" + "".join(f"  {l//1024:>4}k" for l in lengths)
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for label, row in results.items():
        vals = "".join(f"  {row[l]:>5.1%}" for l in lengths)
        print(f"{str(label):<30}{vals}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> None:
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    model = _make_model(args, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)
    print(f"Context lengths: {args.lengths}", flush=True)
    print(f"Trials per length: {args.trials}", flush=True)
    print()

    bench = PasskeyBenchmark(
        model=model,
        context_lengths=args.lengths,
        n_trials=args.trials,
        device=str(device),
        segment_len=args.window_size,
    )

    print("Running passkey recall sweep …", flush=True)
    results = bench.run()

    # Display
    label = Path(args.checkpoint).stem if args.checkpoint else "random-init"
    lengths = sorted(results)
    header = f"{'model':<40}" + "".join(f"  {l//1024:>4}k" for l in lengths)
    sep = "-" * len(header)
    vals = "".join(f"  {results[l]:>5.1%}" for l in lengths)

    print(sep)
    print(header)
    print(sep)
    print(f"{label:<40}{vals}")
    print(sep)

    # Summary
    avg = sum(results.values()) / len(results)
    best_len = max(results, key=results.get)
    print(f"\nAverage accuracy : {avg:.1%}")
    print(f"Best accuracy    : {results[best_len]:.1%}  (context length {best_len:,})")

    if avg < 0.05:
        print("\n[note] Accuracy is near chance — the model has not been trained yet.")
        print("       Run scripts/train.py first, then re-evaluate with --checkpoint.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Passkey recall sweep across context lengths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evaluation
    p.add_argument(
        "--lengths", type=int, nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
        metavar="N",
        help="Context lengths to evaluate (space-separated token counts)",
    )
    p.add_argument("--trials", type=int, default=10,
                   help="Number of independent passkey trials per context length")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a .safetensors checkpoint (omit for random-init baseline)")

    # Model architecture (must match checkpoint if loading one)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--window-size", type=int, default=512,
                   help="Sliding-window size (must match training segment length)")

    # L3
    p.add_argument("--use-l3", action="store_true")
    p.add_argument("--l3-path", type=str, default="/tmp/drex_l3")

    # Infrastructure
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])

    return p


if __name__ == "__main__":
    run_eval(_parser().parse_args())
