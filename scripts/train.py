#!/usr/bin/env python3
"""
scripts/train.py — DrexTransformer training on TinyStories.

Usage (quick smoke-run, ~10 min on CPU/MPS):
    python scripts/train.py --steps 1000 --log-every 100

Full run (MPS, ~2 h):
    python scripts/train.py --steps 50000 --lr 3e-4 --use-l3

Resume from checkpoint:
    python scripts/train.py --resume checkpoints/step_1000.safetensors --steps 5000

Training data:
    First run downloads roneneldan/TinyStories (~400MB) via HuggingFace datasets.
    Subsequent runs use the cached copy in ~/.cache/huggingface.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from drex.models.transformer import DrexConfig, DrexTransformer
from drex.training.data import SegmentDataset, collate_fn, tokenize_chars
from drex.training.optimizer import build_optimizer, cosine_schedule_with_warmup
from drex.utils.config import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_tinystories(split: str = "train", max_chars: int | None = None) -> str:
    """Download and concatenate TinyStories text. Returns a single long string."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading TinyStories [{split}] …", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    chunks: list[str] = []
    total = 0
    for row in ds:
        text: str = row["text"] + "\n"
        chunks.append(text)
        total += len(text)
        if max_chars is not None and total >= max_chars:
            break
    print(f"  Loaded {total:,} chars ({len(chunks):,} stories)", flush=True)
    return "".join(chunks)


def _make_dataset(
    text: str,
    segment_len: int,
    stride: int,
) -> SegmentDataset:
    tokens = tokenize_chars(text)
    return SegmentDataset(tokens, segment_len=segment_len, stride=stride)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # ── device ──────────────────────────────────────────────────────────────
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

    # ── data ────────────────────────────────────────────────────────────────
    raw_text = _load_tinystories(
        split="train",
        max_chars=args.max_chars,
    )
    segment_len = args.segment_len
    dataset = _make_dataset(raw_text, segment_len=segment_len, stride=segment_len)
    print(f"Dataset: {len(dataset):,} segments of {segment_len} tokens", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,          # MPS requires num_workers=0
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # ── model ───────────────────────────────────────────────────────────────
    config = DrexConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        vocab_size=256,                 # character-level (byte alphabet)
        window_size=args.segment_len,
        max_seq_len=args.segment_len * 8,
        dropout=args.dropout,
        gradient_checkpointing=args.grad_ckpt,
        use_l3=args.use_l3,
        l3_base_path=args.l3_path,
        l3_compress=args.l3_compress,
    )
    model = DrexTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters", flush=True)

    # ── optimizer + schedule ────────────────────────────────────────────────
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    # DataLoader steps per epoch (approximate — streaming data may vary)
    total_steps = args.steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1,
    )

    # ── resume ──────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume:
        global_step = load_checkpoint(model, args.resume)
        print(f"Resumed from step {global_step}", flush=True)

    # ── checkpoint dir ──────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── training ────────────────────────────────────────────────────────────
    model.train()
    states = model.init_states(args.batch_size, device)

    running_loss = 0.0
    running_n = 0
    t0 = time.perf_counter()

    def _iter_loader():
        """Yield batches indefinitely, re-shuffling each epoch."""
        while True:
            yield from loader

    for src, tgt in _iter_loader():
        if global_step >= total_steps:
            break

        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, states = model(src, states)
        states = [s.detach() for s in states]   # TBPTT boundary

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            tgt.reshape(-1),
        )
        loss.backward()

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        global_step += 1

        running_loss += loss.item()
        running_n += 1

        # ── logging ─────────────────────────────────────────────────────
        if global_step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = running_loss / running_n
            ppl = math.exp(avg_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            tokens_per_sec = (running_n * args.batch_size * segment_len) / elapsed
            print(
                f"step {global_step:>6}  loss {avg_loss:.4f}  ppl {ppl:7.2f}"
                f"  lr {lr_now:.2e}  {tokens_per_sec:,.0f} tok/s",
                flush=True,
            )
            running_loss = 0.0
            running_n = 0
            t0 = time.perf_counter()

        # ── checkpointing ────────────────────────────────────────────────
        if args.save_every > 0 and global_step % args.save_every == 0:
            ckpt_path = ckpt_dir / f"step_{global_step:07d}.safetensors"
            save_checkpoint(model, ckpt_path, step=global_step)
            print(f"  Checkpoint saved → {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = ckpt_dir / f"step_{global_step:07d}_final.safetensors"
    save_checkpoint(model, final_path, step=global_step)
    print(f"\nTraining complete. Final checkpoint → {final_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train DrexTransformer on TinyStories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--max-chars", type=int, default=50_000_000,
                   help="Maximum characters to load from TinyStories (~50M ≈ full train split)")
    p.add_argument("--segment-len", type=int, default=512,
                   help="Tokens per training segment (TBPTT chunk length)")
    p.add_argument("--batch-size", type=int, default=8)

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-ckpt", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM at cost of speed")

    # Optimizer
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # L3
    p.add_argument("--use-l3", action="store_true",
                   help="Enable L3 Rust disk cache (requires maturin develop)")
    p.add_argument("--l3-path", type=str, default="/tmp/drex_l3")
    p.add_argument("--l3-compress", action="store_true")

    # Infrastructure
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a .safetensors checkpoint to resume from")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=2000,
                   help="Save checkpoint every N steps (0 = only final)")
    p.add_argument("--log-every", type=int, default=100)

    return p


if __name__ == "__main__":
    train(_parser().parse_args())
