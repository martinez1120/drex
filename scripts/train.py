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

from drex.models.memory import (
    LayerState,
    MemoryModule,
    MemoryState,
    WRITE_RATE_LO,
    WRITE_RATE_HI,
)
from drex.models.transformer import DrexConfig, DrexTransformer
from drex.training.data import SegmentDataset, collate_fn, tokenize_chars
from drex.training.optimizer import build_optimizer, cosine_schedule_with_warmup
from drex.utils.config import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# ord('\n') in the char-level vocabulary used by tokenize_chars.
# Story boundaries in TinyStories are represented as this token.
_NEWLINE_TOKEN: int = 10


def _patch_ssl_for_download() -> None:
    """
    Disable SSL certificate verification for the current process.

    Use only in development environments where HuggingFace Hub connections
    fail due to a corporate proxy performing SSL inspection.  Do not use
    in production or CI; fix the certificate bundle instead.
    """
    import ssl as _ssl

    import httpx as _httpx

    # Patch stdlib ssl context (used by urllib/requests)
    _unverified = _ssl._create_unverified_context
    _ssl.create_default_context = lambda *a, **kw: _unverified()
    _ssl._create_default_https_context = _unverified

    # Patch httpx (used by huggingface_hub ≥0.20)
    _OrigClient = _httpx.Client

    class _NoVerifyClient(_OrigClient):
        def __init__(self, *args, **kwargs):
            kwargs["verify"] = False
            super().__init__(*args, **kwargs)

    _OrigAsyncClient = _httpx.AsyncClient

    class _NoVerifyAsyncClient(_OrigAsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["verify"] = False
            super().__init__(*args, **kwargs)

    _httpx.Client = _NoVerifyClient  # type: ignore[misc]
    _httpx.AsyncClient = _NoVerifyAsyncClient  # type: ignore[misc]

    print(
        "  [WARNING] SSL certificate verification disabled (--no-ssl-verify).",
        file=sys.stderr,
        flush=True,
    )


def _load_tinystories(
    split: str = "train",
    max_chars: int | None = None,
    data_file: str | None = None,
    no_ssl_verify: bool = False,
) -> str:
    """
    Load TinyStories text for the given split.

    Priority:
      1. ``data_file`` — path to a local text file (newline-separated stories).
         Use this to skip the HuggingFace download entirely.
      2. HuggingFace Hub (roneneldan/TinyStories, streaming mode).
         Set ``no_ssl_verify=True`` if your network proxy intercepts SSL.

    Returns a single concatenated string of story text.
    """
    if data_file is not None:
        print(f"Loading TinyStories [{split}] from {data_file} …", flush=True)
        with open(data_file, encoding="utf-8") as fh:
            text = fh.read()
        if max_chars is not None:
            text = text[:max_chars]
        n_stories = text.count("\n")
        print(f"  Loaded {len(text):,} chars (~{n_stories:,} stories)", flush=True)
        return text

    if no_ssl_verify:
        _patch_ssl_for_download()

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
# Validation helper
# ---------------------------------------------------------------------------


def _validate(
    model: DrexTransformer,
    val_loader: DataLoader,
    config: DrexConfig,
    device: torch.device,
) -> float:
    """
    Compute average cross-entropy loss over the validation set.

    Each batch uses fresh zero states (no TBPTT threading across shuffled
    document boundaries).  Returns average loss as a float.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            logits, _ = model(src)   # states=None → model.init_states() called internally
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                tgt.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def _reset_boundary_states(
    states: list,
    tgt: torch.Tensor,
    model: DrexTransformer,
    device: torch.device,
) -> list:
    """
    Zero LayerState for batch elements whose target segment contains a story
    boundary (token _NEWLINE_TOKEN = ord('\\n') = 10).

    Prevents L2 Infini-Attention memory from threading across unrelated
    TinyStories documents when --reset-on-boundary is enabled.
    """
    has_bnd = (tgt == _NEWLINE_TOKEN).any(dim=1)   # (B,) bool
    if not has_bnd.any():
        return states
    fresh = model.init_states(tgt.size(0), device)
    new_states: list = []
    for s, f in zip(states, fresh):
        # Zero out M and z only for batch elements that crossed a boundary.
        M_new = torch.where(has_bnd[:, None, None, None], f.memory.M, s.memory.M)
        z_new = torch.where(has_bnd[:, None, None], f.memory.z, s.memory.z)
        new_states.append(
            LayerState(memory=MemoryState(M=M_new, z=z_new), step=s.step)
        )
    return new_states


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
        data_file=args.data_file,
        no_ssl_verify=args.no_ssl_verify,
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

    # ── validation data (optional) ──────────────────────────────────────────
    val_loader: DataLoader | None = None
    if args.val_every > 0:
        val_text = _load_tinystories(
            split="validation",
            max_chars=args.val_max_chars,
            no_ssl_verify=args.no_ssl_verify,
        )
        val_dataset = _make_dataset(val_text, segment_len=segment_len, stride=segment_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,          # deterministic; no TBPTT threading needed
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        print(f"  Validation: {len(val_dataset):,} segments of {segment_len} tokens",
              flush=True)

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
        use_episodic_memory=args.use_episodic_memory,
        episodic_gate_thresh=args.episodic_gate_thresh,
        use_null_gate=not args.no_null_gate,
        full_seq_residual=args.full_seq_residual,
        memory_last_layer_only=args.memory_last_layer_only,
    )
    model = DrexTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters", flush=True)
    if config.use_episodic_memory:
        flags = []
        if not config.use_null_gate:
            flags.append("no-null-gate")
        if config.full_seq_residual:
            flags.append("full-seq-residual")
        if config.memory_last_layer_only:
            flags.append("last-layer-only")
        flag_str = "  [" + ", ".join(flags) + "]" if flags else ""
        print(
            f"Episodic memory: enabled  gate_thresh={config.episodic_gate_thresh}{flag_str}",
            flush=True,
        )

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

    # Collect MemoryModule instances for write-rate monitoring.
    # Empty list when use_episodic_memory=False → all write-rate code is a no-op.
    _mem_modules: list[MemoryModule] = [
        m for m in model.modules() if isinstance(m, MemoryModule)
    ]

    running_loss = 0.0
    running_n = 0
    running_wr_sum = 0.0
    running_wr_min = 1.0
    running_wr_max = 0.0
    running_wr_n = 0
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
        if args.reset_on_boundary:
            states = _reset_boundary_states(states, tgt, model, device)

        # Collect per-step write rates (no-op when use_episodic_memory=False)
        if _mem_modules:
            step_rates = [m.last_write_rate() for m in _mem_modules]
            step_mean = sum(step_rates) / len(step_rates)
            running_wr_sum += step_mean
            running_wr_min = min(running_wr_min, min(step_rates))
            running_wr_max = max(running_wr_max, max(step_rates))
            running_wr_n += 1

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            tgt.reshape(-1),
        )
        if not loss.isfinite():
            print(
                f"  [skip] step {global_step}  non-finite loss={loss.item():.4g}"
                " — zeroing grads and resetting states",
                flush=True,
            )
            optimizer.zero_grad()
            global_step += 1
            # Reset TBPTT state to prevent NaN from propagating through future steps.
            states = model.init_states(args.batch_size, device)
            continue
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

            wr_suffix = ""
            if _mem_modules and running_wr_n > 0:
                avg_wr = running_wr_sum / running_wr_n
                wr_suffix = (
                    f"  wr {avg_wr:.3f}"
                    f" [{running_wr_min:.3f},{running_wr_max:.3f}]"
                )
                if avg_wr < WRITE_RATE_LO or avg_wr > WRITE_RATE_HI:
                    wr_suffix += (
                        f"  [WARNING: write rate outside"
                        f" [{WRITE_RATE_LO},{WRITE_RATE_HI}]]"
                    )

            print(
                f"step {global_step:>6}  loss {avg_loss:.4f}  ppl {ppl:7.2f}"
                f"  lr {lr_now:.2e}  {tokens_per_sec:,.0f} tok/s" + wr_suffix,
                flush=True,
            )
            running_loss = 0.0
            running_n = 0
            running_wr_sum = 0.0
            running_wr_min = 1.0
            running_wr_max = 0.0
            running_wr_n = 0
            t0 = time.perf_counter()

        # ── checkpointing ────────────────────────────────────────────────
        if args.save_every > 0 and global_step % args.save_every == 0:
            ckpt_path = ckpt_dir / f"step_{global_step:07d}.safetensors"
            save_checkpoint(model, ckpt_path, step=global_step)
            print(f"  Checkpoint saved → {ckpt_path}", flush=True)

        # ── validation ───────────────────────────────────────────────────
        if args.val_every > 0 and global_step % args.val_every == 0 and val_loader is not None:
            val_loss = _validate(model, val_loader, config, device)
            val_ppl = math.exp(val_loss)
            print(
                f"  [val] step {global_step:>6}"
                f"  val_loss {val_loss:.4f}  val_ppl {val_ppl:7.2f}",
                flush=True,
            )

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
    p.add_argument(
        "--data-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a local UTF-8 text file of stories (newline-separated). "
            "When provided, skips the HuggingFace Hub download entirely. "
            "Useful for air-gapped or network-restricted environments."
        ),
    )
    p.add_argument(
        "--no-ssl-verify",
        action="store_true",
        default=False,
        help=(
            "Disable SSL certificate verification for the HuggingFace Hub download. "
            "Development/lab use only — use when a corporate proxy causes "
            "'self-signed certificate in certificate chain' errors."
        ),
    )

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

    # Episodic memory (Phase 13 validated architecture)
    p.add_argument(
        "--use-episodic-memory",
        action="store_true",
        help="Enable MemoryModule per layer (thresh*=0.70 per exp_48_1, Phase 12)",
    )
    p.add_argument(
        "--episodic-gate-thresh",
        type=float,
        default=0.70,
        help="OR-gate threshold for MemoryModule write gate",
    )
    p.add_argument(
        "--no-null-gate",
        action="store_true",
        help="Ablation: disable the null retrieval gate in MemoryModule (§12.2)",
    )
    p.add_argument(
        "--full-seq-residual",
        action="store_true",
        help="Ablation: broadcast memory residual to all token positions (default: last only)",
    )
    p.add_argument(
        "--memory-last-layer-only",
        action="store_true",
        help="Ablation: attach MemoryModule only to the final transformer layer",
    )

    # Infrastructure
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a .safetensors checkpoint to resume from")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=2000,
                   help="Save checkpoint every N steps (0 = only final)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="Run validation every N steps (0 = disabled)",
    )
    p.add_argument(
        "--val-max-chars",
        type=int,
        default=500_000,
        help="Maximum characters to load from TinyStories validation split",
    )
    p.add_argument(
        "--reset-on-boundary",
        action="store_true",
        default=False,
        help=(
            "Zero LayerState for batch elements whose segment crosses a story "
            "boundary (token 10 = '\\n'). Prevents L2 memory contamination "
            "across TinyStories documents."
        ),
    )

    return p


if __name__ == "__main__":
    train(_parser().parse_args())
