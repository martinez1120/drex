"""
drex — Novel LLM architecture with three-tier tiered memory.

Tiers:
  L1  Sliding window attention (working context, ~2-4k tokens, VRAM)
  L2  Infini-Attention matrix memory (compressed, δ-rule write, linear read)
  L3  Titans-style disk cache (MLP weight snapshots, async speculative prefetch)

The Rust systems layer (drex._sys) provides L3 disk I/O, H2O eviction,
and InfiniGen sketch-based prefetch. It is compiled by maturin; if the
extension is missing the Python model layers still work without L3 storage.

Example usage::

    from drex.models.transformer import DrexTransformer, DrexConfig
    config = DrexConfig(d_model=512, n_heads=8, n_layers=6)
    model = DrexTransformer(config).to("mps")
"""

__version__ = "0.1.0"

# Attempt to import the compiled Rust extension.
# It provides SnapshotStore, MemoryTierManager, and PrefetchEngine.
try:
    from drex._sys import SnapshotStore, SnapshotMeta, MemoryTierManager, PrefetchEngine
    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RUST_AVAILABLE = False
    SnapshotStore = None
    SnapshotMeta = None
    MemoryTierManager = None
    PrefetchEngine = None

__all__ = [
    "__version__",
    "_RUST_AVAILABLE",
    "SnapshotStore",
    "SnapshotMeta",
    "MemoryTierManager",
    "PrefetchEngine",
]
