"""
Tests for drex/__init__.py — version, Rust export presence, and ImportError fallback.
"""

import importlib
import sys

import pytest


def test_version():
    import drex
    assert drex.__version__ == "0.1.0"


def test_rust_available_exports():
    import drex
    assert drex._RUST_AVAILABLE is True
    assert drex.SnapshotStore is not None
    assert drex.SnapshotMeta is not None
    assert drex.MemoryTierManager is not None
    assert drex.PrefetchEngine is not None


def test_rust_unavailable_fallback():
    """Lines 27-32: the except ImportError branch sets all exports to None."""
    import drex

    # Stash the real module and replace with None → forces ImportError on import
    saved_sys_mod = sys.modules.pop("drex._sys", None)
    saved_drex = sys.modules.pop("drex", None)
    sys.modules["drex._sys"] = None  # type: ignore[assignment]

    try:
        import drex as drex_no_rust  # re-executes __init__.py

        assert drex_no_rust._RUST_AVAILABLE is False
        assert drex_no_rust.SnapshotStore is None
        assert drex_no_rust.SnapshotMeta is None
        assert drex_no_rust.MemoryTierManager is None
        assert drex_no_rust.PrefetchEngine is None
    finally:
        # Restore everything so subsequent tests are unaffected
        if saved_sys_mod is not None:
            sys.modules["drex._sys"] = saved_sys_mod
        elif "drex._sys" in sys.modules:
            del sys.modules["drex._sys"]

        if saved_drex is not None:
            sys.modules["drex"] = saved_drex
        elif "drex" in sys.modules:
            del sys.modules["drex"]

        # Force a clean reload so drex._RUST_AVAILABLE is True again
        importlib.reload(sys.modules["drex"])
