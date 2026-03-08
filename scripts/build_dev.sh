#!/usr/bin/env bash
# scripts/build_dev.sh — one-shot developer environment setup.
#
# Run once after cloning the repo.  Safe to re-run (all steps are idempotent).
#
# Usage:
#   bash scripts/build_dev.sh          # full setup
#   bash scripts/build_dev.sh --no-rust  # skip Rust/maturin build (Python only)
set -euo pipefail

SKIP_RUST=0
for arg in "$@"; do
  [[ "$arg" == "--no-rust" ]] && SKIP_RUST=1
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Python venv ──────────────────────────────────────────────────────────────
echo "==> Setting up Python virtual environment"
if ! command -v uv &>/dev/null; then
  echo "    uv not found — installing via pip"
  pip install --quiet uv
fi

if [[ ! -d .venv ]]; then
  uv venv --python 3.12
  echo "    Created .venv"
else
  echo "    .venv already exists"
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> Installing Python dependencies"
uv pip install --quiet -e ".[dev]"

# ── Rust toolchain ───────────────────────────────────────────────────────────
if [[ $SKIP_RUST -eq 0 ]]; then
  echo "==> Checking Rust toolchain"
  if ! command -v cargo &>/dev/null; then
    echo "    cargo not found — install Rust from https://rustup.rs"
    exit 1
  fi
  cargo --version

  echo "==> Installing maturin"
  uv pip install --quiet "maturin>=1.7,<2.0"

  echo "==> Building Rust extension (maturin develop)"
  .venv/bin/maturin develop

  echo "==> Verifying drex._sys import"
  .venv/bin/python -c "import drex._sys; print('    drex._sys OK:', [x for x in dir(drex._sys) if not x.startswith('_')])"
else
  echo "==> Skipping Rust build (--no-rust)"
fi

# ── Smoke tests ──────────────────────────────────────────────────────────────
echo "==> Running test suite"
.venv/bin/pytest tests/python/ -q --tb=short

echo ""
echo "Build complete.  Activate the environment with:"
echo "  source .venv/bin/activate"
