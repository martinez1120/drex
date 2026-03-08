#!/usr/bin/env bash
# scripts/run_eval.sh — run the full passkey recall sweep.
#
# Usage:
#   bash scripts/run_eval.sh                             # random-init baseline
#   bash scripts/run_eval.sh checkpoints/step_*.safetensors  # trained model
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${1:-}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -f "$PYTHON" ]]; then
  echo "ERROR: Python not found at $PYTHON" >&2
  echo "       Run bash scripts/build_dev.sh first." >&2
  exit 1
fi

ARGS=(
  scripts/eval_passkey.py
  --lengths 2048 4096 8192 16384 32768
  --trials 10
)

if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

echo "==> Starting passkey sweep"
"$PYTHON" "${ARGS[@]}"
