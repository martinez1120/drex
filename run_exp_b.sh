#!/usr/bin/env bash
set -e
cd /Users/wscholl/drex
echo "[run_exp_b.sh] Waiting for Exp A to complete..."
# Poll until Exp A log shows the final checkpoint
while ! grep -q "step_0050000_final" results/exp_a_train.log 2>/dev/null; do
  sleep 60
done
echo "[run_exp_b.sh] Exp A complete. Launching Exp B..."
PYTHONPATH=python python3.12 scripts/train.py \
  --steps 50000 --log-every 200 \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --batch-size 8 --segment-len 512 --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --val-every 1000 --val-max-chars 500000 \
  --reset-on-boundary \
  --seed 42 \
  --ckpt-dir checkpoints/exp_b \
  --save-every 5000 \
  --no-ssl-verify \
  2>&1 | tee results/exp_b_train.log
echo "[run_exp_b.sh] Exp B complete."
