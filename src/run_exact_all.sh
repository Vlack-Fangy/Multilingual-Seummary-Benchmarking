#!/usr/bin/env bash
# T1/T2 driver. Usage:  bash src/run_exact_all.sh <GPU> <model> [model...]
#
# One stream per GPU so several models run concurrently on this shared box.
# Resumable: a model/task pair whose output file already exists is skipped.
#
# NOTE: timings recorded during a shared-GPU run are NOT valid for T6 — that
# needs a dedicated card and a warm-up pass. Accuracy is unaffected.
set -u
cd "$(dirname "$0")/.."
PY=/home/harshal/miniconda3/envs/msb/bin/python
GPU="$1"; shift
LIMIT="${LIMIT:-500}"
LOG=results/logs; mkdir -p "$LOG"

for m in "$@"; do
  for task in gsm8k mmlu; do
    out="results/exact/${m}__${task}.jsonl"
    if [ -s "$out" ] && [ -s "results/exact/${m}__${task}__summary.json" ]; then
      echo "SKIP $m/$task"; continue
    fi
    echo "=== gpu$GPU $m $task $(date +%H:%M:%S) ==="
    if $PY src/run_exact.py --task "$task" --model "$m" --limit "$LIMIT" --gpu "$GPU" \
         > "$LOG/${m}__${task}.log" 2>&1; then
      grep -E "^  [a-z]" "$LOG/${m}__${task}.log"
    else
      echo "FAILED $m/$task"; tail -3 "$LOG/${m}__${task}.log"
    fi
  done
done
echo "=== gpu$GPU done $(date +%H:%M:%S) ==="
