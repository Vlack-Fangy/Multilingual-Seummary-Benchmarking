#!/usr/bin/env bash
# Single-GPU serial queue. GPU 1 is off-limits by request, so everything
# outstanding runs one model at a time on GPU 4.
#
# Waits for any model still downloading rather than failing on a partial
# checkpoint — a half-written safetensors set loads and generates garbage.
set -u
cd "$(dirname "$0")/.."
GPU=4

wait_for() {   # wait_for <dir> <expected_GB>
  local d="/home/models/$1" exp="$2" need have
  need=$(echo "$exp*0.99*1000000000/1" | bc)
  for _ in $(seq 1 240); do          # up to ~2h
    # Size of the weight files only. Do NOT require an index: single-shard models
    # (bloomz-7b1, gemma-4-12B-it) ship one model.safetensors and no index at all,
    # so an index check waits forever on a checkpoint that is already complete.
    have=$(find "$d" -maxdepth 1 -name '*.safetensors' -printf '%s\n' 2>/dev/null \
           | awk '{s+=$1} END {print s+0}')
    if [ -n "$have" ] && [ "$have" -ge "$need" ]; then return 0; fi
    sleep 30
  done
  return 1
}

run() {        # run <registry-key> <dir> <expected_GB>
  if [ -s "results/exact/$1__gsm8k__summary.json" ]; then echo "SKIP $1"; return; fi
  echo "--- waiting for $2 ---"
  if ! wait_for "$2" "$3"; then echo "TIMEOUT waiting for $2, skipping"; return; fi
  echo "=== $1 $(date +%H:%M:%S) ==="
  bash src/run_exact_all.sh "$GPU" "$1"
}

# GLM-4.7-Flash is deliberately NOT here: it is reserved as the second judge for
# T4/T5 (different family from gpt-oss-120b, per the single-judge fix). A model
# cannot judge a field it competes in, so it stays out of the contestant set.
run aya-expanse-8b  aya-expanse-8b  16.1
run aya-expanse-32b aya-expanse-32b 64.6
run bloomz-7b1      bloomz-7b1      14.1
echo "=== queue drained $(date +%H:%M:%S) ==="
