#!/usr/bin/env bash
# Iteration 3 driver: every contestant over the full eval set, native condition.
#
# Model-outer loop so each set of weights is loaded exactly once — load time
# dominates at this scale (Sarvam alone is a 128GB fp32 read cast to bf16).
# Ordered cheapest-first so a failure surfaces on an 8B model in minutes rather
# than after a 30B load. Each model writes its own file and is skipped if that
# file already exists, so the run is resumable.
set -u
cd "$(dirname "$0")/.."
PY=/home/harshal/miniconda3/envs/msb/bin/python
COND=${COND:-native}
LOGDIR=results/logs; mkdir -p "$LOGDIR"

MODELS=(
  Llama-3.1-8B-Instruct
  Qwen3-8B
  Qwen3.5-9B
  Mistral-7B-Instr-v0.3
  Qwen3-14B-Instruct
  Mistral-Small-3.2-24B
  gemma-2-27b-it
  Qwen3-30B-A3B
  Qwen3-32B
  sarvam-30b
)

for m in "${MODELS[@]}"; do
  out="results/generations/${m}__${COND}.jsonl"
  if [ -s "$out" ]; then echo "SKIP $m (already done)"; continue; fi
  echo "=== $m ($COND) $(date +%H:%M:%S) ==="
  if $PY src/generate.py --model "$m" --condition "$COND" > "$LOGDIR/${m}__${COND}.log" 2>&1; then
    grep -E "^  [a-z]{2}: |^loaded " "$LOGDIR/${m}__${COND}.log"
  else
    # Never let one model abort the sweep; record and continue.
    echo "FAILED $m — see $LOGDIR/${m}__${COND}.log"
    tail -3 "$LOGDIR/${m}__${COND}.log"
  fi
done
echo "=== done $(date +%H:%M:%S) ==="
