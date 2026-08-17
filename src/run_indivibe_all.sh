#!/usr/bin/env bash
# T3/T4 roster. gemma-4-31B-it was added after the 8-model roster was set: it had
# not been measured then, and is now the best model in the study on every axis,
# so excluding it would be indefensible. 9 models -> 36 pairwise comparisons.
set -u
cd "$(dirname "$0")/.."
PY=/home/harshal/miniconda3/envs/msb/bin/python
GPU="${1:-0}"
MODELS="sarvam-30b sarvam-m gemma-4-31B-it gemma-4-26B-A4B-it gemma-4-12B-it \
        Mistral-Small-3.2-24B Ministral-3-14B Qwen3-30B-A3B Llama-3.1-8B-Instruct"
mkdir -p results/logs
for m in $MODELS; do
  out="results/indivibe/${m}__chat.jsonl"
  if [ -s "$out" ]; then echo "SKIP $m"; continue; fi
  echo "=== $m $(date +%H:%M:%S) ==="
  if $PY src/run_indivibe.py --model "$m" --gpu "$GPU" > "results/logs/${m}__indivibe.log" 2>&1; then
    grep -E "compliance|^generated" "results/logs/${m}__indivibe.log"
  else
    echo "FAILED $m"; tail -3 "results/logs/${m}__indivibe.log"
  fi
done
echo "=== indivibe done $(date +%H:%M:%S) ==="
