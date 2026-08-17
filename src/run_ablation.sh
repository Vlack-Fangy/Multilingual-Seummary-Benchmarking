#!/usr/bin/env bash
# ABLATION: can in-context exemplars substitute for romanized training data?
# 0-shot numbers already exist; this adds 3-shot on the SAME items.
#   gap closes  -> capability is latent, deficit is distribution shift (fixable at inference)
#   gap persists-> deficit is representational, needs training
# Models span all three robustness tiers found in T1.
set -u
cd "$(dirname "$0")/.."
PY=/home/harshal/miniconda3/envs/msb/bin/python
GPU="${1:-0}"
for m in sarvam-30b gemma-4-31B-it Qwen3-30B-A3B Mistral-Small-3.2-24B; do
  out="results/exact/${m}__gsm8k_3shot__summary.json"
  if [ -s "$out" ]; then echo "SKIP $m"; continue; fi
  echo "=== $m 3-shot $(date +%H:%M:%S) ==="
  # 6144 ctx: three worked CoT exemplars plus the item do not fit in 4096.
  $PY src/run_exact.py --task gsm8k --model "$m" --shots 3 --limit 300 \
      --langs hi bn pa ta te --scripts native roman --gpu "$GPU" --max-model-len 6144 \
      > "results/logs/${m}__gsm8k_3shot.log" 2>&1 \
    && grep -E "^  [a-z]" "results/logs/${m}__gsm8k_3shot.log" \
    || { echo "FAILED $m"; tail -3 "results/logs/${m}__gsm8k_3shot.log"; }
done
echo "=== ablation done $(date +%H:%M:%S) ==="
