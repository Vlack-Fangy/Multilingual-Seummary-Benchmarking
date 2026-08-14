#!/usr/bin/env bash
# Fetch phase-2 model weights into the shared /home/models tree.
# Resumable: hf download skips files already present, so re-running is cheap.
set -u
DEST=/home/models
HF=/home/harshal/miniconda3/envs/car/bin/hf

MODELS=(
  # Tier 1 — family gaps
  google/gemma-4-12B-it
  # BF16 variants: the default Ministral-3 releases ship natively FP8
  # (quant_method: fp8, 9.8GB for an 8B model), which would confound the Mistral
  # arm against an otherwise bf16 field — the same trap that excluded
  # mistral_nemo_instruct. The -BF16 repos carry quantization_config: null.
  mistralai/Ministral-3-8B-Instruct-2512-BF16
  mistralai/Ministral-3-14B-Instruct-2512-BF16
  sarvamai/sarvam-m
  google/gemma-4-26B-A4B-it
  google/gemma-4-31B-it
  # Tier 1b — Sarvam's own Gemma comparison point (gated=manual; user has access)
  google/gemma-3-27b-it
  # Tier 2 — multilingual + Sarvam's comparison set
  CohereLabs/aya-expanse-8b
  CohereLabs/aya-expanse-32b
  allenai/OLMo-3.1-32B-Think
  zai-org/GLM-4.7-Flash
  # Tier 3 — labelled 2022 baseline
  bigscience/bloomz-7b1
)

for repo in "${MODELS[@]}"; do
  name="${repo##*/}"
  out="$DEST/$name"
  echo "=== $repo -> $out  $(date +%H:%M:%S) ==="
  # One --exclude per pattern: the flag takes a single TEXT value, and passing
  # several made the CLI treat them as explicit filenames ("Ignoring --exclude
  # since filenames have been explicitly set") which downloaded nothing.
  # consolidated.safetensors duplicates the sharded weights in Mistral repos.
  if $HF download "$repo" --local-dir "$out" \
        --exclude "*.gguf" --exclude "*.pth" --exclude "original/*" \
        --exclude "*consolidated*" 2>&1 | tail -2; then
    du -sh "$out" 2>/dev/null
  else
    echo "FAILED $repo (gated? check access)"
  fi
done
echo "=== all done $(date +%H:%M:%S) ==="
df -h /home | tail -1
