# Multilingual Summarization Benchmark

Research scripts for comparing language models across Hindi, Bengali, Punjabi,
Tamil, Telugu and English. The pipeline measures tokenizer fertility, generates
summaries, scores saved outputs and compares native-script and romanized tasks.
It also contains exact-answer QA, log-likelihood, pairwise judging and latency
experiments developed alongside the summarization study.

## What is included

- `src/common.py`: model registry, prompts and character-based output budgets.
- `src/download_data.py`, `build_eval_set.py`: dataset preparation and seeded sampling.
- `src/fertility.py`: tokens per word, character and byte.
- `src/generate.py`, `score.py`: generation and offline summary scoring.
- `src/run_exact.py`, `run_loglik.py`: QA and likelihood experiments.
- `src/judge*.py`, `analyze*.py`: model judging and result analysis.

## Setup

Use Python 3.12 on Linux. GPU generation requires a compatible NVIDIA driver,
vLLM and enough memory for the selected checkpoint. CPU preparation and analysis
can be run separately. Model weights are not included.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MSB_MODELS_ROOT="$HOME/models"
```

`requirements-pinned.txt` records the original inference-version choices;
`requirements.txt` adds the preparation/scoring dependencies. Place checkpoint
folders under `MSB_MODELS_ROOT` with the names listed in `src/common.py`.
Obtain gated models and accept their terms separately.

## Run a small experiment

```bash
python src/download_data.py
python src/build_eval_set.py
python src/fertility.py
CUDA_VISIBLE_DEVICES=0 python src/generate.py \
  --model Llama-3.1-8B-Instruct --condition native --langs hi --limit 5
python src/score.py --condition native
```

Use only GPUs allocated to you. Fertility scans available registered tokenizers;
check its output before generation because token budgets depend on it.
Download/preparation can be substantial even when generation uses five examples.
Romanized experiments require the corresponding romanized inputs; do not assume
native preparation creates every optional condition. See each command's `--help`.

Additional exact-answer experiment:

```bash
python src/run_exact.py --task mmlu --model Llama-3.1-8B-Instruct \
  --langs hi --scripts native --limit 5 --gpu 0
```

## Checks and outputs

```bash
python -m compileall -q src
python -m unittest discover -s tests -v
```

New local data, generations, logs and results are ignored. Older aggregate
reports already present in this repository are historical snapshots; they are
not fresh validation of the current source. Full model experiments were not
rerun for this packaging release.

## Interpretation

Character-based budgets reduce tokenizer-driven length differences, but do not
remove architecture, training-data or decoding confounds. XL-Sum is monolingual;
CrossSum-IN is cross-lingual. Compare them as distinct tasks. Model judges are
fallible; judge agreement is not human validation. Report truncation and script
compliance alongside quality scores.

## Data and license

Original code is MIT licensed. Datasets and models retain their own terms.
XL-Sum and CrossSum-IN have non-commercial/share-alike restrictions; obtain them
from their publishers. FLORES and other benchmark sources are referenced in the
download scripts. No new corpora, checkpoint weights or raw predictions are
included in this release. See `THIRD_PARTY_NOTICES.md`.
