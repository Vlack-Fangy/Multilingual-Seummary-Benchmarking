# Multilingual Summarization Benchmark — Model Inventory & Gaps

Compare summarization capability across model families at matched sizes (7–30B), under
native-script / romanized / code-switched conditions, in Hindi, Bengali, Punjabi, Tamil,
Telugu, English.

**Policy: no models are downloaded for this project.** Everything runs from the shared
`/home/models` tree. Where the handoff brief names a model that is not present there, it is
recorded below as *unaccounted* and excluded from the design rather than fetched.

**Compute: GPU 4 only** (H200 NVL, 143 GB, idle at time of survey). The other five GPUs on
`sentinel` are in use by other jobs at 84–100% util. All work is single-GPU and serial.

Verified 2026-08-11 against `/home/models` (3.3 TB, world-writable, shared with other users).

---

## 1. Band coverage from local weights only

| Family | 7–12B | 13–20B | 21–30B |
|---|---|---|---|
| **Sarvam** | — *(none exists)* | — *(none exists)* | `sarvam-30b` ✅ |
| **Gemma** | ❌ unaccounted | — *(none exists)* | ⚠️ `gemma-2-27b-it` only |
| **Qwen** | `Qwen3.5-9B` ✅ · `Qwen3-8B` | `Qwen3-14B-Instruct` ✅ | `Qwen3-30B-A3B-Instruct-2507` ✅ · `Qwen3-32B` |
| **Llama** | `Llama-3.1-8B-Instruct` ✅ | — *(none exists)* | — *(none exists)* |
| **Mistral** | ⚠️ `mistral_nemo_instruct` / `Mistral-7B-Instruct-v0.3` | ❌ unaccounted | `Mistral-Small-3.2-24B-Instruct-2506` ✅ |

`—` = the family ships nothing in this band (ecosystem fact, not a local gap).
`❌` = the model exists upstream but is not on disk.
`⚠️` = a substitute is on disk but it is not the model the brief specified.

**Net: the 21–30B panel is viable for four of five families. The 7–12B panel loses Gemma
entirely and degrades Mistral by roughly two years.**

---

## 2. Unaccounted models

Named in the handoff brief, **not present** in `/home/models`. Not downloaded, per policy.

| Model | Band | Consequence |
|---|---|---|
| **Gemma 4 12B Unified** | 7–12B | Gemma has **no representative at all** in the lower panel. |
| **Gemma 4 26B A4B** | 21–30B | Upper-panel Gemma falls back to `gemma-2-27b-it` — two generations old (Jun 2024), 8K context, no Gemma-4 multilingual training. Any "Gemma vs Sarvam" claim is really "Gemma **2** vs Sarvam". |
| **Ministral 3 8B** | 7–12B | Mistral's lower-panel slot falls to Mistral NeMo 12B or Mistral-7B-v0.3 (May 2024, 32k vocab — pre-Tekken, so it does not even test the tokenizer of interest). |
| **Ministral 3 14B** | 13–20B | The 13–20B panel collapses to a **single model** (`Qwen3-14B-Instruct`). No cross-family comparison is possible in this band. |
| **Qwen3.6-27B** | 21–30B | Substitutable — `Qwen3-30B-A3B-Instruct-2507` and `Qwen3-32B` are present and adequate. Low impact. |
| **Qwen3.5-27B** | 21–30B | Same. Low impact. |
| **Sarvam-M (23.6B)** | 21–30B | Loses the cleanest contrast available: Sarvam-M is built *on* Mistral Small, so Sarvam-M vs `Mistral-Small-3.2-24B` would have isolated the effect of Indic post-training on an identical backbone. Notable scientific loss. |

### Impact on the research question

Two of the five families cannot be compared at matched size against Sarvam on current-generation
weights. If the headline claim is *"how do model families differ on Indic summarization at
matched size"*, then as things stand it is answerable for **Sarvam · Qwen · Mistral** at 21–30B,
and for **Qwen · Llama** at 7–12B. Gemma cannot participate on equal footing anywhere.

Three ways forward, in order of cost — pick one:
1. **Ship the 21–30B panel only** (4 families, one clean band). Honest, publishable, no downloads.
2. **Report Gemma 2 explicitly as a generation-lagged datapoint**, clearly labelled, not as "Gemma".
3. **Add Gemma 4 12B + 26B-A4B and Ministral 3 8B/14B to the shared tree** — ~150 GB against
   2.4 TB free on `/home`. Requires lifting the no-download policy.

---

## 3. Traps found during the survey

Directory names in `/home/models` do not reliably describe contents. Verified from `config.json`:

- **`diffusiongemma-26B-A4B-it` is not Gemma 4.** `model_type=diffusion_gemma`,
  `DiffusionGemmaForBlockDiffusion`. It shares the Gemma-4 26B-A4B backbone (vocab 262144,
  128 experts) but decodes by **block diffusion, not autoregression**. Generation semantics and
  `max_new_tokens` do not transfer. **Unusable as the Gemma AR baseline.**
- **`Pixtral-12B-2409` is not Pixtral.** `model_type=granite` — it is an IBM Granite model.
- **`sarvam-30b` is stored in fp32** (128.6 GB per the safetensors index; `torch_dtype` is unset).
  Loading with `dtype="auto"` will attempt 128 GB and leave ~15 GB for KV cache on a 143 GB card.
  **Must load with `dtype="bfloat16"` explicitly** (~64 GB).
- **`sarvam-30b` needs `trust_remote_code=True`** (`modeling_sarvam_moe.py`). Native vLLM support
  is still an open PR (vllm-project/vllm#33942); the repo ships `hotpatch_vllm.py` requiring
  **vllm==0.15.0**, or a vendor fork. This is the main integration risk in the run plan.
- **`Mistral-Small-3.2-24B` ships no HF tokenizer** — only `tekken.json`. `AutoTokenizer` will
  fail; it needs `mistral-common`. This matters directly, since Tekken is one of the
  high-fertility tokenizers the study is built around.
- **`model_list.txt` overstates the tree.** It lists `gemma-3-12b-it`, `gemma-3-27b-it`,
  `gemma-2-2b-it`, `InternVL3-14B`, `internvl3-8b` — none of which exist on disk. Do not plan
  against that file.
- `Qwen3-14B` (56 GB) appears to duplicate weights; `Qwen3-14B-Instruct` (28 GB) is the clean
  bf16 copy and is the one to use.
- `/home/models` is `777` and shared. `test_file_200_per_language.jsonl` and `test.jsonl` there
  belong to another user (`sudipto`, an MMLU-style QA set in en/hi/bn/kn/ta) and are unrelated
  to this project. Do not modify anything in that tree.

---

## 4. Tokenizer fertility — DONE (iteration 1)

Full results: [`results/fertility/FINDINGS.md`](results/fertility/FINDINGS.md).
Measured on 1012 sentence-aligned FLORES devtest sentences across all six languages.

Brahmic means, tokens/word: **Sarvam 1.88** · Mistral-Small-3.2 3.08 · Gemma-2-27B 3.56 ·
Qwen3.5-9B 5.00 · Qwen3 (all sizes) 8.22 · Llama-3.1-8B 8.91 · Mistral-7B-v0.3 9.61.
Judge `gpt-oss-120b` 2.59.

**The 10.43 discrepancy is resolved: Hindi is not representative.** Llama-3.1-8B measures **8.91**
across all five Indic languages — the same ballpark as the published figure. The earlier 2.35 came
from measuring Hindi alone, and **Hindi is Llama's best Indic language by 5×** (hi 2.67, te 13.30).
Never quote a single-language fertility number; Hindi is the most misleading choice available.

Other results that change how later iterations should be read:
- **Premise confirmed** — 4.37× spread inside the 21–30B band alone (Sarvam 1.88 → Qwen3 8.22).
- **Every Qwen3 size shares a byte-identical tokenizer**, so fertility is perfectly confounded with
  size within that line — which makes Qwen3 a controlled ladder for separating the two.
  `Qwen3.5-9B` is the exception: a new tokenizer cut Indic fertility ~39% versus Qwen3.
- **Telugu and Tamil are the expensive languages, not Punjabi.** Context truncation will bite
  Dravidian first.
- Sarvam is the most *uniform* across scripts (1.71× spread vs Llama's 4.98×), which is a cleaner
  signal of multilingual intent than the headline mean.

---

## 5. Datasets — corrections to the brief

| Brief | Actual |
|---|---|
| `mistralai/Ministral-3-8B` / `-14B` | 404. Real: `mistralai/Ministral-3-{8B,14B}-Instruct-2512` |
| CrossSum-IN "under `google/`" | `google/IndicGenBench_crosssum_in` — ungated, per-language JSON |
| XL-Sum via parquet mirror | HF parquet auto-conversion **fails** on this repo. Use `hf_hub_download` on the per-language `.tar.bz2` and read the JSONL directly — no script, no `trust_remote_code`, no version pin |

**CrossSum-IN is cross-lingual** (`crosssum_english-hi_test.json`: English doc → Indic summary),
not monolingual like XL-Sum. It is a second task, not a held-out replication of the XL-Sum
numbers — disagreement between them does not cleanly indicate contamination.

**IndicXlit blocker:** `ai4bharat-transliteration` depends on `fairseq` (and `urduhack` →
TensorFlow). `fairseq` does not build on Python 3.13, which is what every conda env here runs.
This sits on the critical path for the romanized condition. Needs a dedicated py3.10 env or
direct model invocation, and it is a day of work, not a `pip install`.

---

## 6. Design decisions — settled 2026-08-11

1. **Script-matched output.** Devanagari in → Devanagari out; same for Bengali, Gurmukhi, Tamil,
   Telugu. Roman in → romanized out (Hinglish/Tanglish/…) where the model can manage it, English
   accepted as fallback.
   - *Consequence, accepted:* the roman condition scores in Latin space against a romanized gold
     while the native condition scores in Brahmic space — different character inventories, so the
     native→roman delta is not one clean number. **Mitigation:** score the roman condition *both*
     ways — (a) Latin-space vs IndicXlit-romanized gold, (b) back-transliterated to native vs
     native gold — so cross-condition comparison stays available.
   - *"Is a romanized summary feasible" is a measured result, not an assumption.* Script-ID every
     output (Unicode-range classifier, Bhasha-Abhijnaanam as reference) and report the rate at
     which each model complies, silently falls back to English, or code-switches. Expect wide
     variance: Hinglish and Tanglish are well-attested written practice; Punglish is barely
     attested at all.
2. **Output length capped in characters, not tokens** — a uniform `max_new_tokens` would give
   high-fertility models fewer words of output and manufacture the fertility→quality correlation
   the study exists to test. Per-model token ceiling scaled by measured fertility.
   **Additionally: stratify inputs by article character length** (buckets, e.g. short / medium /
   long) to test the fertility→context-budget interaction directly rather than assuming it.
3. **No thinking mode** anywhere in the contestant set. Filed as a future extension.
4. **Compute all metrics** — chrF/chrF++, ROUGE, BERTScore, LLM-judge — and select afterwards
   rather than pre-committing.
5. **Each dataset runs its own native task shape.** XL-Sum = monolingual (same language in, same
   language out). CrossSum-IN = cross-lingual, as designed. CrossSum-IN is *not* forced into the
   XL-Sum grid as a fourth column.
6. **English prompts fixed across all cells**, isolating input script as the variable.
7. **Judge: `gpt-oss-120b`, local, reference-based.**

### Judge selection

| Candidate | Fits GPU 4 | Indic fertility (hi/ta/te) | Overlap with contestants |
|---|---|---|---|
| **`gpt-oss-120b`** | **65 GB, MXFP4** ✅ | **1.1 / 2.0 / 2.3** | **none** |
| `gpt-oss-20b` | 42 GB bf16 ✅ | same tokenizer | none |
| `Qwen2.5-72B-Instruct-AWQ` | 39 GB ✅ | 4.0 / 7.5 / **12.4** ❌ | Qwen ⚠️ |
| `Qwen2.5-32B-Instruct` | 62 GB ✅ | poor | Qwen ⚠️ |
| `DeepSeek-R1-0528` | 642 GB ❌ | — | none |
| `Llama-3.1-70B-Instruct` | **empty 84 KB stub** ❌ | — | Llama ⚠️ |

`gpt-oss-120b` wins on all three axes: it fits with room for KV cache, its `o200k_harmony`
tokenizer handles Indic script *better than most contestants* (Telugu 2.3 vs Qwen2.5-72B's 12.4),
and there is **no OpenAI model in the contestant set**, so it is the only strong local judge with
zero self-preference bias. Any Qwen or Llama judge would be scoring its own family.

- **Reference-based judging** (judge sees the gold summary and compares) rather than reference-free.
  METAL's warning about weak multilingual judges applies mainly to reference-free evaluation, where
  the judge must independently assess Tamil fluency. Comparison against a gold is a much lower
  competence bar and is what makes a non-Indic-specialised judge defensible.
- **`gpt-oss-20b` as a second judge on a subsample** for inter-judge agreement. Agreement on rank
  order means the judge is not the bottleneck; divergence is caught for ~5% of the compute.
- Judge reasoning effort **medium** — item 3's "no thinking" governs contestants, not the judge.

### Still unresolved

- **Sarvam-M** was approved for inclusion but is **not in `/home/models`** and the no-download
  policy forbids fetching it (~47 GB). It remains the cleanest experiment available — identical
  Mistral-Small backbone to a model already on disk, isolating the effect of Indic post-training.
  Pull this one model, or drop it?

## 7. Reproducing

```bash
pip install transformers huggingface_hub mistral-common pandas tabulate
python src/download_data.py     # XL-Sum (431 MB), CrossSum-IN, FLORES devtest
python src/fertility.py         # tokenizer fertility, CPU only, no weights loaded
python src/build_eval_set.py    # 1800-item length-stratified eval sample (seeded)
```

`MODELS` in `src/fertility.py` points at `/home/models`; edit those paths to run elsewhere.

**Data is not redistributed in this repo.** XL-Sum and CrossSum-IN are **CC BY-NC-SA**
(non-commercial research only), and vendoring them would push that NonCommercial + ShareAlike
restriction onto everything here. The download script fetches them from source instead, and the
eval sample is regenerated exactly from seed `20260811`.

## 8. Environment

- `car` env: torch 2.13.0, transformers 5.14.1, datasets 5.0.1 — usable, no vLLM anywhere yet.
- HF reachable via IITD proxy (`xen03.iitd.ac.in:3128`); token present, user `Vlack-Fangy`.
- 2.4 TB free on `/home`, 1.6 TB on `/`.
