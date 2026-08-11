# Multilingual Summarization Benchmarking — Consolidated Results

**Status as of 2026-08-11.** Iterations 1–3 complete. Romanized and code-switched conditions,
and LLM-judge scoring, are not yet run.

Compares summarization across model families at matched sizes (7–30B) in Hindi, Bengali, Punjabi,
Tamil, Telugu and English, with tokenizer fertility as the candidate explanatory variable.

All numbers below are measured on this hardware from these weights. Nothing is quoted from
published tables — see §6.1 for why that distinction turned out to matter.

---

## 1. What was run

| | |
|---|---|
| **Generations** | 18,000 (10 models × 6 languages × 3 length buckets × 100 items) |
| **Corpus** | XL-Sum test split, deduplicated |
| **Task** | monolingual: native script in → native script out |
| **Prompt** | English instructions in every cell, so input script is the only variable |
| **Decoding** | greedy (temp 0, seed 0), thinking mode off everywhere |
| **Length control** | capped in **characters** (4× median gold), converted per model via its own measured tok/char |
| **Hardware** | one H200 NVL (GPU 4), vLLM 0.27.0, bf16, serial |
| **Outcome** | zero failures, zero empty outputs, zero context truncation |

Fertility is measured separately on 1,012 sentence-aligned FLORES devtest sentences, so every
language sees identical content and only the tokenizer varies.

### Length stratification

Bucket boundaries are held **identical across languages** (short <1500 chars, medium 1500–3500,
long >3500). Bucketing each language at its own quantiles would have hidden the effect under
test, because a "long" Tamil article and a "long" Hindi article would be different sizes.

---

## 2. Model inventory

Contestants — all local, unquantized, bf16:

| Family | 7–12B | 13–20B | 21–30B |
|---|---|---|---|
| Sarvam | — | — | `sarvam-30b` |
| Gemma | ❌ | — | ⚠️ `gemma-2-27b-it` |
| Qwen | `Qwen3.5-9B`, `Qwen3-8B` | `Qwen3-14B-Instruct` | `Qwen3-30B-A3B`, `Qwen3-32B` |
| Llama | `Llama-3.1-8B-Instruct` | — | — |
| Mistral | ⚠️ `Mistral-7B-Instruct-v0.3` | ❌ | `Mistral-Small-3.2-24B` |

`—` family ships nothing in band · `❌` exists upstream but not on disk · `⚠️` generation-lagged substitute

**Excluded with cause**, verified from `config.json` rather than directory name:
`diffusiongemma-26B-A4B-it` (block-diffusion decoding, not autoregressive) ·
`mistral_nemo_instruct` (bitsandbytes 4-bit — would confound size with quantization) ·
`Pixtral-12B-2409` (`model_type=granite`; an IBM model) ·
`Llama-3.1-70B-Instruct` (empty 84 KB stub).

**Not acquired:** Gemma 4 (12B, 26B-A4B), Ministral 3 (8B, 14B), Sarvam-M. Consequences in §7.

---

## 3. Result — tokenizer fertility

Tokens per whitespace word, FLORES devtest.

| Model | hi | bn | pa | ta | te | en | **Brahmic mean** |
|---|---|---|---|---|---|---|---|
| `sarvam-30b` | 1.39 | 1.68 | 1.63 | 2.37 | 2.33 | 1.24 | **1.88** |
| `Mistral-Small-3.2-24B` | 1.95 | 2.93 | 3.19 | 3.61 | 3.71 | 1.27 | **3.08** |
| `gemma-2-27b-it` | 1.96 | 3.72 | 3.37 | 4.19 | 4.57 | 1.23 | **3.56** |
| `Qwen3.5-9B` | 3.33 | 4.38 | 4.26 | 6.68 | 6.33 | 1.26 | **5.00** |
| `Qwen3-8B / 14B / 30B-A3B / 32B` | 4.76 | 7.12 | 7.76 | 10.06 | 11.40 | 1.26 | **8.22** |
| `Llama-3.1-8B-Instruct` | 2.67 | 8.02 | 8.19 | 12.39 | 13.30 | 1.24 | **8.91** |
| `Mistral-7B-Instruct-v0.3` | 5.33 | 7.53 | 12.05 | 10.45 | 12.71 | 1.37 | **9.61** |
| *judge* `gpt-oss-120b` / `-20b` | 1.65 | 2.34 | 2.72 | 3.17 | 3.06 | 1.23 | **2.59** |

**3.1 A 4.4× spread exists at matched size.** Within the 21–30B band alone, `sarvam-30b` (1.88)
to `Qwen3-30B-A3B` (8.22). The premise the project was built on is real.

**3.2 Never quote single-language fertility.** The handoff brief cited Llama-3.1-8B at 10.43;
an initial Hindi-only measurement gave 2.35, appearing to contradict it. Measured across all five
Indic languages it is **8.91** — the published figure was approximately right, and the
single-language number was the misleading one. **Hindi is Llama's best Indic language by 5×**
(2.67 vs 13.30 Telugu).

**3.3 Uniformity across scripts differs more than the mean suggests.** Max/min across the five
Indic languages: `sarvam-30b` **1.71×**, Qwen3 2.40×, `gemma-2` 2.33×, `Llama-3.1-8B` **4.98×**.
Sarvam is not merely efficient, it is even-handed; Llama is competitive on Devanagari and
collapses on Dravidian.

**3.4 All four Qwen3 sizes share a byte-identical tokenizer** (identical token counts in all six
languages). This makes Qwen3 a controlled ladder — see §5.1. `Qwen3.5-9B` is the exception: its
new tokenizer cut Indic fertility ~39%, so a 9B model tokenizes Indic better than a
previous-generation 32B.

**3.5 Dravidian is expensive, Punjabi is not.** Every model's two worst languages are Tamil and
Telugu. Punjabi is thin on *data availability* but mid-pack on fertility.

---

## 4. Result — native-script summarization quality (chrF)

| Family | Band | Model | hi | bn | pa | ta | te | en | **Indic mean** |
|---|---|---|---|---|---|---|---|---|---|
| Llama | 7–12B | `Llama-3.1-8B-Instruct` | 30.67 | 33.59 | 26.98 | 31.20 | 27.50 | 30.08 | **29.99** |
| Gemma | 21–30B | `gemma-2-27b-it` | 29.47 | 31.90 | 26.21 | 32.29 | 28.56 | 32.42 | **29.69** |
| Qwen | 21–30B | `Qwen3-30B-A3B` | 29.50 | 30.87 | 28.36 | 31.47 | 27.30 | 29.94 | **29.50** |
| Mistral | 7–12B | `Mistral-7B-Instruct-v0.3` | 27.77 | 31.66 | 25.03 | 33.87 | 26.13 | 27.76 | **28.89** |
| Sarvam | 21–30B | `sarvam-30b` | 28.28 | 30.24 | 26.44 | 31.54 | 26.55 | 29.06 | **28.61** |
| Mistral | 21–30B | `Mistral-Small-3.2-24B` | 29.03 | 30.69 | 27.76 | 29.04 | 26.30 | 29.94 | **28.56** |
| Qwen | 13–20B | `Qwen3-14B-Instruct` | 27.69 | 30.79 | 26.29 | 29.36 | 24.80 | 29.71 | **27.78** |
| Qwen | 7–12B | `Qwen3.5-9B` | 26.69 | 27.30 | 26.38 | 25.22 | 24.29 | 29.05 | **25.98** |
| Qwen | 21–30B | `Qwen3-32B` | 26.66 | 28.12 | 24.59 | 25.77 | 23.56 | 30.40 | **25.74** |
| Qwen | 7–12B | `Qwen3-8B` | 25.99 | 27.05 | 23.97 | 27.13 | 24.09 | 29.89 | **25.65** |

**The entire field spans 25.65–29.99.** No model dominates: the per-language winner changes four
times across six languages (Llama on hi/bn, Qwen3-30B-A3B on pa, Mistral-7B on ta, gemma-2 on
te/en). **Read §6 before drawing conclusions from this table.**

---

## 5. Result — fertility does not predict quality

| Test | Result |
|---|---|
| Fertility vs chrF, across 10 models (Indic mean) | **r = −0.11, p = 0.76** |
| Fertility vs chrF, across 50 model×language cells | **r = −0.19, p = 0.18** |

Neither is distinguishable from no relationship. The best-tokenizing model (`sarvam-30b`, 1.88 →
28.61) and the worst (`Mistral-7B-v0.3`, 9.61 → 28.89) sit **0.28 chrF apart** across a 5×
fertility difference.

### 5.1 The controlled test: the Qwen3 ladder

Fertility held *exactly* constant (byte-identical tokenizer, §3.4):

| Model | Fertility | chrF |
|---|---|---|
| `Qwen3-8B` | 8.22 | 25.65 |
| `Qwen3-14B-Instruct` | 8.22 | 27.78 |
| `Qwen3-30B-A3B` | 8.22 | **29.50** |
| `Qwen3-32B` | 8.22 | 25.74 |

**A 3.85-point chrF spread with fertility fixed** — wider than the spread across the entire
1.88→9.61 fertility range. Whatever drives quality here, it is not the tokenizer.

⚠️ **Do not read this as "MoE beats dense."** `Qwen3-30B-A3B-Instruct-**2507**` is the July 2025
post-training refresh, while `Qwen3-8B` and `Qwen3-32B` are the original April 2025 hybrid
releases. Its 3.76-point lead over `Qwen3-32B` confounds architecture with post-training vintage
and cannot be attributed to either. The ladder is a clean control **for fertility** — that is what
the byte-identical tokenizer buys — and nothing else. Separating architecture from vintage would
need `Qwen3-32B-Instruct-2507`, which is not on disk.

### 5.2 What fertility *does* predict: cost

Identical 1,800-item workload, generation time only:

| Model | Fertility | Load | Generate |
|---|---|---|---|
| `sarvam-30b` | 1.88 | 46 s | **27 s** |
| `Qwen3.5-9B` | 5.00 | 99 s | 123 s |
| `Qwen3-30B-A3B` | 8.22 | 95 s | 186 s |
| `Mistral-Small-3.2-24B` | 3.08 | 78 s | 240 s |
| `Qwen3-8B` | 8.22 | 41 s | 250 s |
| `Llama-3.1-8B-Instruct` | 8.91 | 15 s | 280 s |
| `gemma-2-27b-it` | 3.56 | 31 s | 355 s |
| `Qwen3-14B-Instruct` | 8.22 | 47 s | 434 s |
| `Mistral-7B-Instruct-v0.3` | 9.61 | 42 s | 438 s |
| `Qwen3-32B` | 8.22 | 94 s | **999 s** |

**37× between fastest and slowest** for the same work. Within a single model the effect is clean
because architecture is held constant — `Llama-3.1-8B` on 300 items: Hindi **14 s** → Telugu
**87 s** (6.2×), tracking its per-language fertility (2.67 → 13.30) almost exactly.

Fertility is a deployment-cost argument, not a capability argument.

---

## 6. Result — behaviour, and why the metric is suspect

| Model | script ok | length ratio | cap hits | chrF |
|---|---|---|---|---|
| `Llama-3.1-8B-Instruct` | 99.9% | 1.11 | 0 | 29.99 |
| `gemma-2-27b-it` | 99.8% | 1.06 | 0 | 29.69 |
| `Qwen3-30B-A3B` | 99.9% | 1.28 | 0 | 29.50 |
| `Mistral-7B-Instruct-v0.3` | **97.4%** | **2.59** | **178** | 28.89 |
| `sarvam-30b` | 99.8% | 1.17 | 3 | 28.61 |
| `Mistral-Small-3.2-24B` | 99.9% | 1.13 | 0 | 28.56 |
| `Qwen3-14B-Instruct` | 99.9% | 1.09 | 0 | 27.78 |
| `Qwen3.5-9B` | 99.9% | 0.96 | 0 | 25.98 |
| `Qwen3-32B` | 99.9% | 0.87 | 1 | 25.74 |
| `Qwen3-8B` | 99.9% | 0.96 | 0 | 25.65 |

**6.1 Script compliance does not differentiate in the native condition** — 99.8–99.9% for nine of
ten models. It is expected to differentiate in the romanized condition, which is why it is
instrumented. `Mistral-7B-v0.3` is the sole outlier and the only runaway generator (2.59× the
reference length, 178 cap hits against 0–3 for everyone else). It stands in for the absent
Ministral 3, so this reflects that substitution's limits, not current Mistral.

**6.2 chrF is not simply tracking output length.** Correlation between chrF and length ratio
across models is r = 0.27 (p = 0.46) — the obvious confound was checked and rejected.

**6.3 Robustness to article length varies more than average quality.** chrF, short → long bucket:

| Model | short | long | drop |
|---|---|---|---|
| `Llama-3.1-8B-Instruct` | 30.27 | 29.39 | **0.88** |
| `Mistral-7B-Instruct-v0.3` | 29.30 | 27.57 | 1.73 |
| `sarvam-30b` | 29.75 | 27.21 | 2.54 |
| `Qwen3-14B-Instruct` | 29.49 | 26.44 | 3.05 |
| `Mistral-Small-3.2-24B` | 30.26 | 27.19 | 3.07 |
| `Qwen3-30B-A3B` | 30.96 | 27.88 | 3.08 |
| `Qwen3.5-9B` | 28.14 | 24.77 | 3.37 |
| `Qwen3-32B` | 28.19 | 24.72 | 3.47 |
| `Qwen3-8B` | 28.01 | 24.44 | 3.58 |
| `gemma-2-27b-it` | 31.88 | 28.10 | **3.78** |

Every model degrades; `gemma-2` has the best short-article score and the worst degradation.

**6.4 No model was context-limited.** Zero truncation across all 18,000 items, including
`gemma-2-27b-it` at its 8192 ceiling — every other contestant supports 32k–262k. Its efficient
tokenizer offsets the small window, so context length and fertility partly cancel. The truncation
machinery was verified separately by forcing a 4k context, where `Llama-3.1-8B` dropped **68.2%
of every Telugu article and 0% of Hindi** — the fertility→context effect in isolation.

---

## 7. Caveats and threats to validity

**7.1 chrF may not be discriminating summarization quality.** The whole field spans 4.3 chrF
points. The literature this project rests on predicts exactly this: IndicGenBench reports ChrF
*because* token-level metrics fail on low-resource languages, HinGE finds five standard NLG
metrics ineffective, ITEM examines this precise reliability question. **The §5 null result is
currently "fertility does not predict chrF."** It becomes "does not predict quality" only if the
LLM judge agrees. That makes judging the critical path, ahead of the romanized condition.

**7.2 Contamination is a live alternative explanation for the §4 ranking.** The top two models
are the two *oldest* — `gemma-2-27b-it` (Jun 2024) and `Llama-3.1-8B` (Jul 2024) — from when
XL-Sum was a widely used benchmark. XL-Sum is BBC content, publicly crawled, and in mT5's
finetuning lineage. The ranking may partly measure XL-Sum familiarity rather than skill.

**7.3 Sarvam placing 5th of 10 should raise suspicion of the metric before the model**, given 7.1
and 7.2. An Indic-specialist model that tokenizes Indic 4.4× more efficiently than the field and
generates 37× faster, scoring mid-pack on Indic summarization, is a result that needs a second
measurement before it is believed.

**7.4 The design is not fully crossed.** Sarvam appears only in 21–30B, Llama only in 7–12B; only
Qwen and Mistral span bands. "Family effect controlling for size" is **not identifiable** with
this inventory. Claims must be scoped to matched-size comparisons within a band.

**7.5 The Gemma arm is two generations old.** Any "Gemma vs Sarvam" statement is really
"Gemma **2** vs Sarvam." The 13–20B band contains one model and supports no cross-family
comparison at all.

**7.6 Single seed, greedy decoding.** No variance estimate across runs. Greedy makes results
reproducible but gives no confidence intervals on per-model differences of ~1 chrF.

---

## 8. Pipeline defects found (methodological record)

Each of these would have silently corrupted results rather than failing loudly:

1. **Sarvam reasons by default and ignores its own `<|nothink|>` marker.** With
   `enable_thinking=False` correctly rendered (token id 28 verified present), it still emitted
   English `<think>` traces and hit the cap on 20/20 items, producing **zero summaries**. Fixed by
   prefilling a closed `<think></think>` block. Undetected, the flagship model would have produced
   1,800 empty cells.
2. **vLLM's `chat_template_kwargs` is accepted but never reaches the template.** We render the
   chat template ourselves and pass **token ids** — control tokens only work carrying their
   special-token id; re-tokenizing a rendered string turns them into literal characters.
3. **The script classifier counted URLs as script evidence.** Clean Devanagari summaries
   containing `www.bbc.co.uk/hindi/...` classified as "mixed". URLs/emails/handles now stripped
   before profiling — this matters most in the romanized condition, where script compliance *is*
   the result.
4. **A uniform `max_tokens` would have manufactured the headline.** At the same token budget,
   Llama would get roughly a quarter of Sarvam's Telugu output length. Capping in characters and
   converting per model via measured tok/char keeps the budget equal; verified non-binding
   (`finish_reason=stop` for 17,818/18,000 — the 182 exceptions are 178 from the single runaway
   model, 3 Sarvam, 1 Qwen3-32B).
5. **`rouge_score`'s default tokenizer strips non-ASCII**, silently returning 0.0 for every
   Indic row. ROUGE-L is hand-rolled over whitespace tokens and flagged as unreliable regardless.
6. **Fixed 8192 context crashed the first sweep** on a 8,937-token Llama prompt — the study's own
   phenomenon, since article length in tokens is a function of fertility. Context is now per-model
   native (capped 32k) with article-only truncation and per-row accounting.

---

## 9. Not yet done

| | Status |
|---|---|
| **LLM-judge scoring** (`gpt-oss-120b`, reference-based, `-20b` for inter-judge agreement) | **critical path** — §7.1 |
| Romanized condition (script robustness — the core contribution) | blocked on IndicXlit/`fairseq` under py3.13; pure-Python fallback ready |
| Code-switched condition | no gold data for hi/bn/pa/te; CS-Sum is Tamil-only and dialogue-shaped |
| CrossSum-IN (cross-lingual, downloaded) | not run |
| **Sarvam-M** | deferred by decision — the only near-controlled test of Indic post-training on an identical backbone |
| Gemma 4, Ministral 3 | not acquired; see §7.5 |
| Contamination controls, multi-seed variance | open |

## Reproducing

```bash
python src/download_data.py     # XL-Sum, CrossSum-IN, FLORES devtest
python src/fertility.py         # §3 — CPU only, no weights loaded
python src/build_eval_set.py    # 1800-item stratified sample, seed 20260811
bash   src/run_all.sh           # §4 — all 10 models, native condition
python src/score.py --condition native
```

Corpora are not redistributed here: XL-Sum and CrossSum-IN are CC BY-NC-SA, and vendoring them
would push a NonCommercial + ShareAlike restriction onto this repository.
