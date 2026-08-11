# Iteration 1 — Tokenizer fertility

**Corpus:** FLORES devtest, 1012 sentences, sentence-aligned across all six languages
(via `google/IndicGenBench_flores_in`; `facebook/flores` and `openlanguagedata/flores_plus`
are both gated and need manual terms acceptance).
**Definition:** tokens per whitespace-delimited word. `tok_per_char` and `tok_per_byte` are also
in `fertility.csv`. Because the corpus is parallel, every language sees identical content — the
only thing varying across a row is the tokenizer.

## Contestants

| Family | Band | Model | hi | bn | pa | ta | te | en | **Brahmic mean** |
|---|---|---|---|---|---|---|---|---|---|
| Sarvam | 21–30B | `sarvam-30b` | 1.39 | 1.68 | 1.63 | 2.37 | 2.33 | 1.24 | **1.88** |
| Mistral | 21–30B | `Mistral-Small-3.2-24B` | 1.95 | 2.93 | 3.19 | 3.61 | 3.71 | 1.27 | **3.08** |
| Gemma | 21–30B | `gemma-2-27b-it` | 1.96 | 3.72 | 3.37 | 4.19 | 4.57 | 1.23 | **3.56** |
| Qwen | 21–30B | `Qwen3-30B-A3B` | 4.76 | 7.12 | 7.76 | 10.06 | 11.40 | 1.26 | **8.22** |
| Qwen | 21–30B | `Qwen3-32B` | 4.76 | 7.12 | 7.76 | 10.06 | 11.40 | 1.26 | **8.22** |
| Qwen | 13–20B | `Qwen3-14B-Instruct` | 4.76 | 7.12 | 7.76 | 10.06 | 11.40 | 1.26 | **8.22** |
| Qwen | 7–12B | `Qwen3.5-9B` | 3.33 | 4.38 | 4.26 | 6.68 | 6.33 | 1.26 | **5.00** |
| Qwen | 7–12B | `Qwen3-8B` | 4.76 | 7.12 | 7.76 | 10.06 | 11.40 | 1.26 | **8.22** |
| Llama | 7–12B | `Llama-3.1-8B-Instruct` | 2.67 | 8.02 | 8.19 | 12.39 | 13.30 | 1.24 | **8.91** |
| Mistral | 7–12B | `Mistral-7B-Instruct-v0.3` | 5.33 | 7.53 | 12.05 | 10.45 | 12.71 | 1.37 | **9.61** |
| *judge* | — | `gpt-oss-120b` / `-20b` | 1.65 | 2.34 | 2.72 | 3.17 | 3.06 | 1.23 | **2.59** |

## Findings

**1. The 10.43 discrepancy is resolved — Hindi is not representative.**
The brief cites Llama-3.1-8B at 10.43 on Brahmic means; a Hindi-only smoke test gave 2.35. Measured
across all five Indic languages, Llama-3.1-8B is **8.91** — the same ballpark as the published
figure. The gap was the sampling: **Hindi is Llama's best Indic language by a factor of five**
(hi 2.67 vs te 13.30). Any fertility number quoted from a single language is misleading, and Hindi
is the most misleading choice available.

**2. Within-model variance across languages rivals between-model variance.**
Max/min over the five Indic languages, per model:

| Model | min | max | ratio |
|---|---|---|---|
| `Llama-3.1-8B-Instruct` | 2.67 (hi) | 13.30 (te) | **4.98×** |
| `Qwen3-*` (all) | 4.76 (hi) | 11.40 (te) | 2.40× |
| `gemma-2-27b-it` | 1.96 | 4.57 | 2.33× |
| `Mistral-Small-3.2-24B` | 1.95 | 3.71 | 1.90× |
| `sarvam-30b` | 1.39 | 2.37 | **1.71×** |

Sarvam is not merely more efficient, it is the most *uniform* across scripts — which is what a
model trained for 22 Indian languages should look like, and is a cleaner signal of intent than
the headline mean. Llama is the opposite: competitive on Devanagari, collapsing on Dravidian.

**3. The premise holds: 4.37× spread at matched size.**
Within the 21–30B band alone, `sarvam-30b` (1.88) to `Qwen3-30B-A3B` (8.22). Same band, same task,
4.4× difference in how much context an article consumes. The brief's "roughly 5× spread" is
confirmed on our own measurements.

**4. Every Qwen3 model shares a byte-identical tokenizer.**
`Qwen3-8B`, `-14B`, `-30B-A3B` and `-32B` produce *exactly* the same token counts in all six
languages. Fertility is therefore perfectly confounded with model size inside the Qwen3 line —
useful, because it makes Qwen3 a controlled ladder for separating size effects from fertility
effects. **`Qwen3.5-9B` is the exception** (5.00 vs 8.22, vocab 248320 vs 151936): Qwen 3.5 shipped
a new tokenizer that cut Indic fertility by ~39%. A 9B model tokenizes Indic text substantially
better than a 32B model of the previous generation.

**5. Telugu and Tamil are the expensive languages, not Punjabi.**
Every model's worst two Indic languages are Dravidian. Punjabi sits mid-pack throughout. The brief
flags Punjabi as thinnest on data-availability axes — true — but on *fertility* it is not the
problem case, so context-budget truncation will bite Telugu and Tamil first.

**6. The judge choice is validated on its own terms.**
`gpt-oss-120b` at 2.59 tokens/word tokenizes Indic script better than **every contestant except
Sarvam** — better than Gemma, Mistral, and 3× better than Qwen3. It can hold article + reference +
candidate in context where a Qwen-family judge would truncate. (`gpt-oss-120b` and `-20b` are
byte-identical, so the inter-judge agreement check is uncontaminated by tokenizer differences.)

## Caveat

These are *tokenizer* measurements. They bound the context budget and cost, and they are a
candidate explanatory variable — nothing here yet shows that fertility predicts summarization
quality. That is what iterations 3–5 test. Note in advance that fertility is confounded with
vocabulary size, training-data mix, and (within Qwen3) model size, so a correlation will need
disentangling rather than being read straight off.

## Diagnostic (not contestants)

Included to locate the byte-fallback regime: `Llama-2-7b-chat` (32k vocab) at **11.18** Brahmic
mean, with Telugu at **19.59**. This is where a tokenizer with essentially no Indic vocabulary
lands, and it closely matches the "16.78 on Odia" figure quoted in the brief — evidence that some
of the published table describes small-vocab, pre-2024 tokenizers rather than current models.
