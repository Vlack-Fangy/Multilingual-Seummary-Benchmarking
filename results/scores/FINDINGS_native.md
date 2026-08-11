# Iteration 3 — Native-script summarization, 10 models × 6 languages

**18,000 generations** (10 models × 6 languages × 3 length buckets × 100 items), XL-Sum,
native script in → native script out, English instructions, greedy decoding, no thinking mode.
Zero failures, zero empty outputs, zero context truncation.

## Headline: tokenizer fertility does **not** predict summarization quality

This was the variable the whole comparison was built around. It does not survive contact with data.

| Test | Result |
|---|---|
| Fertility vs chrF, across 10 models (Indic mean) | **r = −0.11, p = 0.76** |
| Fertility vs chrF, across 50 model×language cells | **r = −0.19, p = 0.18** |

Neither is distinguishable from no relationship. The ordering makes the point plainly — the
best-tokenizing model and the worst-tokenizing model land within 1.4 chrF of each other:

| Model | Fertility | chrF |
|---|---|---|
| `sarvam-30b` | **1.88** (best) | 28.61 |
| `Mistral-Small-3.2-24B` | 3.08 | 28.56 |
| `gemma-2-27b-it` | 3.56 | **29.69** |
| `Qwen3.5-9B` | 5.00 | 25.98 |
| `Qwen3-*` (8B/14B/30B/32B) | 8.22 | 25.65 – 29.50 |
| `Llama-3.1-8B-Instruct` | 8.91 | **29.99** (best) |
| `Mistral-7B-Instruct-v0.3` | **9.61** (worst) | 28.89 |

### The cleanest evidence: the Qwen3 ladder

All four Qwen3 sizes share a **byte-identical tokenizer** (verified in iteration 1 — identical
token counts in all six languages). Fertility is therefore held *perfectly* constant:

| Model | Fertility | chrF |
|---|---|---|
| `Qwen3-8B` | 8.22 | 25.65 |
| `Qwen3-14B-Instruct` | 8.22 | 27.78 |
| `Qwen3-30B-A3B` | 8.22 | **29.50** |
| `Qwen3-32B` | 8.22 | 25.74 |

**A 3.85-point chrF spread with fertility exactly fixed** — larger than the spread across the
entire fertility range from 1.88 to 9.61. Whatever drives quality here, it is not the tokenizer.

Note also `Qwen3-30B-A3B` (MoE, ~3B active) beating `Qwen3-32B` (dense) by 3.76 chrF at
comparable total size, on the same tokenizer.

## What fertility *does* predict: cost

It was never in doubt that fertility is real; it simply lands on throughput, not quality.
Total generation time for the identical 1,800-item workload:

| Model | Fertility | Wall clock |
|---|---|---|
| `sarvam-30b` | 1.88 | **27 s** |
| `Qwen3-30B-A3B` | 8.22 | 186 s |
| `Qwen3-32B` | 8.22 | **999 s** |

Within one model, Llama-3.1-8B on 300 items: Hindi 14 s → Telugu **87 s** (6.2×), tracking
its per-language fertility (hi 2.67, te 13.30) almost exactly. Fertility is a deployment-cost
argument, not a capability argument.

## Other results

**Script compliance is near-universal and is not a differentiator in the native condition.**
99.8–99.9% for nine of ten models. Only `Mistral-7B-Instruct-v0.3` slips (97.4%), and it is also
the only model that runs away: length ratio **2.59** vs reference and 178 cap hits, against
0–1 for everyone else. It stands in for the absent Ministral 3, so this reflects that
substitution's limits, not current Mistral.

**Every model degrades on longer articles**, but not equally — `Llama-3.1-8B` loses 0.88 chrF
from short to long, `gemma-2-27b-it` loses 3.78. Robustness to article length varies more than
average quality does.

**No model was context-limited.** Zero truncation across all 18,000 items, including
`gemma-2-27b-it` at its 8192 ceiling — its efficient tokenizer offsets the small window.
Context length and fertility partly cancel.

**No model dominates.** Best-per-language changes four times across six languages: Llama (hi, bn),
Qwen3-30B-A3B (pa), Mistral-7B (ta), gemma-2 (te, en).

## Caveats — read before quoting any of the above

1. **chrF may not be discriminating summarization quality.** The whole field spread is
   25.65–30.14, and the literature this project is built on says exactly this would happen:
   IndicGenBench reports ChrF *because* token-level metrics fail, HinGE finds five standard NLG
   metrics ineffective, ITEM examines precisely this reliability question. The negative fertility
   result is currently **"fertility does not predict chrF"**, and only becomes "does not predict
   quality" once the LLM judge agrees. That is iteration 5, and it is now the critical path.
2. **Contamination is a live alternative explanation for the ranking.** The top two models are
   the two *oldest* — `gemma-2-27b-it` (Jun 2024) and `Llama-3.1-8B` (Jul 2024) — from the period
   when XL-Sum was a widely used benchmark. XL-Sum is BBC, publicly crawled, and in mT5's
   finetuning lineage. The ranking may partly measure XL-Sum familiarity rather than
   summarization skill.
3. **Sarvam's mid-pack placement is a finding about chrF, not yet about Sarvam.** An
   Indic-specialist model scoring 5th of 10 on Indic summarization should raise suspicion of the
   metric before the model — particularly given (1) and (2).
4. Gemma is represented by **Gemma 2**, two generations old; the 13–20B band has one model, so it
   supports no cross-family comparison.
