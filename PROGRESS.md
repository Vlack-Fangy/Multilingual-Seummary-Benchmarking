# Progress Report — Multilingual Benchmarking

A plain-language account of what we built, what we found, and every problem we hit
along the way. Written so someone new can follow it end to end.

**The question:** Indian users type their languages two ways — in the native script
(देवनागरी) and romanized in Latin letters (*"Bharat sarkar ne..."*). Models are
benchmarked almost entirely on the first. **How much capability is lost when the
same content arrives in Latin script?**

---

## The core idea, in one example

Sarvam released the *same* GSM8K maths problems in both scripts. Item #1, Hindi:

**Native:**
> जेनेट की बतखें प्रतिदिन 16 अंडे देती हैं। वह हर सुबह नाश्ते में तीन अंडे खाती है…

**Romanized:**
> Jennette ki batkhen din-din 16 ande deti hain. Vah har subah naaste mein teen ande khati hai…

Identical question, identical answer, only the script differs. So any accuracy
difference is *purely* about handling romanized text. We call this the **script gap**.

---

# TIMELINE

## Step 1 — Survey the ground (what actually exists)

Checked every model and dataset in the brief against reality. Several things in the
handoff notes did not survive contact:

| claimed | actual |
|---|---|
| `mistralai/Ministral-3-8B` | 404 — real id is `…-Instruct-2512` |
| CrossSum-IN "under google/" | `google/IndicGenBench_crosssum_in` |
| XL-Sum via parquet mirror | parquet auto-conversion **fails**; must pull `.tar.bz2` |
| `Llama-3.1-70B-Instruct` locally | an **empty 84 KB stub** |
| `Pixtral-12B-2409` locally | actually an IBM **Granite** model |
| `diffusiongemma-26B-A4B-it` | **block-diffusion**, not autoregressive — unusable |
| `mistral_nemo_instruct` | **bitsandbytes 4-bit** — would confound size with quantization |

**Lesson:** directory names and inherited notes are not evidence. Everything was
re-verified from `config.json`.

## Step 2 — Measure tokenizer fertility (tokens per word)

No GPU needed. 1,012 sentence-aligned FLORES sentences, identical content in all six
languages.

| model | hi | ta | te | Brahmic mean |
|---|---|---|---|---|
| `sarvam-30b` | 1.39 | 2.37 | 2.33 | **1.88** |
| `Mistral-Small-3.2-24B` | 1.95 | 3.61 | 3.71 | 3.08 |
| `Qwen3-*` (all sizes) | 4.76 | 10.06 | 11.40 | 8.22 |
| `Mistral-7B-v0.3` | 5.33 | 10.45 | 12.71 | **9.61** |

**A 4.4× spread at matched model size.** Same article costs one model 4× more tokens
than another.

**Problem found:** the brief cited Llama-3.1-8B at 10.43; our first measurement said
**2.35**, an apparent contradiction. Cause: we had measured **Hindi only**, and Hindi
is Llama's best Indic language by 5× (2.67 vs 13.30 Telugu). Across all five it is
**8.91** — the published figure was right. *Never quote single-language fertility.*

## Step 3 — First experiment: summarization (18,000 generations)

10 models × 6 languages × 3 article-length buckets × 100 items, on XL-Sum.

Then we scored it with **chrF** (a standard character-overlap metric) and got a
ranking that looked wrong: an 8B model on top, Sarvam 5th.

### The metric was broken, and here is the proof

Take one article about Indian athletes at the Athens Olympics. The human summary
mentions **two** things: shooters *and* badminton.

| model | covers shooting | covers badminton | chrF |
|---|---|---|---|
| `sarvam-30b` | ✅ | ✅ | **24.02** |
| `Llama-3.1-8B` | ✅ | ❌ | **34.65** |

Sarvam covered both topics and scored **10 points lower**. Why? Llama reused the
article's exact wording (`अपना भाग्य आज़माएँगे` appears in both article and gold
summary); Sarvam paraphrased. **chrF rewards copying.**

Confirmed at scale: we measured `copy_rate` (how much of each summary is copied
verbatim from the source) across all 18,000 rows:

| copy-rate quartile | mean chrF |
|---|---|
| most abstractive | 26.59 |
| most extractive | **29.76** |

**A 3.17-point gap from copying alone, against a 4.34-point spread across all ten
models** — roughly 73% of the apparent quality range was extractiveness, not skill.

### The judge settled it

An LLM judge (`gpt-oss-120b`) reading article + reference + candidate gave a
completely different ranking. The decisive test — which metric recovers the fact that
bigger models summarize better?

| | correlation with model size |
|---|---|
| **LLM judge** | **ρ = +0.905** (p = 0.0003) |
| chrF | ρ = +0.134 (p = 0.71) |

The judge recovers it; chrF does not. **chrF was retired.**

## Step 4 — Pivot: use Sarvam's own released benchmark

We found Sarvam publishes their eval datasets, including **`_roman` splits** for every
language. This removed our biggest blocker (we no longer needed to synthesize
romanized text) and gave exact-match scoring — no judge, no metric dispute.

**Validation against their published numbers** (the most important check in the project):

| | ours | Sarvam published | diff |
|---|---|---|---|
| GSM8K-IN | 0.900 | 0.92 | **0.020** ✅ |
| GSM8K-IN-R | 0.798 | 0.82 | **0.022** ✅ |
| MMLU-IN | 0.596 | 0.79 | **0.194** ❌ |

GSM8K reproduces within 2 points on both scripts — the harness is sound. MMLU does
not, almost certainly 0-shot vs the standard 5-shot protocol, so **MMLU is reported as
secondary only**.

## Step 5 — The main result (15 models)

| model | English | Indic native | **script gap (Dravidian)** |
|---|---|---|---|
| **`gemma-4-31B-it`** | 0.978 | 0.951 | **8.3** |
| `sarvam-30b` | 0.868 | 0.667 | **13.2** |
| `sarvam-m` | 0.946 | 0.900 | **16.8** |
| `gemma-4-12B-it` | 0.954 | 0.903 | 24.4 |
| `gemma-4-26B-A4B-it` | 0.944 | 0.914 | 25.2 |
| `Llama-3.1-8B` | 0.852 | 0.526 | 30.1 |
| `gemma-3-27b-it` | 0.952 | 0.907 | 34.2 |
| `Param2-17B-A2.4B`* | 0.816 | 0.730 | 38.0 |
| `Ministral-3-8B` | 0.828 | 0.609 | 51.4 |
| `gemma-2-27b-it` | 0.910 | 0.854 | 51.5 |
| `Qwen3-14B` | 0.954 | 0.866 | 61.4 |
| `Ministral-3-14B` | 0.954 | 0.830 | 66.3 |
| `Mistral-Small-3.2-24B` | 0.946 | 0.894 | 66.4 |
| `Qwen3-30B-A3B` | 0.956 | 0.860 | 68.1 |

\* *thinking-only model, not directly comparable*

**Most models lose 50–68 points when the same maths problem is romanized.** A handful
lose under 20.

### What explains it — four lines of evidence

**1. Not model size.** Scaling within a family barely moves it:
Qwen3 14B→30B: 61.4 → 68.1 (*worse*). Mistral 8B→14B→24B: 51.4 → 66.3 → 66.4 (flat).

**2. Not tokenizer efficiency.** Across all models, fertility vs quality: **r = −0.11
(p = 0.76)** — no relationship. Sarvam's fertility is nearly flat across languages
(1.39–2.37) yet its script gap varies 7×.

**3. Not "Indic-first training".** `Param2` (BharatGen) has the *same* 2.4B active
parameters as `sarvam-30b`, is Indic-first, trained on 22T tokens — and has **better
native Indic accuracy** (0.730 vs 0.667) but a **38.0-point gap vs Sarvam's 13.2**.

**4. It IS post-training on romanized data.** The controlled test — same backbone,
only post-training differs:

| | English | Indic native | gap (Dravidian) |
|---|---|---|---|
| `Mistral-Small-3.2-24B` (base) | 0.946 | 0.894 | **66.4** |
| `sarvam-m` (= same model + Sarvam's Indic post-training) | **0.946** | **0.900** | **16.8** |

**4× improvement, English unchanged, native Indic slightly better.** Sarvam documents
training on native / romanized / code-mixed in a 50/25/25 split.

### Gemma got better every generation

At matched ~27–31B size: `gemma-2` **51.5** → `gemma-3` **34.2** → `gemma-4-31B`
**8.3**. A 6× reduction, monotonic. `gemma-4-31B` is now the best model in the study on
every axis — and beats Sarvam at its own speciality.

## Step 6 — Ablation: can prompting substitute for training?

If you show a model 3 worked examples in romanized script, does the gap close?

| model | 0-shot gap | 3-shot gap | change |
|---|---|---|---|
| `gemma-4-31B` | 5.6 | 4.3 | −1.3 |
| `Qwen3-30B-A3B` | 46.4 | 45.7 | **−0.7** |
| `Mistral-Small-24B` | 41.6 | 34.6 | **−7.0** |

**No.** Romanized accuracy improves (Qwen 0.396→0.455) but native improves too, so the
*gap* is untouched. Few-shot makes models better at the task, not at reading romanized
text. **The deficit is representational — only training fixes it.**

## Step 7 — Register compliance (a different capability)

Given a romanized prompt, does the model *reply* in romanized script?

| model | replies romanized | **defects to English** | switches to native script |
|---|---|---|---|
| **`sarvam-m`** | **100.0%** | **0%** | 0 |
| `Llama-3.1-8B` | 91.9% | 2.4% | 12 |
| `gemma-4-31B` | 84.1% | 2.0% | 33 |
| `sarvam-30b` | 63.4% | **29.2%** | 0 |
| `Mistral-Small-24B` | 32.9% | **44.8%** | 41 |

Real examples, all from our run data:

**Defection to English** — `sarvam-30b` given a romanized Bengali prompt:
> **Q:** *Ajke ami third time-er jonyo amar driving test-e fail korechhi…*
> **A:** *"I'm so sorry to hear that you're going through this. It's completely understandable…"*

**Correct behaviour** — `sarvam-m`, same register held:
> **Q:** *Ekti train sokal 9:00 AM-e 80 km/h speed-e City A theke City B-r dike rouna dey…*
> **A:** *"Cholun, problem-ta step by step solve kora jak…"*

**Mid-reply script switch** — `gemma-4-31B` starts romanized, then flips to Bengali script:
> **A:** *"Ei problem-ti solve korar jonno amader step-by-step calculation korte hobe:*
> **১. প্রথম ট্রেনের এক ঘণ্টার দূরত্ব:**…"

**This ranks models differently from the accuracy test.** `sarvam-m` is first here;
`gemma-4-31B`, which wins on accuracy, is 5th. `Llama-3.1-8B` is 2nd here and near-last
on accuracy. Getting the answer right and staying in the user's register are separate
capabilities.

---

# EVERY PROBLEM WE HIT

Ordered by how badly each would have corrupted results.

### Silent-corruption class (would have produced confident, wrong numbers)

**1. Sarvam produced zero summaries and looked fine.** It reasons by default, ignores
its own `<|nothink|>` marker, and burned the whole token budget on English `<think>`
traces — **20/20 items, no summary at all**. Fixed by prefilling a closed
`<think></think>` block. Undetected, the flagship model would have contributed 1,800
empty cells.

**2. vLLM's `chat_template_kwargs` is accepted and silently ignored.** Our
"thinking off" setting never reached the template. Fixed by rendering the chat template
ourselves and passing **token ids** — control tokens only work if they carry their
special-token id; re-tokenizing a rendered string turns them into ordinary characters.

**3. Guided decoding collapsed multiple-choice to "A".** Constraining output to a bare
letter meant the model never got to reason: it answered **"A" 141 times out of 150**
against balanced gold — exactly chance accuracy (0.267). Fixed with free generation +
parsing.

**4. The same bug crippled the judge.** Forcing JSON output meant `gpt-oss-120b` never
reached its reasoning channel: coverage = 2 for *every* model, and one good summary
scored 1/1/1. Unconstrained, it separated the same five candidates 4/3/2/2/5 correctly.

**5. `isalpha()` silently dropped 40% of every Indic string.** Devanagari vowel signs
(`ा`, `ि`), virama and anusvara are Unicode *combining marks*, so `isalpha()` returns
False. Our script profiler counted **12 of 20** characters in
`अवसाद के बारे में पता चला` — biasing every classification toward "Latin".

**6. Script compliance reported 0/250 for all nine models.** When the expected script
*is* Latin, our dominance test compared a value against itself and could never pass.
The replies were flawless romanized Bengali.

**7. "Latin script" ≠ "romanized Indic".** After fixing #6, a romanized-Bengali prompt
answered in fluent **English** still counted as compliant — both are Latin. Fixed with
an English function-word density detector (English prose 0.44–0.48; romanized Indic
0.00–0.02). Without this, `sarvam-30b` looked like a 100% performer instead of 63.4%.

**8. Few-shot collapsed accuracy from 0.666 to 0.070.** Exemplars concatenated into one
user turn read as a pattern to continue: the model emitted
`#### 18 / #### 3 / #### 180 / #### 180…` in a loop. Few-shot must be **multi-turn**,
with exemplar answers in assistant turns. Fixed → 0.680.

**9. A model at chance looked "robust".** `Mistral-7B-v0.3` scores 0.062 native /
0.080 romanized on Hindi — a *negative* script gap that would have ranked it above
Sarvam. It simply cannot do the task in any script. Added a floor guard (native
accuracy < 0.25 → excluded) plus a scale-free retention measure.

**10. `rouge_score` returns 0.0 for all Indic text** — its default tokenizer strips
non-ASCII. Hand-rolled instead, and flagged as unreliable regardless.

### Environment and packaging class

**11. `transformers` 5.15.0 — released the day before — broke every Gemma 4 model.**
A new per-layer attribute guard raised `AmbiguousGlobalPerLayerAttributeError` on every
Gemma 4 config. vLLM pins only `transformers>=5.5.3`, so a fresh install took the broken
release. **Pinned 5.14.1.** A near-miss worth recording: an override got past the config
error and then failed on weight shapes (512 into 256), which read as "architecture
unsupported" — forcing it would have built a *structurally wrong model that still
produced numbers*.

**12. Ministral 3 ships natively FP8.** 9.8 GB for an 8B model. Would have confounded
the Mistral arm against an otherwise bf16 field. Switched to the `-BF16` repos.

**13. Two Mistral models, two different load recipes.** `Mistral-Small-3.2` needs all
three mistral flags (it ships only `tekken.json`); Ministral 3 needs mistral
tokenizer+config but the *default* weight loader.

**14. A fixed 8192 context crashed the first sweep** on an 8,937-token Llama prompt —
the study's own phenomenon, since token length is a function of fertility. Now per-model
native context with article-only truncation and per-row accounting.

### Data-plumbing class

**15. FLORES is gated** on both `facebook/flores` and `flores_plus`. Used the ungated
`google/IndicGenBench_flores_in` instead.

**16. Its rows are shuffled per language.** "Sentence #1" in Hindi was different content
from "sentence #1" in Tamil. Caught by an assertion; fixed by aligning on the English
key. Left unfixed, every cross-language fertility number would have mixed content
difficulty with language difficulty.

**17. Sarvam's own datasets are not schema-consistent.** `mmlu`/`trivia-qa` use
`choices` = array + `answer` = index; `arc-challenge` uses `choices` = {label, text} +
`answerKey` = letter; `trivia-qa` ships **no test split**, only validation. Both failed
producing *empty files* rather than loud errors, so the driver reported "done".

**18. My first fix for #17 broke the other two datasets** — an operator-precedence error
made the dict-detection branch raise on plain arrays. Caught only by testing the adapter
against all four datasets rather than the one being fixed.

**19. XL-Sum's "summary" is not a summary.** It is the BBC article's own teaser
standfirst. This is why every model sits near 2.66/5 on judged coverage — they are
scored against a target that is not quite the task.

### Housekeeping class

**20. `bloomz-7b1` shipped the same 14 GB weights twice** (safetensors *and* .bin).
**21. The download waiter required an index file** — but single-shard models
(`bloomz-7b1`, `gemma-4-12B-it`) ship one `model.safetensors` and no index, so it would
have waited forever on a complete download.
**22. GLM-4.7-Flash was queued as a contestant** — it is reserved as the *second judge*,
and a model cannot judge a field it competes in.
**23. The ablation would have overwritten the 0-shot results** it was meant to be
compared against — the shot count was missing from the output filename.
**24. A smoke-test file made the driver skip a real run.** `sarvam-30b`'s ablation was
"SKIP"ped because a 100-item test had written the summary file it checks for.
**25. `pkill -f "tail -f …log"` matched my own shell** and killed the command mid-run.

---

# WHAT'S PENDING

| item | status |
|---|---|
| **T4 pairwise judging** (win rates, 2 judges, both orderings) | generations done for 9 models; judging not run |
| **T5 summarization on the new roster** | 5 of 9 models have no summarization data |
| **MMLU 5-shot** to close the 19-point validation gap | not run |
| **T6 latency / GPU-hours** on a dedicated GPU | needs an exclusive card; current timings are from shared GPUs and invalid |
| **`Bhasha-Abhijnaanam`** to replace our hand-rolled romanized detector | pending |
| **Aya Expanse 8b/32b** | **blocked** — HTTP 403, needs a click-through on the HF model page |
| **CrossSum-IN** (contamination control) | downloaded, never run |
| **Human validation** | none — our largest methodological gap |
| Code-switched condition (Indi-RomCoM) | release location unstated in the paper |

---

# FUTURE SCOPE

**1. Close the loop on the mechanism.** Everything points to romanized training data,
but that is inference from four correlations. The decisive experiment is to
post-train one model with and without a romanized data slice and measure the gap
directly — `sarvam-m` shows this costs a post-training run, not a pretrain.

**2. Punjabi and Bengali need attention for opposite reasons.** Punjabi is *easiest* on
the accuracy test and *hardest* on nothing; Bengali is mid-pack on accuracy but where
register compliance collapses (`gemma-4-12B` 38%, `Ministral-3-14B` 26%). Which language
is "hard" depends entirely on which capability you measure.

**3. The two capabilities should be reported separately, always.** Accuracy-under-
romanization and register-compliance rank models differently. A leaderboard reporting
one implies the other and would mislead.

**4. Real user text, not synthetic romanization.** Sarvam's `_roman` splits are
model-generated. Dakshina holds *attested* native/romanized pairs; the honest next step
is checking whether our gaps hold on text people actually typed.

**5. Beyond news and maths.** Every dataset here is BBC news or grade-school maths.
The motivating use case — WhatsApp-style code-mixed writing — is untested.

**6. Cost.** Fertility does not predict quality but it does predict cost: a **37×
runtime spread** across models for identical work, and **6.2× within a single model**
across languages (Llama: Hindi 14s → Telugu 87s for 300 items). That is a deployment
argument nobody has quantified properly.

---

## How to reproduce

```bash
python src/download_data.py           # corpora
python src/fertility.py               # tokenizer fertility (CPU only)
bash   src/run_exact_all.sh 0 <model> # T1/T2 script gap + language penalty
python src/analyze_exact.py           # gaps, with floor guard + parse-rate split
bash   src/run_indivibe_all.sh 0      # T3/T4 generations
python src/analyze_indivibe.py        # register compliance
bash   src/run_ablation.sh 0          # few-shot ablation
python src/audit_outputs.py           # eyeball audit: lucky parses, leakage, degeneracy
```

Corpora are not redistributed: XL-Sum and CrossSum-IN are CC BY-NC-SA.
