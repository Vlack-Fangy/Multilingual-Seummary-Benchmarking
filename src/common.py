"""Shared config: model registry, length budgets, script identification, prompts."""
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FERTILITY_CSV = ROOT / "results" / "fertility" / "fertility.csv"

LANGS = ["hi", "bn", "pa", "ta", "te", "en"]

LANG_NAME = {"hi": "Hindi", "bn": "Bengali", "pa": "Punjabi",
             "ta": "Tamil", "te": "Telugu", "en": "English"}

# Native script per language. Punjabi has two; XL-Sum's BBC Punjabi is Gurmukhi.
LANG_SCRIPT = {"hi": "Devanagari", "bn": "Bengali", "pa": "Gurmukhi",
               "ta": "Tamil", "te": "Telugu", "en": "Latin"}

# Median gold summary length per language, from the 1800-item eval set. The prompt
# states a word target (models follow word counts far better than character counts);
# the character budget below is what actually bounds generation.
GOLD_WORDS = {"hi": 27, "bn": 20, "pa": 27, "ta": 18, "te": 18, "en": 21}
GOLD_CHARS = {"hi": 134, "bn": 144, "pa": 134, "ta": 171, "te": 144, "en": 122}

# Generation is capped in CHARACTERS, then converted to a token ceiling per model
# using that model's measured tok_per_char. A uniform max_tokens would give
# Llama ~1/6 the Telugu output of Sarvam and would manufacture the very
# fertility->quality correlation this study exists to test.
# 4x the median gold length: generous enough that the cap is a runaway guard,
# not a binding constraint on well-behaved output.
CHAR_BUDGET_MULTIPLE = 4

# Qwen3 hybrids default to thinking ON. Decision 3 fixes thinking OFF across the
# contestant set: summarization is not a reasoning task, and leaving it on would
# change token accounting as well as quality.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

# (key, path, family, band, extra vllm kwargs)
MODELS = {
    "sarvam-30b": dict(
        path="/home/models/sarvam-30b", family="Sarvam", band="21-30B",
        # Stored fp32 (128.6GB) with torch_dtype unset: dtype="auto" would eat the
        # card and leave no room for KV cache. Must be cast explicitly.
        vllm=dict(dtype="bfloat16", trust_remote_code=True),
        # Sarvam reasons by default and ignores its own <|nothink|> marker: with
        # enable_thinking=False correctly rendered, it still emitted English <think>
        # traces and burned the entire token budget before writing any summary.
        # Prefilling a closed thinking block forces it straight to content — this is
        # the template's own convention for a non-reasoning assistant turn.
        chat_kwargs=NO_THINK, prefill=["<think>", "</think>"]),
    "Qwen3-30B-A3B": dict(
        path="/home/models/Qwen3-30B-A3B-Instruct-2507", family="Qwen", band="21-30B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "Mistral-Small-3.2-24B": dict(
        path="/home/models/Mistral-Small-3.2-24B-Instruct-2506", family="Mistral", band="21-30B",
        # Ships tekken.json and no HF tokenizer.
        vllm=dict(dtype="bfloat16", tokenizer_mode="mistral", config_format="mistral",
                  load_format="mistral")),
    "Qwen3-32B": dict(
        path="/home/models/Qwen3-32B", family="Qwen", band="21-30B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "gemma-2-27b-it": dict(
        path="/home/models/gemma-2-27b-it", family="Gemma", band="21-30B",
        vllm=dict(dtype="bfloat16")),
    "Qwen3-14B-Instruct": dict(
        path="/home/models/Qwen3-14B-Instruct", family="Qwen", band="13-20B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "Qwen3.5-9B": dict(
        path="/home/models/Qwen3.5-9B", family="Qwen", band="7-12B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "Qwen3-8B": dict(
        path="/home/models/Qwen3-8B", family="Qwen", band="7-12B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "Llama-3.1-8B-Instruct": dict(
        path="/home/models/Llama-3.1-8B-Instruct", family="Llama", band="7-12B",
        vllm=dict(dtype="bfloat16")),
    "Mistral-7B-Instr-v0.3": dict(
        path="/home/models/Mistral-7B-Instruct-v0.3", family="Mistral", band="7-12B",
        vllm=dict(dtype="bfloat16")),
}


def load_tok_per_char():
    """{model: {lang: tokens per character}} from the iteration-1 measurement."""
    out = {}
    with FERTILITY_CSV.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["model"], {})[r["lang"]] = float(r["tok_per_char"])
    return out


def max_tokens_for(model_key, lang, tpc=None):
    """Token ceiling equivalent to the language's character budget for THIS model."""
    tpc = tpc or load_tok_per_char()
    budget_chars = GOLD_CHARS[lang] * CHAR_BUDGET_MULTIPLE
    per_char = tpc[model_key][lang]
    return int(budget_chars * per_char) + 32  # +32 for preamble/punctuation slack


# ---------------------------------------------------------------- script ID

SCRIPT_RANGES = {
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali":    [(0x0980, 0x09FF)],
    "Gurmukhi":   [(0x0A00, 0x0A7F)],
    "Tamil":      [(0x0B80, 0x0BFF)],
    "Telugu":     [(0x0C00, 0x0C7F)],
    "Latin":      [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}


# URLs, emails and @handles are Latin by necessity in any language, so they are
# evidence about the source text, not about the script the model chose to write in.
# BBC copy carries them often enough that leaving them in misclassifies otherwise
# clean native-script summaries as "mixed".
_URLISH = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+|\b\S+\.(?:com|org|net|co\.uk|in)\b/?\S*)",
                     re.IGNORECASE)


def script_profile(text):
    """Fraction of alphabetic characters falling in each script. Digits, spaces and
    punctuation are excluded: they are script-neutral and would dilute the signal."""
    text = _URLISH.sub(" ", text)
    counts = {k: 0 for k in SCRIPT_RANGES}
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        cp = ord(ch)
        for name, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                break
    if not total:
        return {k: 0.0 for k in counts}, 0
    return {k: v / total for k, v in counts.items()}, total


def classify_script(text, expected, romanized_ok=False, threshold=0.85):
    """Label what script a model actually replied in.

    Returns one of: expected script name, "Latin", "mixed", "empty".
    Whether a model complies with the requested script is a measured result of
    this study, not an assumption, so this runs on every generation.
    """
    prof, total = script_profile(text)
    if total < 5:
        return "empty"
    if prof.get(expected, 0) >= threshold:
        return expected
    if prof.get("Latin", 0) >= threshold:
        return "Latin"
    return "mixed"


# ---------------------------------------------------------------- prompts

_NATIVE = (
    "Summarize the following {language} news article.\n\n"
    "Requirements:\n"
    "- Write the summary in {language}, using {script} script — the same script as the article.\n"
    "- About {words} words, one or two sentences.\n"
    "- Output only the summary, with no preamble, title, or explanation.\n\n"
    "Article:\n{article}\n\nSummary:"
)

_ROMAN = (
    "Summarize the following {language} news article. The article is written in "
    "romanized {language} (Latin script).\n\n"
    "Requirements:\n"
    "- Write the summary in romanized {language} using Latin script, matching the "
    "article's style. If you cannot, write the summary in English.\n"
    "- About {words} words, one or two sentences.\n"
    "- Output only the summary, with no preamble, title, or explanation.\n\n"
    "Article:\n{article}\n\nSummary:"
)


def build_prompt(article, lang, condition="native"):
    """English instructions in every cell, so input script is the only variable."""
    tpl = _ROMAN if condition == "roman" else _NATIVE
    return tpl.format(language=LANG_NAME[lang], script=LANG_SCRIPT[lang],
                      words=GOLD_WORDS[lang], article=article)
