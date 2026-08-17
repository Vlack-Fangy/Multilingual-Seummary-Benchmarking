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

    # ---- phase 2 additions ----
    # Gemma 4 has configurable thinking; NO_THINK is passed defensively because an
    # unnoticed thinking mode is what made Sarvam emit <think> on 20/20 items.
    "gemma-4-12B-it": dict(
        path="/home/models/gemma-4-12B-it", family="Gemma", band="7-12B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "gemma-4-26B-A4B-it": dict(
        path="/home/models/gemma-4-26B-A4B-it", family="Gemma", band="21-32B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "gemma-4-31B-it": dict(
        path="/home/models/gemma-4-31B-it", family="Gemma", band="21-32B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "gemma-3-27b-it": dict(
        path="/home/models/gemma-3-27b-it", family="Gemma", band="21-32B",
        vllm=dict(dtype="bfloat16")),
    # BF16 repos: the default Ministral-3 releases are natively FP8, which would
    # confound the Mistral arm against an otherwise bf16 field.
    # Ministral-3 needs a hybrid loader. load_format="mistral" wants
    # consolidated.safetensors, which we exclude at download as a duplicate of the
    # shards; the pure-HF path instead tries to build a PixtralProcessor (these are
    # multimodal) and dies on `[IMG]`. Mistral tokenizer+config bypasses the image
    # processor, while the DEFAULT weight loader reads the shards we actually have.
    "Ministral-3-8B": dict(
        path="/home/models/Ministral-3-8B-Instruct-2512-BF16", family="Mistral", band="7-12B",
        vllm=dict(dtype="bfloat16", tokenizer_mode="mistral", config_format="mistral")),
    "Ministral-3-14B": dict(
        path="/home/models/Ministral-3-14B-Instruct-2512-BF16", family="Mistral", band="13-20B",
        vllm=dict(dtype="bfloat16", tokenizer_mode="mistral", config_format="mistral")),
    "sarvam-m": dict(
        path="/home/models/sarvam-m", family="Sarvam", band="21-32B",
        vllm=dict(dtype="bfloat16"), chat_kwargs=NO_THINK),
    "aya-expanse-8b": dict(
        path="/home/models/aya-expanse-8b", family="Cohere", band="7-12B",
        vllm=dict(dtype="bfloat16")),
    "aya-expanse-32b": dict(
        path="/home/models/aya-expanse-32b", family="Cohere", band="21-32B",
        vllm=dict(dtype="bfloat16")),
    # OLMo-3.1-*-Think hard-codes an OPEN <think> tag into the generation prompt
    # ("<|im_start|>assistant\n<think>") with no toggle, so every reply begins
    # inside a reasoning block. At the normal 512-token cap the reasoning consumes
    # the budget and the answer is truncated — that measures truncation, not
    # ability. gen_budget widens the cap. thinking_only marks it for reporting as a
    # labelled separate entry: it cannot sit in a ranking of non-thinking models.
    "OLMo-3.1-32B-Think": dict(
        path="/home/models/OLMo-3.1-32B-Think", family="AI2", band="21-32B",
        vllm=dict(dtype="bfloat16"), gen_budget=6, thinking_only=True),
    # BharatGen Param2: 17B total / 2.4B active MoE — the same active budget as
    # sarvam-30b, from a different organisation. A third independent Indic-first
    # model, so it tests whether the robustness result generalises beyond Sarvam.
    # CAVEATS: (a) "Thinking" model whose chat template carries NO thinking toggle,
    # so it cannot be placed on the non-thinking footing every other contestant is
    # on — report it as a labelled separate entry, not inside the ranking;
    # (b) 4096 context, so T1/T2 only — it cannot take T5's 10k-char articles.
    "Param2-17B-A2.4B": dict(
        path="/home/models/Param2-17B-A2.4B-Thinking", family="BharatGen", band="13-20B",
        vllm=dict(dtype="bfloat16", trust_remote_code=True),
        gen_budget=6, thinking_only=True),
    "bloomz-7b1": dict(
        path="/home/models/bloomz-7b1", family="BigScience", band="7-12B",
        vllm=dict(dtype="bfloat16")),
}


# Cap context at 32k: enough for the longest article in the eval set even at the
# worst measured fertility (~10k chars x 1.7 tok/char), while keeping KV cache
# affordable. Models are given the smaller of this and their native limit.
MAX_CTX = 32768


def model_ctx(model_key):
    """Native context length from config, capped at MAX_CTX.

    gemma-2-27b-it is the binding constraint at 8192 — every other contestant
    supports 32k-262k. Articles that do not fit are truncated and flagged rather
    than dropped, so all models are still scored on the same item set.
    """
    import json
    spec = MODELS[model_key]
    for fn in ("config.json", "params.json"):
        p = Path(spec["path"]) / fn
        if p.exists():
            c = json.loads(p.read_text())
            t = c.get("text_config", c)
            n = t.get("max_position_embeddings")
            if n:
                return min(int(n), MAX_CTX)
    return 8192


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

# Brand names and acronyms are Latin by convention in every language — "FACEBOOK",
# "YouTube", "iPhone" say nothing about the script the model chose. They are dense
# enough in BBC copy to flip a short Gurmukhi reply to "Latin" on character count.
# Romanized Indic text is written in ordinary lowercase, so removing ALL-CAPS and
# CamelCase tokens cannot hide genuine romanization.
_BRANDISH = re.compile(r"\b(?:[A-Z]{2,}|[A-Za-z]+[A-Z][A-Za-z]*)\b")


def script_profile(text):
    """Fraction of alphabetic characters falling in each script. Digits, spaces and
    punctuation are excluded: they are script-neutral and would dilute the signal."""
    text = _BRANDISH.sub(" ", _URLISH.sub(" ", text))
    counts = {k: 0 for k in SCRIPT_RANGES}
    total = 0
    for ch in text:
        # NOT isalpha(): Indic vowel signs, virama and anusvara (U+093E etc) are
        # combining marks in category Mn/Mc, for which isalpha() is False. Filtering
        # on it dropped roughly a third of every Devanagari/Bengali/Tamil string and
        # biased the whole profile toward Latin. Membership of a script's codepoint
        # range is the right test; digits, spaces and punctuation fall outside all
        # ranges and are skipped anyway.
        if ch.isspace() or ch.isdigit():
            continue
        cp = ord(ch)
        for name, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                total += 1
                break
    if not total:
        return {k: 0.0 for k in counts}, 0
    return {k: v / total for k, v in counts.items()}, total


def classify_script(text, expected, romanized_ok=False, threshold=0.60):
    """Label what script a model actually replied in.

    Returns one of: expected script name, "Latin", "mixed", "empty".
    Whether a model complies with the requested script is a measured result of
    this study, not an assumption, so this runs on every generation.

    Dominance, not purity. Correct Indic writing routinely embeds Latin brand
    names, loanwords and glosses — "नए iPhone 11 Pro लॉन्च", "अवसाद (postpartum
    depression)". An 85% purity rule marked those as "mixed", which would
    under-count compliance, and would do so hardest in the romanized condition
    where loanword density is highest — biasing the very comparison T3 exists to
    make. A reply is in the expected script when that script dominates.
    """
    prof, total = script_profile(text)
    if total < 5:
        return "empty"
    lat = prof.get("Latin", 0.0)
    # Latin must be handled first and separately. When expected == "Latin" the
    # two variables below hold the SAME number, so a strict `lat > exp` test can
    # never fire — that bug reported 0/250 romanised compliance for all 9 models,
    # including replies that were flawless romanized Bengali.
    if expected == "Latin":
        return "Latin" if lat >= threshold else "mixed"
    exp = prof.get(expected, 0.0)
    if exp >= threshold and exp > lat:
        return expected
    if lat >= threshold and lat > exp:
        return "Latin"
    return "mixed"


# ------------------------------------------------- romanized language ID

# Script ranges cannot tell romanized Indic from English: both are Latin. A
# romanized-Bengali prompt answered in fluent English scored "compliant" under
# the script test, which is exactly the register defection Indi-RomCoM measures.
# High-frequency English function words are the cheap discriminator — they are
# grammatical, so genuine English is dense in them, while romanized Indic uses
# its own function words (ka/ke/ki, ne, hai, aur, jonyo, korte, cheyandi...).
_EN_FUNC = {
    "the", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in",
    "for", "on", "with", "as", "at", "by", "from", "that", "this", "these",
    "those", "it", "its", "you", "your", "we", "our", "they", "their", "he",
    "she", "his", "her", "and", "or", "but", "if", "then", "than", "so",
    "because", "while", "when", "where", "which", "who", "what", "how", "can",
    "could", "should", "would", "will", "have", "has", "had", "do", "does",
    "did", "not", "there", "here", "about", "into", "over", "after", "before",
}
# Function words shared across romanized Indic languages. Presence of these is
# positive evidence the reply is romanized Indic rather than English.
_INDIC_ROM = {
    "hai", "hain", "ka", "ke", "ki", "ko", "se", "me", "mein", "aur", "ye",
    "yeh", "wo", "woh", "kya", "nahi", "nahin", "bhi", "par", "liye", "kar",
    "karna", "hota", "hoti", "raha", "rahi", "tha", "thi", "aap", "aapka",
    "tum", "hum", "bilkul", "jonyo", "korte", "kore", "ami", "tumi", "amar",
    "eta", "ekta", "cheyandi", "chesi", "meeru", "naa", "adi", "unna",
    "seiya", "irukku", "avar", "enna", "vandhu", "da", "di", "na", "ne",
}


def romanized_profile(text):
    """(english_ratio, indic_ratio) over lowercase word tokens."""
    words = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if len(w) > 1]
    if not words:
        return 0.0, 0.0
    en = sum(1 for w in words if w in _EN_FUNC)
    ind = sum(1 for w in words if w in _INDIC_ROM)
    return en / len(words), ind / len(words)


def classify_romanized(text, en_thresh=0.18):
    """'english' | 'romanized-indic' | 'short'.

    Threshold from the observed split: genuine English prose runs ~0.25-0.35
    function-word density, romanized Indic ~0.02-0.10 even when code-mixed.
    """
    words = re.findall(r"[a-zA-Z']+", text)
    if len(words) < 8:
        return "short"
    en, ind = romanized_profile(text)
    if ind >= 0.04 and ind * 2 >= en:
        return "romanized-indic"
    return "english" if en >= en_thresh else "romanized-indic"


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
