"""Tokenizer fertility across contestants x 6 languages, on FLORES+ devtest.

Fertility is the study's candidate explanatory variable, so it has to be measured
under ONE definition rather than mixed with published figures. The handoff brief
cites Llama-3.1-8B at 10.43 tokens/word on Brahmic means; a smoke test here gave
2.35 on Hindi. This script reports several denominators so the gap is diagnosable:

  tok_per_word  tokens / whitespace-delimited word   (the usual definition)
  tok_per_char  tokens / unicode character           (script-neutral)
  tok_per_byte  tokens / UTF-8 byte                  (1.0 == pure byte fallback)

tok_per_byte is the tell: a tokenizer with no vocabulary coverage for a script
degrades to byte fallback, and since Brahmic codepoints are 3 bytes in UTF-8,
that alone inflates tok_per_word by ~3x versus a tokenizer that has the script.
"""

import os
from pathlib import Path
MODELS_ROOT = Path(os.environ.get("MSB_MODELS_ROOT", "./models")).expanduser()
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FLORES = ROOT / "data" / "raw" / "flores"
OUT = ROOT / "results" / "fertility"

LANGS = ["hi", "bn", "pa", "ta", "te", "en"]

# Non-contestant tokenizers: judges, plus diagnostics for the 10.43 question —
# older small-vocab tokenizers that predate Indic coverage and should sit in the
# byte-fallback regime.
EXTRA = [
    ("gpt-oss-120b",    str(MODELS_ROOT / 'gpt-oss-120b'),       "OpenAI", "judge", "judge"),
    ("gpt-oss-20b",     str(MODELS_ROOT / 'gpt-oss-20b'),        "OpenAI", "judge", "judge"),
    ("Llama-2-7b-chat", str(MODELS_ROOT / 'Llama-2-7b-chat-hf'), "Llama",  "-",     "diagnostic"),
    ("Gemma-3-4B-it",   str(MODELS_ROOT / 'Gemma-3-4B-it'),      "Gemma",  "-",     "diagnostic"),
]


def build_models():
    """(label, path, family, band, role) — contestants derived from C.MODELS.

    This list used to be maintained by hand alongside common.MODELS. The two
    drifted: the nine-model T4/T5 roster was added to common.py only, so
    fertility.csv silently lacked five of them and max_tokens_for() — which sizes
    the summarization budget from tok_per_char — raised KeyError at generation
    time. Deriving from C.MODELS makes that class of drift impossible.
    """
    out = [(k, s["path"], s["family"], s["band"], "contestant")
           for k, s in sorted(C.MODELS.items())]
    have = {m[0] for m in out}
    return out + [e for e in EXTRA if e[0] not in have]


MODELS = build_models()


class Tok:
    """Uniform encode() over HF tokenizers and Mistral's Tekken."""

    def __init__(self, path):
        self.kind = None
        tekken = Path(path) / "tekken.json"
        hf_tok = Path(path) / "tokenizer.json"
        if not hf_tok.exists() and tekken.exists():
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            self.t = MistralTokenizer.from_file(str(tekken)).instruct_tokenizer.tokenizer
            self.kind = "tekken"
        else:
            from transformers import AutoTokenizer
            self.t = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.kind = "hf"

    def encode(self, s):
        if self.kind == "tekken":
            return self.t.encode(s, bos=False, eos=False)
        return self.t.encode(s, add_special_tokens=False)

    @property
    def vocab_size(self):
        try:
            return self.t.vocab_size if self.kind == "hf" else self.t.n_words
        except Exception:
            return None


def load_flores():
    """{lang: [sentence, ...]} from FLORES devtest, one sentence per line."""
    data = {}
    for lg in LANGS:
        lines = (FLORES / f"{lg}.txt").read_text(encoding="utf-8").splitlines()
        data[lg] = [s for s in lines if s.strip()]
    return data


def main():
    import warnings
    warnings.filterwarnings("ignore")

    OUT.mkdir(parents=True, exist_ok=True)
    flores = load_flores()
    n = min(len(v) for v in flores.values())
    # FLORES is sentence-aligned, so truncating to a common N keeps every
    # language on identical content — the comparison is then purely tokenizer.
    flores = {k: v[:n] for k, v in flores.items()}
    print(f"FLORES+ devtest: {n} aligned sentences/language\n")

    # Denominators are language properties, not model properties: compute once.
    denom = {}
    for lg, sents in flores.items():
        denom[lg] = {
            "words": sum(len(s.split()) for s in sents),
            "chars": sum(len(s) for s in sents),
            "bytes": sum(len(s.encode("utf-8")) for s in sents),
        }

    rows = []
    for label, path, family, band, role in MODELS:
        if not Path(path).exists():
            print(f"{label:24} SKIP (absent)")
            continue
        try:
            tok = Tok(path)
        except Exception as e:
            print(f"{label:24} FAIL {type(e).__name__}: {str(e)[:70]}")
            continue
        cells = {}
        for lg, sents in flores.items():
            ntok = sum(len(tok.encode(s)) for s in sents)
            d = denom[lg]
            cells[lg] = ntok / d["words"]
            rows.append({
                "model": label, "family": family, "band": band, "role": role,
                "lang": lg, "vocab_size": tok.vocab_size, "tokens": ntok,
                "tok_per_word": round(ntok / d["words"], 4),
                "tok_per_char": round(ntok / d["chars"], 4),
                "tok_per_byte": round(ntok / d["bytes"], 4),
            })
        brahmic = [cells[l] for l in ("hi", "bn", "pa", "ta", "te")]
        print(f"{label:24} " + " ".join(f"{l}={cells[l]:5.2f}" for l in LANGS)
              + f"  | brahmic_mean={sum(brahmic)/len(brahmic):5.2f}")

    (OUT / "fertility.json").write_text(json.dumps(rows, indent=2))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fertility.csv", index=False)

    piv = df[df.role != "diagnostic"].pivot_table(
        index=["family", "band", "model"], columns="lang",
        values="tok_per_word", sort=False)[LANGS]
    piv["brahmic_mean"] = piv[["hi", "bn", "pa", "ta", "te"]].mean(axis=1)
    (OUT / "fertility_table.md").write_text(piv.round(2).to_markdown())
    print(f"\nwrote {OUT}/fertility.{{json,csv}} and fertility_table.md")


if __name__ == "__main__":
    sys.exit(main())
