"""Score persisted generations. Reads only from disk — never re-runs a model.

Metric choice follows the project's constraint that token-level metrics are
unreliable for these languages (IndicGenBench reports ChrF for exactly this
reason, and HinGE finds standard NLG metrics ineffective on code-mixed text):

  chrF / chrF++  character n-gram F-score. Script-agnostic, no tokenizer, works
                 on Devanagari and Latin alike. PRIMARY.
  ROUGE-L        whitespace-token LCS. Reported for continuity with prior work,
                 but flagged: it is not trustworthy on Brahmic scripts and must
                 not be the basis of a claim on its own.
  length_ratio   generated chars / reference chars. Catches degenerate output
                 that scores acceptably.
  script_ok      fraction replying in the requested script — a result, not a check.

  python src/score.py --condition native
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import sacrebleu

import common as C

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results" / "scores"


def rouge_l(ref, hyp):
    """LCS-based F1 over whitespace tokens.

    Deliberately NOT rouge_score: its default tokenizer strips non-ASCII, which
    silently zeroes every Indic-script row rather than failing loudly.
    """
    r, h = ref.split(), hyp.split()
    if not r or not h:
        return 0.0
    # O(len(r)*len(h)) LCS, rolling row.
    prev = [0] * (len(h) + 1)
    for i in range(1, len(r) + 1):
        cur = [0] * (len(h) + 1)
        for j in range(1, len(h) + 1):
            cur[j] = prev[j - 1] + 1 if r[i - 1] == h[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return 2 * p * rc / (p + rc)


def score_file(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        ref, hyp = r["reference"], r["gen"]
        expected = C.LANG_SCRIPT[r["lang"]]
        rec = dict(r)
        rec["chrf"] = sacrebleu.sentence_chrf(hyp, [ref], word_order=0).score
        rec["chrf_pp"] = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
        rec["rouge_l"] = 100.0 * rouge_l(ref, hyp)
        rec["length_ratio"] = len(hyp) / max(1, len(ref))
        rec["script_ok"] = int(r.get("script") == expected)
        rec["empty"] = int(len(hyp.strip()) == 0)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="native")
    args = ap.parse_args()

    files = sorted(GEN.glob(f"*__{args.condition}.jsonl"))
    if not files:
        raise SystemExit(f"no generation files for condition={args.condition}")

    allrows = []
    for f in files:
        rows = score_file(f)
        allrows.extend(rows)
        print(f"scored {f.name}: {len(rows)} rows")

    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(allrows)
    keep = ["model", "family", "band", "lang", "bucket", "chrf", "chrf_pp", "rouge_l",
            "length_ratio", "script_ok", "empty", "prompt_truncated", "chars_dropped",
            "gen_chars", "gen_tokens", "prompt_tokens", "finish_reason", "ctx"]
    df[keep].to_csv(OUT / f"rows_{args.condition}.csv", index=False)

    agg = (df.groupby(["family", "band", "model", "lang"], sort=False)
             .agg(chrf=("chrf", "mean"), chrf_pp=("chrf_pp", "mean"),
                  rouge_l=("rouge_l", "mean"), len_ratio=("length_ratio", "mean"),
                  script_ok=("script_ok", "mean"), empty=("empty", "mean"),
                  trunc=("prompt_truncated", "mean"), n=("chrf", "size"))
             .reset_index())
    agg.to_csv(OUT / f"by_lang_{args.condition}.csv", index=False)

    piv = agg.pivot_table(index=["family", "band", "model"], columns="lang",
                          values="chrf", sort=False)
    cols = [l for l in C.LANGS if l in piv.columns]
    piv = piv[cols]
    piv["indic_mean"] = piv[[c for c in cols if c != "en"]].mean(axis=1)
    piv = piv.sort_values("indic_mean", ascending=False)
    (OUT / f"chrf_{args.condition}.md").write_text(piv.round(2).to_markdown())

    print(f"\n=== chrF by model x language ({args.condition}) ===")
    print(piv.round(2).to_string())
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
