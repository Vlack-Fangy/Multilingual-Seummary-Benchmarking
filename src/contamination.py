"""Contamination and extractiveness probes for the native-condition generations.

Two motivating observations:

1. The chrF ranking is topped by the two OLDEST models (gemma-2-27b-it, Jun 2024;
   Llama-3.1-8B, Jul 2024) — from exactly when XL-Sum was a widely used benchmark.
   XL-Sum is BBC content, publicly crawled, and in mT5's finetuning lineage.
2. On a worked example, the chrF winner had copied a phrase from the article
   rather than summarising it.

Both suggest the chrF ranking may reward copying and/or prior exposure rather
than summarisation skill. This measures that directly, on CPU, without re-running
any model:

  copy_rate    fraction of the candidate's character 5-grams that appear verbatim
               in the SOURCE ARTICLE. High = extractive, low = abstractive.
  ref_echo     fraction of the candidate's character 5-grams that appear in the
               GOLD but NOT in the article. Content matching the reference that
               could not have been copied from the source is the interesting
               signal — it is what a model that had *seen the reference* would
               produce.
"""
import json
from pathlib import Path

import common as C

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results" / "scores"
N = 5


def grams(s, n=N):
    s = "".join(s.split())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def main():
    articles = {}
    for line in (ROOT / "data" / "eval" / "xlsum_eval.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            articles[(r["lang"], r["id"])] = r["text"]

    rows = []
    for f in sorted(GEN.glob("*__native.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            art = articles.get((r["lang"], r["id"]), "")
            g, a, ref = grams(r["gen"]), grams(art), grams(r["reference"])
            if not g:
                continue
            # Reference n-grams not present in the article: content a model could
            # only produce by paraphrasing well or by having memorised the gold.
            ref_only = ref - a
            rows.append({
                "model": r["model"], "family": r["family"], "lang": r["lang"],
                "bucket": r["bucket"],
                "copy_rate": len(g & a) / len(g),
                "ref_echo": (len(g & ref_only) / len(ref_only)) if ref_only else None,
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "contamination_rows.csv", index=False)

    agg = (df.groupby("model", sort=False)
             .agg(copy_rate=("copy_rate", "mean"), ref_echo=("ref_echo", "mean"),
                  n=("copy_rate", "size")).reset_index())

    sc = pd.read_csv(OUT / "by_lang_native.csv")
    chrf = sc[sc.lang != "en"].groupby("model").chrf.mean().rename("chrf").reset_index()
    agg = agg.merge(chrf, on="model").sort_values("chrf", ascending=False)
    agg.to_csv(OUT / "contamination.csv", index=False)

    print("=== extractiveness vs chrF (native condition) ===")
    print(agg.round(4).to_string(index=False))

    import scipy.stats as ss
    r1, p1 = ss.pearsonr(agg.copy_rate, agg.chrf)
    r2, p2 = ss.pearsonr(agg.ref_echo, agg.chrf)
    print(f"\ncorr(copy_rate, chrF) = {r1:.3f} (p={p1:.4f})")
    print(f"corr(ref_echo,  chrF) = {r2:.3f} (p={p2:.4f})")


if __name__ == "__main__":
    main()
