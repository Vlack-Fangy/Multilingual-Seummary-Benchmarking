"""Build the length-stratified XL-Sum evaluation sample.

Article length is a factor in this study, not a nuisance: at high fertility the
same article consumes several times the context budget, so any degradation that
only appears on long inputs would otherwise be invisible. Bucket boundaries are
therefore held IDENTICAL across languages — bucketing each language at its own
quantiles would hide exactly the effect we are trying to measure, because a
"long" Tamil article and a "long" Hindi article would be different sizes.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "xlsum"
OUT = ROOT / "data" / "eval"

LANGS = ["hi", "bn", "pa", "ta", "te", "en"]

# Character-count boundaries, shared by every language.
BUCKETS = [("short", 0, 1500), ("medium", 1500, 3500), ("long", 3500, 10**9)]

MIN_SUMMARY_CHARS = 20
MIN_TEXT_CHARS = 200


def load(lang, split="test"):
    p = RAW / f"{lang}_{split}.jsonl"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    summary_stats, all_rows = [], []

    for lang in LANGS:
        rows = load(lang)
        seen, clean = set(), []
        for r in rows:
            text, summ = (r.get("text") or "").strip(), (r.get("summary") or "").strip()
            if len(text) < MIN_TEXT_CHARS or len(summ) < MIN_SUMMARY_CHARS:
                continue
            if text in seen:  # XL-Sum has some duplicated articles
                continue
            seen.add(text)
            clean.append({"id": r.get("id"), "url": r.get("url"), "title": r.get("title"),
                          "text": text, "summary": summ})

        picked_counts = {}
        for name, lo, hi in BUCKETS:
            pool = [r for r in clean if lo <= len(r["text"]) < hi]
            rng.shuffle(pool)
            take = pool[: args.per_bucket]
            picked_counts[name] = (len(take), len(pool))
            for r in take:
                r = dict(r)
                r["lang"], r["bucket"] = lang, name
                r["text_chars"], r["summary_chars"] = len(r["text"]), len(r["summary"])
                r["text_words"] = len(r["text"].split())
                all_rows.append(r)

        summary_stats.append((lang, len(rows), len(clean), picked_counts))
        got = " ".join(f"{n}={picked_counts[n][0]}/{picked_counts[n][1]}" for n, _, _ in BUCKETS)
        print(f"{lang}: raw={len(rows):6} usable={len(clean):6} | {got}")

    dst = OUT / "xlsum_eval.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "n_items": len(all_rows), "per_bucket": args.per_bucket, "seed": args.seed,
        "buckets": {n: [lo, hi] for n, lo, hi in BUCKETS},
        "per_lang": {l: {k: v[0] for k, v in c.items()} for l, _, _, c in summary_stats},
    }
    (OUT / "xlsum_eval_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {dst}  ({len(all_rows)} items)")


if __name__ == "__main__":
    main()
