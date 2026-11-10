"""T5 — win rates from pairwise summarization verdicts.

Each (pair, item) was judged in BOTH orderings and the two are averaged, so a judge
that simply prefers whichever summary came second cannot manufacture a ranking.

The truncation column is printed FIRST and is not optional. T4's ranking turned out
to correlate with generation truncation at r = -0.778 -- our token budget, not model
quality -- and that was discovered only after the run. Any model above the threshold
here invalidates its own row, so the check gates the ranking rather than annotating
it afterwards.

Position consistency is reported per dimension, split into soft disagreements (a tie
in one ordering, a winner in the other -- the judge wavered on whether the gap cleared
its threshold) and hard flips (A<->B, an actual reversal). Only hard flips threaten
the ranking; raw agreement understates reliability whenever ties are common.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "judge_pairwise_summ" / "pairwise_summ_raw.jsonl"
GEN = ROOT / "results" / "generations"
DIMS = ["coverage", "faithfulness", "fluency", "verbosity"]
TRUNC_LIMIT = 0.10      # above this a model's row is not comparable


def truncation():
    """{model: fraction of summaries cut off by our budget}"""
    out = {}
    for m in C.ROSTER:
        p = GEN / f"{m}__native.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            out[m] = sum(1 for r in rows if r.get("finish_reason") == "length") / len(rows)
    return out


def cut_items():
    """{(model, lang, id)} whose generation was truncated.

    Truncation does not merely shorten a summary here, it breaks extraction. A
    thinking model cut off mid-deliberation ends on scratch work -- "Word count:
    Brett(1) Painter,(2) 25,(3)..." or "गणना: 1 एनडीए, 2 के, 3 तीन..." -- so the
    trailing block is counting notes rather than a summary. Six such rows reached
    the judged sample for Param2-17B-A2.4B (2.4% of its items), each handing the
    judge nonsense to score.

    Dropping any comparison where EITHER side was truncated is the same
    both-finished control applied to T4, and it costs almost nothing for the seven
    models that truncate at 0.0-0.2%.
    """
    bad = set()
    for m in C.ROSTER:
        p = GEN / f"{m}__native.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("finish_reason") == "length":
                    bad.add((m, r["lang"], r["id"]))
    return bad


def win_rates(rows, models=None):
    s, n = defaultdict(float), defaultdict(float)
    for r in rows:
        for d in DIMS:
            a, b = r["model_a"], r["model_b"]
            v = r[d]
            sa = 1.0 if v == "A" else 0.0 if v == "B" else 0.5
            s[(a, d)] += sa
            s[(b, d)] += 1 - sa
            n[(a, d)] += 1
            n[(b, d)] += 1
    ms = models or sorted({m for m, _ in s})
    return pd.DataFrame({d: {m: 100 * s[(m, d)] / n[(m, d)] for m in ms if n[(m, d)]}
                         for d in DIMS})


def main():
    if not P.exists():
        raise SystemExit(f"no verdicts at {P}")
    rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no verdicts")
    print(f"{len(rows)} verdicts\n")

    # ---- gate: is any model's row invalidated by our own generation budget? ----
    tr = truncation()
    print("-- generation truncation (the Defect 29 gate) --")
    bad = []
    for m, v in sorted(tr.items(), key=lambda x: -x[1]):
        flag = "  <-- NOT COMPARABLE" if v > TRUNC_LIMIT else ""
        if v > TRUNC_LIMIT:
            bad.append(m)
        print(f"  {m:24} {100*v:5.1f}%{flag}")
    print(f"\n  {'all models within threshold; ranking is comparable' if not bad else 'EXCLUDE: ' + ', '.join(bad)}\n")

    # ---- drop comparisons where either side's generation was truncated ----
    bad = cut_items()
    before = len(rows)
    rows = [r for r in rows
            if (r["model_a"], r["lang"], r["item"]) not in bad
            and (r["model_b"], r["lang"], r["item"]) not in bad]
    dropped = before - len(rows)
    print(f"-- dropped {dropped} verdicts ({100*dropped/before:.1f}%) where a summary was "
          f"truncated --\n   truncation breaks extraction, not just length: a thinking model\n"
          f"   cut off mid-deliberation ends on word-counting scratch work.\n")

    # ---- position consistency, per dimension ----
    g = defaultdict(dict)
    for r in rows:
        g[(r["model_a"], r["model_b"], r["lang"], r["item"])][r["flip"]] = r
    per = {d: [0, 0, 0] for d in DIMS}
    tot = agree = 0
    for v in g.values():
        if 0 in v and 1 in v:
            for d in DIMS:
                tot += 1
                ok = v[0][d] == v[1][d]
                agree += ok
                per[d][0] += ok
                per[d][1] += 1
                if "tie" in (v[0][d], v[1][d]) and not ok:
                    per[d][2] += 1
    if tot:
        print(f"position consistency: {100*agree/tot:.1f}%  ({tot} comparable cells)")
        print(f"  {'dimension':14}{'consistent':>12}{'soft (tie/win)':>16}{'hard flip':>11}")
        for d in DIMS:
            a, n_, soft = per[d]
            print(f"  {d:14}{100*a/n_:11.1f}%{100*soft/n_:15.1f}%{100*(n_-a-soft)/n_:10.1f}%")
        print()

    # ---- overall ----
    tbl = win_rates(rows)
    tbl["overall"] = tbl.mean(axis=1)
    tbl = tbl.sort_values("overall", ascending=False).round(1)
    print("T5 — summarization win rate % (vs all other models, both orderings averaged)\n")
    print(tbl.to_string())

    # ---- per language: does the ranking hold, or is it language-specific? ----
    print("\n-- overall win rate by language --")
    out = {}
    for lg in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lg]
        out[lg] = win_rates(sub).mean(axis=1)
    df = pd.DataFrame(out).round(1)
    df["spread"] = (df.max(axis=1) - df.min(axis=1)).round(1)
    print(df.loc[tbl.index].to_string())

    d = P.parent
    tbl.to_csv(d / "win_rates_summ.csv")
    df.to_csv(d / "win_rates_summ_by_lang.csv")
    print(f"\nwrote {d/'win_rates_summ.csv'} and {d/'win_rates_summ_by_lang.csv'}")


if __name__ == "__main__":
    main()
