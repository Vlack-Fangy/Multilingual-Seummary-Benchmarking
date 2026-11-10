"""T4 — win rates from pairwise verdicts.

Each (pair, item) was judged in BOTH orderings. We average them: a model "wins"
an item on a dimension only by the combined verdict, so a judge that simply
prefers whichever reply came second cannot manufacture a ranking.

Position consistency is reported as a diagnostic — it is the fraction of
(pair, item, dimension) cells where the two orderings agreed after un-flipping.
Low consistency means the two replies were genuinely close, not that the judge
is broken, but it bounds how much weight the ranking can carry.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "judge_pairwise" / "pairwise_raw.jsonl"
IV = ROOT / "results" / "indivibe"
DIMS = ["fluency", "script", "usefulness", "verbosity"]
TRUNC_LIMIT = 0.10      # above this a model's row is not comparable


def truncation():
    """{model: fraction of replies cut off by our own generation budget}"""
    out = {}
    for f in sorted(IV.glob("*__chat.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            out[f.stem.replace("__chat", "")] = \
                sum(1 for r in rows if r.get("finish_reason") == "length") / len(rows)
    return out


def main():
    rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no verdicts")
    print(f"{len(rows)} verdicts\n")

    # ---- gate: is the ranking measuring quality, or our token budget? ----
    # This existed only in my head when T4 was first analysed, and the answer turned
    # out to be "the budget": win rate correlated with truncation at r = -0.778,
    # because the 1024-token indivibe budget cut off 90% of sarvam-30b's replies.
    # The judge was correctly penalising replies that stop mid-sentence. Printing
    # this first makes the confound impossible to miss on any future run.
    tr = truncation()
    if tr:
        print("-- generation truncation (Defect 29 gate) --")
        bad = [m for m, v in tr.items() if v > TRUNC_LIMIT]
        for m, v in sorted(tr.items(), key=lambda x: -x[1]):
            print(f"  {m:24} {100*v:5.1f}%" + ("  <-- NOT COMPARABLE" if v > TRUNC_LIMIT else ""))
        if bad:
            print(f"\n  WARNING: {len(bad)} model(s) above {100*TRUNC_LIMIT:.0f}% truncation.")
            print("  Their win rates reflect the generation budget, not summary quality.")
            print("  Regenerate at a larger budget before reporting this as a ranking.")
        else:
            print("\n  all models within threshold; ranking is comparable")
        print()

    # ---- position consistency, per (pair, item, script, dimension) ----
    g = defaultdict(dict)
    for r in rows:
        key = (r["model_a"], r["model_b"], r["lang"], r["prompt_script"], r["item"])
        g[key][r["flip"]] = r
    agree = tot = 0
    per_dim = {d: [0, 0, 0] for d in DIMS}  # [agree, total, tie_involved]
    for v in g.values():
        if 0 in v and 1 in v:
            for d in DIMS:
                tot += 1
                ok = v[0][d] == v[1][d]
                agree += ok
                per_dim[d][0] += ok
                per_dim[d][1] += 1
                # A tie in one ordering and a winner in the other is a *soft*
                # disagreement: the judge did not reverse its preference, it only
                # wavered on whether the gap cleared its threshold. Hard flips
                # (A<->B) are the ones that actually threaten the ranking.
                if "tie" in (v[0][d], v[1][d]) and not ok:
                    per_dim[d][2] += 1
    if tot:
        print(f"position consistency: {100*agree/tot:.1f}%  ({tot} comparable cells)")
        print("  (both orderings averaged regardless, so this bounds confidence "
              "rather than biasing the ranking)")
        print(f"  {'dimension':12} {'consistent':>10} {'soft (tie/win)':>15} {'hard flip':>10}")
        for d in DIMS:
            a, n, soft = per_dim[d]
            hard = n - a - soft
            print(f"  {d:12} {100*a/n:9.1f}% {100*soft/n:14.1f}% {100*hard/n:9.1f}%")
        print()

    # ---- win rates: each verdict contributes 1 (win), 0.5 (tie), 0 (loss) ----
    score = defaultdict(lambda: defaultdict(float))
    n = defaultdict(lambda: defaultdict(float))
    for r in rows:
        for d in DIMS:
            a, b = r["model_a"], r["model_b"]
            v = r[d]
            sa = 1.0 if v == "A" else 0.0 if v == "B" else 0.5
            score[a][d] += sa
            score[b][d] += 1 - sa
            n[a][d] += 1
            n[b][d] += 1

    tbl = pd.DataFrame({d: {m: 100 * score[m][d] / n[m][d] for m in score} for d in DIMS})
    tbl["overall"] = tbl.mean(axis=1)
    tbl = tbl.sort_values("overall", ascending=False).round(1)
    print("T4 — win rate % (vs all other models, both orderings averaged)\n")
    print(tbl.to_string())

    # ---- native vs romanised: where does the ranking change? ----
    print("\n-- overall win rate by prompt script --")
    out = {}
    for sc in ("native", "romanised"):
        s2 = defaultdict(float); n2 = defaultdict(float)
        for r in [x for x in rows if x["prompt_script"] == sc]:
            for d in DIMS:
                a, b = r["model_a"], r["model_b"]
                sa = 1.0 if r[d] == "A" else 0.0 if r[d] == "B" else 0.5
                s2[a] += sa; s2[b] += 1 - sa; n2[a] += 1; n2[b] += 1
        out[sc] = {m: 100 * s2[m] / n2[m] for m in s2}
    df = pd.DataFrame(out).round(1)
    df["shift"] = (df["romanised"] - df["native"]).round(1)
    print(df.sort_values("romanised", ascending=False).to_string())
    tbl.to_csv(ROOT / "results" / "judge_pairwise" / "win_rates.csv")
    df.to_csv(ROOT / "results" / "judge_pairwise" / "win_rates_by_script.csv")


if __name__ == "__main__":
    main()
