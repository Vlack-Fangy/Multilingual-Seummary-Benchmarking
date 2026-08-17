"""T3 — script/language compliance on indivibe chat.

Does the model answer in the script it was addressed in? Recomputed from the
stored generations rather than from the run-time labels, so a classifier fix does
not require regenerating anything.

Native prompt  -> expected reply in that language's script.
Romanised prompt -> expected reply in Latin (romanized), per Sarvam's framing of
romanized Latin as the colloquial register. Replying in native script to a
romanized prompt is a register switch, which is what this measures.
"""
import json
from pathlib import Path

import pandas as pd

import common as C

ROOT = Path(__file__).resolve().parent.parent
IV = ROOT / "results" / "indivibe"


def main():
    rows = []
    for f in sorted(IV.glob("*__chat.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            exp = "Latin" if r["script"] == "romanised" else C.LANG_SCRIPT[r["lang"]]
            got = C.classify_script(r["gen"], exp)
            r["expected_script"] = exp
            r["got_script"] = got
            r["compliant"] = int(got == exp)
            rows.append(r)
    if not rows:
        raise SystemExit("no indivibe generations found")
    df = pd.DataFrame(rows)
    df.to_csv(IV / "compliance_rows.csv", index=False)

    piv = (df.pivot_table(index="model", columns="script", values="compliant", aggfunc="mean")
             * 100).round(1)
    piv["drop"] = (piv.get("native", 0) - piv.get("romanised", 0)).round(1)
    piv = piv.sort_values("romanised", ascending=False)
    print("T3 — script compliance %, by prompt script (n=250 per cell)\n")
    print(piv.to_string())

    print("\n-- where do non-compliant romanised replies go? --")
    bad = df[(df.script == "romanised") & (df.compliant == 0)]
    if len(bad):
        print(bad.groupby("model").got_script.value_counts().to_string())
    else:
        print("  none")

    print("\n-- romanised compliance by language --")
    lp = (df[df.script == "romanised"]
          .pivot_table(index="model", columns="lang", values="compliant", aggfunc="mean") * 100)
    cols = [c for c in ["hi", "bn", "pa", "ta", "te"] if c in lp.columns]
    print(lp[cols].round(1).sort_values(cols[0], ascending=False).to_string())
    print(f"\nwrote {IV/'compliance_rows.csv'}")


if __name__ == "__main__":
    main()
