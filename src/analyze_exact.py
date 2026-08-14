"""T1/T2 analysis: script gap and language penalty, with the format confound split out.

Raw accuracy conflates two failures: the model got it wrong, and the model did not
emit an answer we could parse. Those are different capabilities, and the second is
systematically worse in romanized script — so a raw script gap silently mixes a
knowledge gap with a format-following gap.

Every number is therefore reported three ways:
  acc          raw accuracy (unparsed counted wrong)
  parsed_pct   fraction of items where an answer could be extracted  -> T3 compliance
  acc_parsed   accuracy among parseable items                        -> knowledge only
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "results" / "exact"
LANGS = ["hi", "bn", "pa", "ta", "te"]
# Native-script accuracy below this means the model cannot do the task at all,
# so any script gap is noise around zero rather than robustness.
FLOOR = 0.25

FAMILY = {"hi": "Indo-Aryan", "bn": "Indo-Aryan", "pa": "Indo-Aryan",
          "ta": "Dravidian", "te": "Dravidian"}


def load():
    rows = []
    for f in sorted(EX.glob("*__*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        raise SystemExit("no results yet")
    df["parsed"] = df["pred"].notna()

    g = (df.groupby(["model", "task", "lang", "script"])
           .agg(n=("correct", "size"), acc=("correct", "mean"),
                parsed_pct=("parsed", "mean")).reset_index())
    gp = (df[df.parsed].groupby(["model", "task", "lang", "script"])
            .correct.mean().rename("acc_parsed").reset_index())
    g = g.merge(gp, on=["model", "task", "lang", "script"], how="left")
    g.to_csv(EX / "summary_all.csv", index=False)

    for task in sorted(df.task.unique()):
        t = g[g.task == task]
        nat = t[t.script == "native"].set_index(["model", "lang"])
        rom = t[t.script == "roman"].set_index(["model", "lang"])
        if rom.empty:
            continue
        j = nat.join(rom, lsuffix="_n", rsuffix="_r", how="inner")
        j["gap_raw"] = 100 * (j.acc_n - j.acc_r)
        j["gap_parsed"] = 100 * (j.acc_parsed_n - j.acc_parsed_r)
        j["parse_drop"] = 100 * (j.parsed_pct_n - j.parsed_pct_r)
        # A model that cannot do the task in ANY script has no measurable script
        # robustness: Mistral-7B-v0.3 scores 0.06 native / 0.08 romanized on Hindi
        # gsm8k, a "-1.8pt gap" that would rank it above Sarvam. Floor cells are
        # excluded from gaps and family means rather than silently averaged in.
        j["at_floor"] = j.acc_n < FLOOR
        # Retention is scale-free and does not reward uniform failure.
        j["retention"] = (j.acc_r / j.acc_n).where(~j.at_floor)
        j = j.reset_index()
        n_floor = int(j.at_floor.sum())

        print(f"\n{'='*88}\nT1 SCRIPT GAP — task={task}   (positive = romanized is worse)\n{'='*88}")
        print(f"{'model':24} {'lang':5} {'raw gap':>9} {'gap|parsed':>11} {'retention':>10} {'note':>10}")
        for _, r in j.sort_values(["model", "lang"]).iterrows():
            if r.at_floor:
                print(f"{r.model:24} {r.lang:5} {'--':>9} {'--':>11} {'--':>10} "
                      f"{'AT FLOOR':>10}  (native acc {r.acc_n:.3f})")
            else:
                print(f"{r.model:24} {r.lang:5} {r.gap_raw:8.1f}pt {r.gap_parsed:10.1f}pt "
                      f"{r.retention:10.2f} {'':>10}")

        j["family"] = j.lang.map(FAMILY)
        ok = j[~j.at_floor]
        if n_floor:
            print(f"\n  ({n_floor} model-language cells excluded as at-floor, "
                  f"native acc < {FLOOR})")
        fam = ok.groupby(["model", "family"])[["gap_raw", "gap_parsed", "retention"]].mean().round(2)
        print(f"\n-- by language family (task={task}) --")
        print(fam.to_string())

        # T2: language penalty vs the English control
        en = t[(t.script == "native") & (t.lang == "en")].set_index("model").acc
        ind = (t[(t.script == "native") & (t.lang.isin(LANGS))]
               .groupby("model").acc.mean())
        lp = (100 * (en - ind)).dropna().round(1).sort_values()
        if len(lp):
            print(f"\n-- T2 language penalty, English minus Indic mean (task={task}) --")
            print(lp.to_string())

    print(f"\nwrote {EX/'summary_all.csv'}")


if __name__ == "__main__":
    main()
