"""Generate the results tables for RESULTS.md from source CSVs.

The project rule is that every figure in the writeup must be re-derivable from the
data rather than transcribed by hand -- transcription is how a stale number survives
a re-run and ends up in a submission. This emits markdown that can be pasted, or
written straight to a file with --out.

  python src/make_results.py --out results/TABLES.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "results" / "exact"
FERT = ROOT / "results" / "fertility" / "fertility.csv"
LANGS = ["hi", "bn", "pa", "ta", "te"]
FAMILY = {"hi": "Indo-Aryan", "bn": "Indo-Aryan", "pa": "Indo-Aryan",
          "ta": "Dravidian", "te": "Dravidian"}
FLOOR = 0.25
OUT = []


def w(s=""):
    OUT.append(s)


def load_exact(shots=0):
    """Load exact-match results for ONE shot setting.

    The k-shot ablation files store task="gsm8k" identically to the 0-shot files --
    the shot count lives only in the filename. A plain glob therefore pools 0-shot
    and 3-shot rows into a single accuracy for every ablated model, which silently
    corrupted the headline gaps for sarvam-30b, Mistral-Small-3.2-24B and
    gemma-4-31B-it. Shots are parsed from the filename and filtered here.
    """
    rows, bad, files = [], 0, []
    for f in sorted(EX.glob("*__*.jsonl")):
        stem = f.stem                      # model__task[_Nshot]
        tail = stem.split("__", 1)[1] if "__" in stem else stem
        k = 0
        if "_" in tail and tail.rsplit("_", 1)[1].endswith("shot"):
            k = int(tail.rsplit("_", 1)[1].removesuffix("shot") or 0)
        if k != shots:
            continue
        files.append(f.name)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    df = pd.DataFrame(rows)
    return df[df.model.isin(C.ROSTER)], bad


def coverage(df):
    """Which roster cells actually exist, and are they post-fix?

    Printed first and deliberately: a table of eight models where three were scored
    under the old parser is not a comparison, and that has to be visible rather than
    discovered later.
    """
    w("## Data coverage\n")
    w("`full_text` = generations stored uncut, so the run is post-Defect-27 and can be")
    w("re-scored offline. A model without it was scored under the old parser and its")
    w("numbers are **not comparable** to the rest.\n")
    w("| model | task | rows | truncated | full_text |")
    w("|---|---|---|---|---|")
    for m in C.ROSTER:
        for t in ("gsm8k", "mmlu"):
            s = df[(df.model == m) & (df.task == t)]
            if s.empty:
                w(f"| `{m}` | {t} | — | — | **MISSING** |")
                continue
            cut = 100 * (s.finish_reason == "length").mean()
            has = "gen" in s.columns and s["gen"].notna().any()
            uncut = has and (s["gen"].astype(str).str.len() >= 1200).any()
            # A run is post-fix if some stored generation exceeds the old 1200 clip,
            # or if nothing was long enough to have been clipped in the first place.
            longest = s["gen"].astype(str).str.len().max() if has else 0
            tag = "yes" if (uncut or (has and longest < 1200)) else ("clipped" if has else "no")
            w(f"| `{m}` | {t} | {len(s)} | {cut:.1f}% | {tag} |")
    w()


def script_gap(df):
    df = df[df.lang.isin(LANGS)].copy()
    df["parsed"] = df["pred"].notna()
    df["cut"] = df.finish_reason.eq("length")
    g = (df.groupby(["model", "task", "lang", "script"])
           .agg(n=("correct", "size"), acc=("correct", "mean"),
                cut=("cut", "mean")).reset_index())
    gp = (df[df.parsed].groupby(["model", "task", "lang", "script"])
            .correct.mean().rename("accp").reset_index())
    g = g.merge(gp, on=["model", "task", "lang", "script"], how="left")
    for task in ("gsm8k", "mmlu"):
        t = g[g.task == task]
        nat = t[t.script == "native"].set_index(["model", "lang"])
        rom = t[t.script == "roman"].set_index(["model", "lang"])
        if rom.empty:
            continue
        j = nat.join(rom, lsuffix="_n", rsuffix="_r", how="inner").reset_index()
        j["at_floor"] = j.acc_n < FLOOR
        j["gap"] = 100 * (j.acc_n - j.acc_r)
        j["gapp"] = 100 * (j.accp_n - j.accp_r)
        j["family"] = j.lang.map(FAMILY)
        ok = j[~j.at_floor]
        w(f"### T1 script gap by language family — {task}\n")
        w("Positive = romanised is worse. Cells with native accuracy below "
          f"{FLOOR} are excluded as at-floor.\n")
        w("| model | Dravidian | Indo-Aryan | ratio | max truncation |")
        w("|---|---|---|---|---|")
        fam = ok.groupby(["model", "family"]).gap.mean().unstack()
        cut = ok.groupby("model")[["cut_n", "cut_r"]].max().max(axis=1)
        # A model is only comparable if every non-floor language reported. A rerun
        # in flight leaves partial cells, and averaging them would silently compare
        # a 2-language mean against a 5-language one.
        seen = ok.groupby("model").lang.nunique()
        expected = j.groupby("model").lang.nunique()
        if "Dravidian" in fam and "Indo-Aryan" in fam:
            for m in fam.sort_values("Dravidian", na_position="last").index:
                d, i = fam.loc[m, "Dravidian"], fam.loc[m, "Indo-Aryan"]
                if pd.isna(d) or pd.isna(i) or seen.get(m, 0) < expected.get(m, 0):
                    w(f"| `{m}` | — | — | — | *incomplete: "
                      f"{seen.get(m,0)}/{len(LANGS)} langs* |")
                    continue
                r = f"{d/i:.1f}x" if i > 0 else "n/a"
                w(f"| `{m}` | {d:.1f} | {i:.1f} | {r} | {100*cut.get(m,0):.0f}% |")
        w()
        nf = j[j.at_floor]
        if len(nf):
            w("Excluded as at-floor: " + ", ".join(
                f"`{r.model}`/{r.lang} (native {r.acc_n:.3f})"
                for _, r in nf.iterrows()) + "\n")


def fertility():
    if not FERT.exists():
        return
    f = pd.read_csv(FERT)
    f = f[f.model.isin(C.ROSTER)]
    p = f.pivot_table(index="model", columns="lang", values="tok_per_word")
    p["Brahmic mean"] = p[[c for c in LANGS if c in p]].mean(axis=1)
    p = p.sort_values("Brahmic mean")
    w("### T6 tokenizer fertility (tokens per word, FLORES devtest)\n")
    cols = [c for c in ["hi", "bn", "pa", "ta", "te", "en"] if c in p]
    w("| model | " + " | ".join(cols) + " | Brahmic mean |")
    w("|---" * (len(cols) + 2) + "|")
    for m, r in p.iterrows():
        w(f"| `{m}` | " + " | ".join(f"{r[c]:.2f}" for c in cols)
          + f" | **{r['Brahmic mean']:.2f}** |")
    w()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    df, bad = load_exact()
    if bad:
        w(f"*note: {bad} unparseable line(s) skipped — a run is still writing*\n")
    if df.empty:
        raise SystemExit("no roster results yet")
    w(f"*Generated from source data. Roster: {len(C.ROSTER)} models.*\n")
    coverage(df)
    script_gap(df)
    fertility()
    text = "\n".join(OUT)
    if a.out:
        Path(ROOT / a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out} ({len(OUT)} lines)")
    else:
        print(text)


if __name__ == "__main__":
    main()
