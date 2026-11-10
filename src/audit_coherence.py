"""Per-pipeline coherence audit of raw model output.

Aggregate scores hide catastrophic failures. Everything this file checks for has
already happened at least once in this project and produced plausible numbers:

  * Sarvam emitting <think> traces instead of summaries on 20/20 items
  * a few-shot prompt collapsing into `#### 18 / #### 3 / #### 180 ...`
  * English replies counted as compliant because Latin script matched
  * MCQ answers collapsing to "A" 141/150 under guided decoding

So every pipeline that emits text gets its own pass. Run it after any generation
or judging run, not just when something looks wrong.

  python src/audit_coherence.py                 # all pipelines
  python src/audit_coherence.py --pipeline t5   # one
  python src/audit_coherence.py --samples 4     # more text to eyeball
"""
import argparse
import json
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RNG = random.Random(20260817)

# Markers that should never survive into a finished reply. Reasoning channels and
# chat-template scaffolding leaking through means the template was applied wrong.
LEAK = ["<think>", "</think>", "<|channel|>", "<|start|>", "<|end|>", "<|im_start|>",
        "<|assistant|>", "[INST]", "<start_of_turn>", "<|nothink|>"]


def degenerate(text, n=5, thresh=0.30):
    """True if one word n-gram occupies more than `thresh` of all n-grams.

    Catches the repetition-loop failure mode. Also returns the compression ratio:
    zlib below ~0.25 on natural text means it is almost all repeats.
    """
    w = text.split()
    if len(w) < n * 3:
        return False, 1.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    top = Counter(grams).most_common(1)[0][1] / len(grams)
    raw = text.encode("utf-8")
    ratio = len(zlib.compress(raw)) / max(len(raw), 1)
    return (top > thresh or ratio < 0.18), ratio


def flag_row(text, finish, expected_script=None):
    """Return list of problem tags for one generation."""
    f = []
    t = (text or "").strip()
    if not t:
        return ["EMPTY"]
    if finish == "length":
        f.append("TRUNCATED")
    deg, _ = degenerate(t)
    if deg:
        f.append("DEGENERATE")
    for m in LEAK:
        if m in t:
            f.append(f"LEAK:{m}")
            break
    if expected_script:
        got = C.classify_script(t, expected_script)
        if got != expected_script:
            f.append(f"SCRIPT:{got}")
    if len(t) < 15:
        f.append("VERY_SHORT")
    return f


def report(name, rows, samples, get_text, get_finish, get_script=None, extra=None):
    """Shared reporting shape: flag table, then real text to read."""
    print(f"\n{'='*74}\n{name}   ({len(rows)} rows)\n{'='*74}")
    if not rows:
        print("  no data")
        return
    tags = defaultdict(Counter)
    flagged = []
    for r in rows:
        exp = get_script(r) if get_script else None
        fl = flag_row(get_text(r), get_finish(r), exp)
        for t in fl:
            tags[r.get("model", "?")][t.split(":")[0]] += 1
        if fl:
            flagged.append((r, fl))
    if tags:
        alltags = sorted({t for c in tags.values() for t in c})
        print(f"\n  {'model':24}" + "".join(f"{t:>13}" for t in alltags))
        for m in sorted(tags):
            n = sum(1 for r in rows if r.get("model") == m) or 1
            print(f"  {m:24}" + "".join(
                f"{100*tags[m][t]/n:12.1f}%" for t in alltags))
    else:
        print("\n  no rows flagged on any check")
    if extra:
        extra(rows)
    # --- the part that matters: actual text ---
    print(f"\n  -- {samples} random samples (read these) --")
    for r in RNG.sample(rows, min(samples, len(rows))):
        txt = (get_text(r) or "").strip()
        exp = get_script(r) if get_script else None
        fl = flag_row(txt, get_finish(r), exp) or ["clean"]
        print(f"\n  [{r.get('model','?')} | {r.get('lang','?')} | "
              f"{r.get('script') or r.get('condition','')}] {','.join(fl)}")
        print(f"    {txt[:300]}" + ("…" if len(txt) > 300 else ""))
    if flagged:
        print(f"\n  -- {min(samples, len(flagged))} FLAGGED samples --")
        for r, fl in RNG.sample(flagged, min(samples, len(flagged))):
            txt = (get_text(r) or "").strip()
            print(f"\n  [{r.get('model','?')} | {r.get('lang','?')}] {','.join(fl)}")
            print(f"    {txt[:300]}" + ("…" if len(txt) > 300 else ""))


def load(glob_pat, base):
    rows = []
    for f in sorted((ROOT / base).glob(glob_pat)):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit_exact(samples):
    """T1/T2 — exact match. No free text, so the checks are about parsing."""
    rows = load("*__gsm8k.jsonl", "results/exact") + load("*__mmlu.jsonl", "results/exact")
    print(f"\n{'='*74}\nT1/T2 exact-match (gsm8k + mmlu)   ({len(rows)} rows)\n{'='*74}")
    if not rows:
        print("  no data"); return
    by = defaultdict(lambda: Counter())
    for r in rows:
        m = r["model"]
        by[m]["n"] += 1
        by[m]["correct"] += int(r.get("correct", 0))
        if r.get("finish_reason") == "length":
            by[m]["hit_cap"] += 1
        p = str(r.get("pred", "")).strip()
        if p in ("", "None", "none"):
            by[m]["unparsed"] += 1
    print(f"\n  {'model':24}{'n':>7}{'acc':>8}{'hit_cap%':>10}{'unparsed%':>11}")
    for m in sorted(by):
        c = by[m]
        print(f"  {m:24}{c['n']:7d}{c['correct']/c['n']:8.3f}"
              f"{100*c['hit_cap']/c['n']:9.1f}%{100*c['unparsed']/c['n']:10.1f}%")
    print("\n  hit_cap% is the number to watch: a model that runs out of tokens before")
    print("  answering scores 0 for a reason unrelated to whether it knew the answer.")


def audit_indivibe(samples, base="results/indivibe"):
    rows = load("*__chat.jsonl", base)
    report("T3/T4 open-ended generation (indivibe chat)", rows, samples,
           lambda r: r["gen"], lambda r: r.get("finish_reason"),
           lambda r: r.get("expected_script"))


def audit_t5(samples):
    rows = load("*__native.jsonl", "results/generations")
    rows = [r for r in rows if "_smoke" not in r.get("model", "")]

    def extra(rs):
        tr = sum(1 for r in rs if r.get("prompt_truncated"))
        print(f"\n  prompts truncated at context limit: {tr}/{len(rs)} "
              f"({100*tr/max(len(rs),1):.1f}%)")

    report("T5 summarization (XL-Sum, native)", rows, samples,
           lambda r: r["gen"], lambda r: r.get("finish_reason"),
           lambda r: "Latin" if r["lang"] == "en" else C.LANG_SCRIPT[r["lang"]],
           extra=extra)


def audit_judge(samples):
    p = ROOT / "results" / "judge_pairwise" / "pairwise_raw.jsonl"
    print(f"\n{'='*74}\nT4 pairwise judge verdicts\n{'='*74}")
    if not p.exists():
        print("  no data"); return
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  {len(rows)} verdicts")
    DIMS = ["fluency", "script", "usefulness", "verbosity"]
    print(f"\n  {'dimension':12}{'A%':>8}{'B%':>8}{'tie%':>8}")
    for d in DIMS:
        c = Counter(r[d] for r in rows)
        n = sum(c.values())
        print(f"  {d:12}{100*c['A']/n:7.1f}%{100*c['B']/n:7.1f}%{100*c['tie']/n:7.1f}%")
    print("\n  A degenerate judge shows up as one column near 100% on every dimension —")
    print("  that is what the guided-decoding judge did (coverage=2 for every model).")
    const = sum(1 for r in rows if len({r[d] for d in DIMS}) == 1)
    print(f"  verdicts identical across all four dimensions: "
          f"{100*const/len(rows):.1f}%  (high = judge not discriminating)")


PIPELINES = {"t1": audit_exact, "t3": audit_indivibe, "t5": audit_t5, "judge": audit_judge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=sorted(PIPELINES) + ["all"], default="all")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--indivibe-dir", default="results/indivibe",
                    help="audit a regenerated T3/T4 set, e.g. results/indivibe_v2")
    a = ap.parse_args()
    for k, fn in PIPELINES.items():
        if a.pipeline in ("all", k):
            if k == "t3":
                fn(a.samples, a.indivibe_dir)
            else:
                fn(a.samples)


if __name__ == "__main__":
    main()
