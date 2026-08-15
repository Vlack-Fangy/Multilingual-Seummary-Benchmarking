"""Eyeball audit: are model + template + settings actually coherent?

Accuracy alone cannot distinguish a working pipeline from a broken one that
still scores. This checks the things aggregates hide:

  * raw text — does the model answer sensibly, in the right language/script?
  * lucky parses — correct items where the requested marker (#### / "Answer:")
    is ABSENT, so the score came from a fallback heuristic
  * degenerate output — repetition, near-empty replies, runaway length
  * template leakage — chat-template tags or reasoning markers in the output
  * script sanity — is a native-script prompt answered in that script?

  python src/audit_outputs.py                 # summary table for every run
  python src/audit_outputs.py --show sarvam-m # print raw samples
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import common as C

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "results" / "exact"

# Anything here in the visible output means the template or stop tokens are wrong.
LEAK = ["<think>", "</think>", "<|nothink|>", "<|channel>", "<|turn>", "<start_of_turn>",
        "<|im_start|>", "[INST]", "<|assistant|>", "<|user|>", "<|end_of_turn|>"]


def degenerate(t, n=6):
    """Same line or token repeated many times — a classic decode failure."""
    if not t.strip():
        return True
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if lines and Counter(lines).most_common(1)[0][1] >= n:
        return True
    words = t.split()
    return len(words) > 40 and Counter(words).most_common(1)[0][1] > len(words) * 0.4


def audit_file(p):
    # Tolerate a half-written final line: these files are read while runs append.
    rows = []
    for l in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    rows = [r for r in rows if "gen" in r]
    if not rows:
        return None
    task = rows[0]["task"]
    marker = r"####" if task == "gsm8k" else r"(?i)answer\s*[:\-]"
    out = []
    for key in sorted({(r["lang"], r["script"]) for r in rows}):
        sub = [r for r in rows if (r["lang"], r["script"]) == key]
        correct = [r for r in sub if r["correct"]]
        lucky = [r for r in correct if not re.search(marker, r["gen"])]
        leaks = [r for r in sub if any(x in r["gen"] for x in LEAK)]
        degen = [r for r in sub if degenerate(r["gen"])]
        empty = [r for r in sub if not r["gen"].strip()]
        # For native-script prompts the reply should be in that script (or Latin
        # for maths working, which is legitimate) — flag only 'mixed'/'empty'.
        exp = C.LANG_SCRIPT.get(key[0], "Latin")
        odd = 0
        if key[1] == "native" and key[0] != "en":
            odd = sum(1 for r in sub if C.classify_script(r["gen"], exp) == "empty")
        out.append(dict(model=rows[0]["model"], task=task, lang=key[0], script=key[1],
                        n=len(sub), lucky=len(lucky), leak=len(leaks),
                        degen=len(degen), empty=len(empty), noscript=odd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="print raw samples for this model")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    if args.show:
        for p in sorted(EX.glob(f"{args.show}__*.jsonl")):
            # Tolerate a half-written final line: these files are read while runs append.
    rows = []
    for l in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            continue
            rows = [r for r in rows if "gen" in r]
            for key in sorted({(r["lang"], r["script"]) for r in rows}):
                sub = [r for r in rows if (r["lang"], r["script"]) == key][: args.n]
                for r in sub:
                    print(f"\n--- {r['model']} {r['task']} {r['lang']}/{r['script']} "
                          f"gold={r['gold']} pred={r['pred']} {'OK' if r['correct'] else 'WRONG'}")
                    print(r["gen"][:400].replace("\n", " ⏎ "))
        return

    allr = []
    for p in sorted(EX.glob("*__*.jsonl")):
        r = audit_file(p)
        if r:
            allr.extend(r)
    if not allr:
        print("No runs carry raw text yet — only runs made after the audit patch do.")
        return
    print(f"{'model':22} {'task':9} {'l/s':10} {'n':>5} {'lucky':>6} {'leak':>5} "
          f"{'degen':>6} {'empty':>6} {'noscript':>9}")
    bad = 0
    for r in allr:
        flag = ""
        if r["leak"] or r["degen"] or r["empty"] or r["lucky"] > 0.15 * r["n"]:
            flag = "  <-- CHECK"
            bad += 1
        print(f"{r['model']:22} {r['task']:9} {r['lang']+'/'+r['script']:10} {r['n']:5} "
              f"{r['lucky']:6} {r['leak']:5} {r['degen']:6} {r['empty']:6} {r['noscript']:9}{flag}")
    print(f"\n{bad} cell(s) flagged for inspection out of {len(allr)}")


if __name__ == "__main__":
    main()
