"""T1/T2 — exact-match evaluation on Sarvam's indic-evals, native vs romanized.

This is the headline measurement and it needs no judge: identical items differing
only in script, scored by exact match. No chrF, no extractiveness confound, no
metric dispute.

  T1 script gap      = acc(native) - acc(roman)     [mmlu, gsm8k: have _roman]
  T2 language penalty = acc(en)    - acc(indic)      [all four datasets]

Absolute scores are NOT the result — the deltas are. "Model X gets 0.92 on
GSM8K-IN" is a math claim; "X loses 10 points to GSM8K-IN-R while Y loses 25" is
a multilinguality claim, and is clean because the task is held constant.

T6 latency (TTFT / end-to-end) is captured here too, from vLLM's own per-request
metrics rather than wall-clock around the batch, so queueing is not billed as
compute.

  python src/run_exact.py --task gsm8k --model sarvam-30b --langs hi --scripts native roman
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_gpu = "0"
if "--gpu" in sys.argv:
    _gpu = sys.argv[sys.argv.index("--gpu") + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exact"

TASKS = {
    "mmlu":     dict(repo="sarvamai/mmlu-indic",           kind="mcq",  roman=True),
    "gsm8k":    dict(repo="sarvamai/gsm8k-indic",          kind="num",  roman=True),
    "arc":      dict(repo="sarvamai/arc-challenge-indic",  kind="mcq",  roman=False),
    "triviaqa": dict(repo="sarvamai/trivia-qa-indic-mcq",  kind="mcq",  roman=False),
}

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# English instructions in every cell, so the input script stays the only variable.
MCQ_PROMPT = ("Answer the following multiple-choice question.\n\n"
              "{question}\n\n{options}\n\n"
              "Think briefly if needed, then end your reply with the answer on its own "
              "line as: Answer: <letter>")

NUM_PROMPT = ("Solve this problem. Think briefly, then give the final numeric answer.\n\n"
              "{question}\n\n"
              "End your reply with the answer on its own line as: #### <number>")


def load_split(task, lang, script):
    """One (language, script) split as a list of dicts."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    cfg = f"{lang}_roman" if script == "roman" else lang
    p = hf_hub_download(TASKS[task]["repo"], f"{cfg}/test-00000-of-00001.parquet",
                        repo_type="dataset")
    return pd.read_parquet(p).to_dict("records")


def build(task, row):
    """(prompt, gold) for one item."""
    kind = TASKS[task]["kind"]
    if kind == "mcq":
        choices = list(row["choices"])
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        gold = LETTERS[int(row["answer"])]
        return MCQ_PROMPT.format(question=row["question"], options=opts), gold, len(choices)
    # gsm8k gold lives in the English CoT as "#### <number>"
    m = re.search(r"####\s*([\-0-9.,]+)", row["answer"])
    gold = m.group(1).replace(",", "").strip(".") if m else None
    return NUM_PROMPT.format(question=row["question"]), gold, 0


def extract_letter(text, n_opts):
    """Answer letter: prefer the requested 'Answer: X' marker, else last bare letter.

    Parse failures are recorded as pred=None rather than silently scored wrong, so
    format-following and knowledge stay separable.
    """
    valid = set(LETTERS[:n_opts or 4])
    m = re.findall(r"(?:answer|उत्तर)\s*[:\-]?\s*\(?([A-Ha-h])\)?", text, re.IGNORECASE)
    if m and m[-1].upper() in valid:
        return m[-1].upper()
    m = re.findall(r"\b([A-H])\b", text)
    for c in reversed(m):
        if c in valid:
            return c
    return None


def extract_num(text):
    """Final numeric answer: prefer the requested #### marker, else last number."""
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--langs", nargs="+", default=["hi", "bn", "pa", "ta", "te", "en"])
    ap.add_argument("--scripts", nargs="+", default=["native", "roman"])
    ap.add_argument("--limit", type=int, default=500, help="items per lang/script (0=all)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--gpu-frac", type=float, default=0.0, help="0 = auto-size to free memory")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    spec = C.MODELS[args.model]
    kind = TASKS[args.task]["kind"]

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    frac = args.gpu_frac
    if frac <= 0:
        import torch
        free, total = torch.cuda.mem_get_info()
        frac = min(0.80, 0.90 * free / total)
        print(f"GPU {_gpu}: {free/2**30:.0f}/{total/2**30:.0f} GiB free -> util={frac:.2f}")

    t0 = time.time()
    llm = LLM(model=spec["path"], max_model_len=args.max_model_len,
              gpu_memory_utilization=frac, **spec.get("vllm", {}))
    print(f"loaded {args.model} in {time.time()-t0:.0f}s")
    tok = llm.get_tokenizer()

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{args.model}__{args.task}.jsonl"
    f_out = dst.open("w", encoding="utf-8")
    summary = []

    for lang in args.langs:
        for script in args.scripts:
            # English has no romanized variant, and some datasets have none at all.
            if script == "roman" and (lang == "en" or not TASKS[args.task]["roman"]):
                continue
            try:
                rows = load_split(args.task, lang, script)
            except Exception as e:
                print(f"  {lang}/{script}: SKIP ({type(e).__name__})")
                continue
            if args.limit:
                rows = rows[: args.limit]

            prompts, golds, nopts = [], [], []
            for r in rows:
                p, g, n = build(args.task, r)
                if g is None:
                    continue
                ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                              add_generation_prompt=True,
                                              **spec.get("chat_kwargs", {}).get("chat_template_kwargs", {}))
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                elif isinstance(ids, dict):
                    ids = ids["input_ids"]
                if ids and isinstance(ids[0], (list, tuple)):
                    ids = ids[0]
                ids = [int(i) for i in ids]
                for t in spec.get("prefill", []):
                    ids.append(int(tok.convert_tokens_to_ids(t)))
                ids = ids[: args.max_model_len - 600]
                prompts.append({"prompt_token_ids": ids})
                golds.append(g)
                nopts.append(n)

            # NOT guided decoding. Constraining the first token to a bare letter
            # collapsed the model onto "A" 94% of the time against balanced gold --
            # it forces a choice out of a distribution the model never intended to
            # place there. Free generation + parse is also what Sarvam did.
            sp = SamplingParams(temperature=0.0, seed=0,
                                max_tokens=256 if kind == "mcq" else 512)

            t1 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            wall = time.time() - t1

            n_ok, n_unparsed, ttfts, e2es = 0, 0, [], []
            for r, o, g, nop in zip(rows, outs, golds, nopts):
                text = o.outputs[0].text.strip()
                pred = extract_letter(text, nop) if kind == "mcq" else extract_num(text)
                ok = (pred == g) if kind == "mcq" else num_eq(pred, g)
                n_ok += ok
                if pred is None:
                    n_unparsed += 1
                m = getattr(o, "metrics", None)
                if m is not None:
                    ft = getattr(m, "first_token_time", None)
                    at = getattr(m, "arrival_time", None)
                    fi = getattr(m, "finished_time", None)
                    if ft and at:
                        ttfts.append(ft - at)
                    if fi and at:
                        e2es.append(fi - at)
                f_out.write(json.dumps({
                    "model": args.model, "family": spec["family"], "band": spec["band"],
                    "task": args.task, "lang": lang, "script": script,
                    "gold": g, "pred": pred, "correct": int(ok),
                    "gen_tokens": len(o.outputs[0].token_ids),
                    "finish_reason": o.outputs[0].finish_reason,
                }, ensure_ascii=False) + "\n")

            acc = n_ok / max(1, len(golds))
            pct = lambda v, q: (sorted(v)[int(q * (len(v) - 1))] if v else None)
            rec = {"model": args.model, "task": args.task, "lang": lang, "script": script,
                   "n": len(golds), "acc": round(acc, 4), "wall_s": round(wall, 1),
                   "ttft_med": pct(ttfts, .5), "e2e_med": pct(e2es, .5), "e2e_p95": pct(e2es, .95),
                   "unparsed": n_unparsed,
                   "gpu_s_per_1k": round(wall / max(1, len(golds)) * 1000, 1)}
            summary.append(rec)
            print(f"  {lang:3}/{script:6} n={len(golds):5} acc={acc:.4f} "
                  f"unparsed={n_unparsed:4} {wall:6.1f}s  gpu_s/1k={rec['gpu_s_per_1k']:7.1f}")

    f_out.close()
    (OUT / f"{args.model}__{args.task}__summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
