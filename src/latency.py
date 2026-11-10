"""T6 — decode latency and throughput, and what a word costs in each language.

IMPORTANT CAVEAT. This box is shared and no GPU was exclusively free when this ran.
Absolute milliseconds are therefore NOT publishable as this hardware's capability --
a neighbour at 100% utilisation inflates them by an unknown factor. What survives is
the RATIO between models, because they are measured back-to-back on the same card
under comparable contention. Treat every absolute number here as indicative and every
ratio as the result.

Method:
  * warm up first -- the first request pays CUDA-graph capture and Triton JIT costs
    that are not representative of steady state
  * batch 1 for interactive latency (what one user feels)
  * batch 16 for serving throughput (what a deployment budgets)
  * fixed max_tokens with ignore_eos, so every model generates exactly the same
    number of tokens and the comparison is not confounded by verbosity

The second half is the part that matters for an Indic product: tokens/sec is
meaningless until multiplied by how many tokens the language actually costs. A model
twice as fast that needs three times the tokens per Hindi word is slower in practice.

  python src/latency.py --gpu 4 --models gemma-4-12B-it gemma-4-26B-A4B-it gemma-4-31B-it
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_gpu = "4"
if "--gpu" in sys.argv:
    _gpu = sys.argv[sys.argv.index("--gpu") + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "latency"
PROMPT = ("You are a supportive assistant. A user has written to you about their week. "
          "Reply thoughtfully and at length.")


def fertility():
    """{model: {lang: tokens per word}} from the T6 measurement."""
    out = {}
    p = ROOT / "results" / "fertility" / "fertility.csv"
    with p.open() as f:
        for r in csv.DictReader(f):
            out.setdefault(r["model"], {})[r["lang"]] = float(r["tok_per_word"])
    return out


def bench(llm, tok, n_tokens, batch, reps=3):
    """Return (ttft_ms, decode_tok_per_s) — median of `reps` timed runs."""
    from vllm import SamplingParams
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    ids = [int(i) for i in ids]
    prompts = [{"prompt_token_ids": ids} for _ in range(batch)]

    # ignore_eos so every model emits exactly n_tokens: otherwise a terse model
    # "wins" by stopping early, which measures verbosity, not speed.
    sp_full = SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True, seed=0)
    sp_one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, seed=0)

    llm.generate(prompts, sp_full, use_tqdm=False)          # warm-up, discarded

    ttfts, rates = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate(prompts, sp_one, use_tqdm=False)
        ttfts.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        llm.generate(prompts, sp_full, use_tqdm=False)
        dt = time.perf_counter() - t0
        rates.append(batch * n_tokens / dt)
    ttfts.sort(); rates.sort()
    return ttfts[len(ttfts)//2], rates[len(rates)//2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--gpu-frac", type=float, default=0.0)
    args = ap.parse_args()

    fert = fertility()
    from vllm import LLM
    rows = []
    for m in args.models:
        spec = C.MODELS[m]
        frac = args.gpu_frac
        if frac <= 0:
            import torch
            free, total = torch.cuda.mem_get_info()
            frac = min(0.80, 0.90 * free / total)
        t0 = time.time()
        llm = LLM(model=spec["path"], max_model_len=4096,
                  gpu_memory_utilization=frac, **spec.get("vllm", {}))
        load_s = time.time() - t0
        tok = llm.get_tokenizer()
        ttft1, rate1 = bench(llm, tok, args.tokens, batch=1)
        _, rate16 = bench(llm, tok, args.tokens, batch=16)
        rows.append(dict(model=m, load_s=round(load_s, 1),
                         ttft_ms=round(ttft1, 1),
                         tok_s_b1=round(rate1, 1), tok_s_b16=round(rate16, 1),
                         s_per_256=round(args.tokens / rate1, 2)))
        print(f"{m:24} TTFT {ttft1:7.1f} ms | b1 {rate1:7.1f} tok/s | "
              f"b16 {rate16:8.1f} tok/s", flush=True)
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        time.sleep(5)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(rows, indent=2))

    # ---- what a word actually costs, per language ----
    print(f"\nSeconds to generate 100 words, batch 1 "
          f"(tokens/word x 100 / decode rate)\n")
    langs = ["en", "hi", "bn", "ta", "te"]
    print(f"{'model':24}" + "".join(f"{l:>9}" for l in langs) + f"{'hi/en':>9}{'ta/en':>9}")
    for r in rows:
        f = fert.get(r["model"], {})
        if not f:
            continue
        secs = {l: 100 * f[l] / r["tok_s_b1"] for l in langs if l in f}
        print(f"{r['model']:24}" + "".join(f"{secs[l]:8.2f}s" for l in langs if l in secs)
              + f"{secs['hi']/secs['en']:8.2f}x{secs['ta']/secs['en']:8.2f}x")
    (OUT / "latency_words.json").write_text(json.dumps(
        {r["model"]: {l: 100 * fert[r["model"]][l] / r["tok_s_b1"]
                      for l in langs if l in fert.get(r["model"], {})} for r in rows}, indent=2))
    print(f"\nwrote {OUT}/latency.json and latency_words.json")


if __name__ == "__main__":
    main()
