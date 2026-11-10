"""T3/T4 — open-ended multilingual generation on Sarvam's Indic Vibe Check.

indivibe is 112 English seed prompts translated into 22 languages in BOTH native and
romanised script -- perfectly parallel, which fixes the worst defect of the XL-Sum
design (a different article per language, so language difficulty and content
difficulty were confounded).

We take the CHAT domain only: the study question is multilinguality, not STEM or
coding ability. 50 seeds x 2 scripts x 5 languages = 500 generations per model.

Output feeds two things:
  T3 -- script/language compliance, per script condition
  T4 -- pairwise judge win rates (src/judge_pairwise.py)

TOKEN BUDGET (Defect 29). The original 1024-token budget truncated 17-90% of replies
and T4's win rate then correlated with truncation at r = -0.778 -- the judge was
scoring our budget rather than the models. The budget is now large enough that
truncation should be negligible, capped against the context window so a thinking
model's multiplier cannot overflow it.

ROBUSTNESS. Generations are written incrementally in chunks, so an interruption keeps
everything produced so far, and a completed model is skipped on re-run. This is meant
to survive unattended overnight execution on a shared box.

  python src/run_indivibe.py --model sarvam-30b --gpu 4 --outdir results/indivibe_v2
"""
import argparse
import json
import os
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
LANG_OF = {"Hindi": "hi", "Bengali": "bn", "Punjabi": "pa", "Tamil": "ta", "Telugu": "te"}


def load_chat():
    """indivibe chat split, restricted to our five Indic languages."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("sarvamai/indivibe", "chat/test.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    df = df[df.language.isin(LANG_OF)].reset_index(drop=True)
    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--gpu-frac", type=float, default=0.0)
    ap.add_argument("--free-margin", type=float, default=0.90,
                    help="share of CURRENTLY FREE memory to claim when auto-sizing. "
                         "Retries lower this rather than lowering a fraction of TOTAL: "
                         "a small fixed fraction of total can fall below a large model's "
                         "weights and fail with 'no memory for cache blocks'.")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--target-words", type=int, default=700,
                    help="budget in WORDS; converted per model via measured Indic "
                         "fertility so every model gets equal room (Defect 31)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default="results/indivibe")
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    OUT = ROOT / args.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{args.model}__chat.jsonl"

    spec = C.MODELS[args.model]
    rows = load_chat()
    if args.limit:
        rows = rows[: args.limit]

    # Resume: a complete file is left alone so the driver can be re-run safely.
    if dst.exists():
        have = sum(1 for l in dst.read_text(encoding="utf-8").splitlines() if l.strip())
        if have >= len(rows):
            print(f"{args.model}: already complete ({have} rows) — skipping")
            return
        print(f"{args.model}: found {have}/{len(rows)} rows, regenerating from scratch")

    # Clamp to what the model actually supports. Param2-17B-A2.4B has
    # max_position_embeddings=4096, and vLLM refuses a larger request outright rather
    # than silently truncating -- so asking for 8192 everywhere kills the run on the
    # first model instead of just giving it less headroom.
    ctx = min(args.max_model_len, C.model_ctx(args.model))
    if ctx != args.max_model_len:
        print(f"context clamped {args.max_model_len} -> {ctx} (model limit)", flush=True)

    from vllm import LLM, SamplingParams
    frac = args.gpu_frac
    if frac <= 0:
        import torch
        free, total = torch.cuda.mem_get_info()
        frac = min(0.92, args.free_margin * free / total)
        print(f"GPU {_gpu}: {free/2**30:.0f}/{total/2**30:.0f} GiB free -> util={frac:.2f}",
              flush=True)

    t0 = time.time()
    llm = LLM(model=spec["path"], max_model_len=ctx,
              gpu_memory_utilization=frac, **spec.get("vllm", {}))
    print(f"loaded {args.model} in {time.time()-t0:.0f}s", flush=True)
    tok = llm.get_tokenizer()

    prompts, keep = [], []
    for r in rows:
        # The prompt IS the item -- already in the target language and script.
        # No English instruction wrapper: adding one would change the register and
        # hand the model a cue about which language to answer in, destroying the
        # compliance measurement this feeds.
        ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                      tokenize=True, add_generation_prompt=True,
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
        prompts.append({"prompt_token_ids": ids})
        keep.append(r)

    # Budget in WORDS, not tokens (Defect 31). A fixed token budget is not a fair
    # budget: at 2048 tokens Qwen3-14B (8.22 Indic tokens/word) may say 249 words
    # while sarvam-30b (1.88) may say 1089 -- a 4.4x disparity in how much each model
    # is allowed to answer. Raising the ceiling uniformly preserves that ratio, so it
    # fixes truncation without fixing fairness. Scaling by each model's measured
    # fertility gives every model the same room to say the same thing.
    longest = max(len(p["prompt_token_ids"]) for p in prompts)
    tpw = C.indic_tok_per_word(args.model)
    want = int(args.target_words * tpw) * spec.get("gen_budget", 1)
    max_tok = max(512, min(want, ctx - longest - 64))
    print(f"budget: {args.target_words} words x {tpw:.2f} tok/word "
          f"x gen_budget {spec.get('gen_budget',1)} = {want}; longest prompt {longest}, "
          f"ctx {ctx} -> max_tokens={max_tok}", flush=True)
    if max_tok < want:
        print(f"  NOTE: context-limited — this model gets "
              f"{max_tok/tpw/spec.get('gen_budget',1):.0f} words, not {args.target_words}",
              flush=True)

    sp = SamplingParams(temperature=0.0, seed=0, max_tokens=max_tok)
    counts, n_cut, n_done = {}, 0, 0
    t1 = time.time()
    with dst.open("w", encoding="utf-8") as f:
        for i in range(0, len(prompts), args.chunk):
            outs = llm.generate(prompts[i:i + args.chunk], sp, use_tqdm=False)
            for r, o in zip(keep[i:i + args.chunk], outs):
                text = o.outputs[0].text.strip()
                lg = LANG_OF[r["language"]]
                # For a romanised prompt the expected reply is Latin; for a native
                # prompt it is that language's script. Compliance is the T3 measure.
                expected = "Latin" if r["script"] == "romanised" else C.LANG_SCRIPT[lg]
                got = C.classify_script(text, expected)
                counts[(lg, r["script"], got == expected)] = \
                    counts.get((lg, r["script"], got == expected), 0) + 1
                n_cut += o.outputs[0].finish_reason == "length"
                n_done += 1
                f.write(json.dumps({
                    "model": args.model, "family": spec["family"], "band": spec["band"],
                    "lang": lg, "script": r["script"], "category": r.get("category"),
                    "prompt": r["prompt"], "original_prompt": r.get("original_prompt"),
                    "gen": text, "gen_chars": len(text),
                    "gen_tokens": len(o.outputs[0].token_ids),
                    "finish_reason": o.outputs[0].finish_reason,
                    "max_tokens": max_tok,
                    "expected_script": expected, "got_script": got,
                    "compliant": int(got == expected),
                }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {n_done}/{len(prompts)}  truncated {100*n_cut/n_done:.1f}%  "
                  f"{time.time()-t1:.0f}s", flush=True)

    for sc in ("native", "romanised"):
        ok = sum(v for (l, s, c), v in counts.items() if s == sc and c)
        tot = sum(v for (l, s, c), v in counts.items() if s == sc)
        if tot:
            print(f"  {sc:10} script compliance {ok}/{tot} = {100*ok/tot:.1f}%")
    print(f"wrote {dst} ({n_done} rows, truncated {100*n_cut/max(n_done,1):.1f}%)")


if __name__ == "__main__":
    main()
