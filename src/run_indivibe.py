"""T4 — open-ended multilingual generation on Sarvam's Indic Vibe Check.

indivibe is 112 English seed prompts translated into 22 languages in BOTH native
and romanised script — perfectly parallel, which fixes the worst defect of our
XL-Sum design (different article per language, so language difficulty and content
difficulty were confounded).

We take the CHAT domain only: the study question is multilinguality, not STEM or
coding ability. 50 seeds x 2 scripts x 5 languages = 500 generations per model.

Output feeds two things:
  T3 — script/language compliance, per script condition
  T4 — pairwise judge win rates (src/judge_pairwise.py)

  python src/run_indivibe.py --model sarvam-30b --gpu 2
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
OUT = ROOT / "results" / "indivibe"

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
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = C.MODELS[args.model]
    rows = load_chat()
    if args.limit:
        rows = rows[: args.limit]

    from vllm import LLM, SamplingParams
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

    prompts, keep = [], []
    for r in rows:
        # The prompt IS the item — already in the target language and script.
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
        prompts.append({"prompt_token_ids": ids[: args.max_model_len - 1024]})
        keep.append(r)

    sp = SamplingParams(temperature=0.0, seed=0,
                        max_tokens=1024 * spec.get("gen_budget", 1))
    t1 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"generated {len(outs)} in {time.time()-t1:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{args.model}__chat.jsonl"
    counts = {}
    with dst.open("w", encoding="utf-8") as f:
        for r, o in zip(keep, outs):
            text = o.outputs[0].text.strip()
            lg = LANG_OF[r["language"]]
            # For a romanised prompt the expected reply is Latin; for a native
            # prompt it is that language's script. Compliance is the T3 measure.
            expected = "Latin" if r["script"] == "romanised" else C.LANG_SCRIPT[lg]
            got = C.classify_script(text, expected)
            counts[(lg, r["script"], got == expected)] = \
                counts.get((lg, r["script"], got == expected), 0) + 1
            f.write(json.dumps({
                "model": args.model, "family": spec["family"], "band": spec["band"],
                "lang": lg, "script": r["script"], "category": r.get("category"),
                "prompt": r["prompt"], "original_prompt": r.get("original_prompt"),
                "gen": text, "gen_chars": len(text),
                "gen_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
                "expected_script": expected, "got_script": got,
                "compliant": int(got == expected),
            }, ensure_ascii=False) + "\n")

    for sc in ("native", "romanised"):
        ok = sum(v for (l, s, c), v in counts.items() if s == sc and c)
        tot = sum(v for (l, s, c), v in counts.items() if s == sc)
        if tot:
            print(f"  {sc:10} script compliance {ok}/{tot} = {100*ok/tot:.1f}%")
    print(f"wrote {dst} ({len(keep)} rows)")


if __name__ == "__main__":
    main()
