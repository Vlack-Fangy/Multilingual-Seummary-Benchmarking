"""Run one model over the eval set and persist raw generations.

Generation is the expensive half of this study, so nothing here scores anything —
outputs are written to disk verbatim and metrics are computed separately. A metric
bug must never cost a re-run.

  python src/generate.py --model Llama-3.1-8B-Instruct --langs hi --limit 20
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")  # GPU 4 only, per project constraint
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# FlashInfer JIT-compiles its sampler via ninja on first use. We decode greedily,
# so the sampler backend cannot affect results — take the prebuilt path and avoid
# depending on a toolchain at runtime.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# Invoking python by absolute path leaves the env's bin off PATH, so subprocess
# lookups (ninja, nvcc) miss tools that are installed.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval" / "xlsum_eval.jsonl"
OUTDIR = ROOT / "results" / "generations"


def load_items(langs, buckets, limit):
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["lang"] in langs]
    if buckets:
        rows = [r for r in rows if r["bucket"] in buckets]
    if limit:
        keep, seen = [], {}
        for r in rows:  # take `limit` per language, spread across buckets
            k = r["lang"]
            if seen.get(k, 0) < limit:
                keep.append(r)
                seen[k] = seen.get(k, 0) + 1
        rows = keep
    return rows


def render(tok, prompt, spec):
    """Apply the model's chat template ourselves rather than via llm.chat().

    vLLM accepts chat_template_kwargs but it did not reach Sarvam's template, which
    left the model reasoning in English and burning the entire token budget before
    writing a summary. Rendering here makes the thinking toggle actually apply and
    makes the exact prompt auditable.

    Returns token ids, not a string: control tokens like Sarvam's <|nothink|> only
    work if they carry their special-token id. Re-tokenizing a rendered string turns
    them into literal characters the model ignores.
    """
    msgs = [{"role": "user", "content": prompt}]
    kw = spec.get("chat_kwargs", {}).get("chat_template_kwargs", {})
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    # transformers 5.x returns a BatchEncoding here, not a bare list.
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):  # batch of one
        ids = ids[0]
    ids = [int(i) for i in ids]
    # Optional assistant prefill (see sarvam-30b in the registry). Converted via
    # the vocab so control tokens keep their special ids.
    for t in spec.get("prefill", []):
        tid = tok.convert_tokens_to_ids(t)
        if tid is None or tid == getattr(tok, "unk_token_id", None):
            raise SystemExit(f"prefill token {t!r} not in vocab")
        ids.append(int(tid))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--langs", nargs="+", default=C.LANGS)
    ap.add_argument("--buckets", nargs="+", default=None)
    ap.add_argument("--condition", default="native", choices=["native", "roman"])
    ap.add_argument("--limit", type=int, default=0, help="items per language (0 = all)")
    ap.add_argument("--max-model-len", type=int, default=0, help="0 = model native, capped at 32k")
    # GPU 4 is shared with the user's other jobs; leave headroom rather than
    # claiming the whole card. 0.80 of 143GB still fits any contestant plus KV cache.
    ap.add_argument("--gpu-frac", type=float, default=0.80)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    spec = C.MODELS[args.model]
    items = load_items(args.langs, args.buckets, args.limit)
    if not items:
        raise SystemExit("no items matched")
    tpc = C.load_tok_per_char()

    ctx = args.max_model_len or C.model_ctx(args.model)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=spec["path"], max_model_len=ctx,
              gpu_memory_utilization=args.gpu_frac, enforce_eager=False,
              **spec.get("vllm", {}))
    load_s = time.time() - t0
    print(f"loaded {args.model} in {load_s:.0f}s (ctx={ctx})")

    tok = llm.get_tokenizer()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    dst = OUTDIR / f"{args.model}__{args.condition}{tag}.jsonl"
    out_f = dst.open("w", encoding="utf-8")
    n_done = 0

    # One batch per language: max_tokens is a per-language character budget
    # converted through THIS model's tok_per_char, so every model gets the same
    # number of characters to work with rather than the same number of tokens.
    for lang in args.langs:
        sub = [r for r in items if r["lang"] == lang]
        if not sub:
            continue
        max_tok = C.max_tokens_for(args.model, lang, tpc)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tok, seed=0)

        budget = ctx - max_tok - 8  # leave room for the generation itself
        convs, kept = [], []
        for r in sub:
            article, dropped, tries = r["text"], 0, 0
            ids = render(tok, C.build_prompt(article, lang, args.condition), spec)
            # Trim the ARTICLE (never the instructions) until the prompt fits.
            # Which model/language pairs need this, and by how much, is itself the
            # fertility-versus-context-budget result — so it is recorded, not hidden.
            while len(ids) > budget and tries < 6 and len(article) > 200:
                keep = max(200, int(len(article) * budget / len(ids) * 0.95))
                dropped += len(article) - keep
                article = article[:keep]
                ids = render(tok, C.build_prompt(article, lang, args.condition), spec)
                tries += 1
            r = dict(r, prompt_tokens=len(ids), truncated=dropped > 0,
                     chars_dropped=dropped)
            convs.append({"prompt_token_ids": ids})
            kept.append(r)

        t1 = time.time()
        outs = llm.generate(convs, sp, use_tqdm=False)
        dt = time.time() - t1

        expected = C.LANG_SCRIPT[lang]
        n_trunc = sum(r["truncated"] for r in kept)
        drop_pct = (100.0 * sum(r["chars_dropped"] for r in kept)
                    / max(1, sum(r["text_chars"] for r in kept)))
        for r, o in zip(kept, outs):
            text = o.outputs[0].text.strip()
            rec = {
                "model": args.model, "family": spec["family"], "band": spec["band"],
                "condition": args.condition, "lang": lang, "bucket": r["bucket"],
                "id": r["id"], "text_chars": r["text_chars"],
                "prompt_tokens": r["prompt_tokens"], "prompt_truncated": r["truncated"],
                "chars_dropped": r["chars_dropped"], "ctx": ctx,
                "max_tokens": max_tok,
                "gen": text, "gen_chars": len(text),
                "gen_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
                "script": C.classify_script(text, expected),
                "reference": r["summary"],
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1

        hit_cap = sum(1 for o in outs if o.outputs[0].finish_reason == "length")
        scripts = {}
        for o in outs:
            s = C.classify_script(o.outputs[0].text.strip(), expected)
            scripts[s] = scripts.get(s, 0) + 1
        print(f"  {lang}: n={len(kept)} max_tok={max_tok} {dt:.0f}s "
              f"| trunc={n_trunc} ({drop_pct:.1f}% chars) hit_cap={hit_cap} | script={scripts}")

    out_f.close()
    print(f"\nwrote {dst} ({n_done} rows)")


if __name__ == "__main__":
    main()
