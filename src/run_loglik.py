"""T1/T2 by LOG-LIKELIHOOD instead of generation — the fix for Defects 26 and 27.

Generative MCQ scoring made us vulnerable to two failure modes that between them
voided every absolute figure for a high-truncation model:

  Defect 26  the model runs out of tokens before answering, and the parser scores
             the unfinished text at chance
  Defect 27  stored output was clipped below the point the parser read, so the
             failure could not even be diagnosed offline

Scoring the options by likelihood removes the entire class: nothing is generated, so
nothing truncates, nothing needs parsing, and no abstention policy is required.

METHOD — chat template by default, cloze available for comparison.

A first version used only the lm-evaluation-harness cloze convention. Measured on
gemma-4-12B-it over 12 items, the candidate letter appeared in the top-20 next tokens:

    bare "Answer:"        10/12   top tokens: \n, " :", " \"
    trailing space         2/12   top tokens: \n, "1", "0"
    explicit phrasing      9/12   top tokens: \n, " The", \n\n
    chat template         12/12   top tokens: "B", "C", "D"

In completion mode Gemma emits newlines and punctuation; under its chat template it
emits the answer letter directly. The cloze convention is therefore NOT portable: it
suited sarvam-m and Mistral-Small (shared lineage and tokenizer, no-candidate = 0) and
failed on gemma-4-31B-it, which returned 23-34% no-candidate and BELOW-CHANCE accuracy.
Those results were discarded rather than reported.

Default is now --mode chat. --mode cloze reproduces the harness convention for
comparison; the difference between them measures how much the completion format
penalises a conversation-tuned model.

The cloze format, for reference:

    <question>
    A. <opt>   B. <opt>   C. <opt>   D. <opt>
    Answer:

and we read the model's probability for the next token being " A" / " B" / ... The
argmax is the answer. This is what the published MMLU numbers are computed with, so
reproducing Sarvam's MMLU-IN ~0.79 is a meaningful check; a bespoke format would make
the comparison meaningless.

Two consequences worth stating:
  * No chat template. The cloze format is a completion task, and it is what the
    convention scores. Instruct models still handle it -- "Answer:" is a strong cue.
  * No length normalisation needed. Every continuation is a single letter token, so
    the unnormalised / per-token / per-byte ambiguity that plagues option-text
    scoring simply does not arise here.

This measures something subtly DIFFERENT from generative scoring: instruction tuning
shifts probability mass toward answering after "Answer:", and chain-of-thought lets a
model think before committing. The two can disagree, and the disagreement is itself a
result -- it is how much format-following costs each model. Report both, never one
instead of the other.

  python src/run_loglik.py --task mmlu --model sarvam-m --gpu 0
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
from run_exact import LETTERS, TASKS, load_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "loglik"
TOPK = 40      # candidate letters sit near the top after "Answer:"; 40 is ample headroom


def cloze(row, task):
    """(context, gold_letter, n_options) in the harness's completion format."""
    ch = row["choices"]
    try:
        labels, texts = list(ch["label"]), list(ch["text"])
    except (KeyError, IndexError, TypeError, ValueError):
        labels, texts = None, list(ch)
    opts = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))
    if "answerKey" in row:
        key = str(row["answerKey"]).strip()
        gold = key if key in LETTERS else (
            LETTERS[labels.index(key)] if labels and key in labels else None)
    else:
        gold = LETTERS[int(row["answer"])]
    return f"{row['question']}\n{opts}\nAnswer:", gold, len(texts)


def letter_ids(tok, n_opts):
    """{letter: [candidate token ids]}.

    Both " A" and "A" are collected: whether the leading space is part of the token
    depends on the tokenizer, and guessing wrong silently scores every option at
    -inf, which would look like a broken model rather than a broken harness.
    """
    out = {}
    for L in LETTERS[:n_opts]:
        ids = set()
        for form in (f" {L}", L):
            e = tok.encode(form, add_special_tokens=False)
            if e:
                ids.add(e[-1] if len(e) == 1 else e[0])
        out[L] = sorted(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--task", default="mmlu", choices=["mmlu", "arc", "triviaqa"])
    ap.add_argument("--mode", default="chat", choices=["chat", "cloze"],
                    help="chat: apply the model's chat template and score the first "
                         "reply token (portable across families). cloze: the bare "
                         "lm-eval-harness completion format (not portable — see module docstring).")
    ap.add_argument("--langs", nargs="+", default=C.LANGS)
    ap.add_argument("--scripts", nargs="+", default=["native", "roman"])
    ap.add_argument("--limit", type=int, default=500, help="items per lang/script")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--gpu-frac", type=float, default=0.0)
    ap.add_argument("--free-margin", type=float, default=0.90,
                    help="share of CURRENTLY FREE memory to claim when auto-sizing. "
                         "Retries lower this rather than lowering a fraction of TOTAL: "
                         "a small fixed fraction of total can fall below a large model's "
                         "weights and fail with 'no memory for cache blocks'.")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    spec = C.MODELS[args.model]
    from vllm import LLM, SamplingParams
    frac = args.gpu_frac
    if frac <= 0:
        import torch
        free, total = torch.cuda.mem_get_info()
        frac = min(0.92, args.free_margin * free / total)
        print(f"GPU {_gpu}: {free/2**30:.0f}/{total/2**30:.0f} GiB free -> util={frac:.2f}")

    t0 = time.time()
    # max_logprobs must be raised at construction: vLLM caps sampled logprobs at 20 by
    # default and rejects the request otherwise. Shrinking K instead would risk the
    # candidate letters falling outside the returned distribution, which registers as
    # "no candidate" and would look like a model failure rather than a harness limit.
    llm = LLM(model=spec["path"], max_model_len=args.max_model_len,
              gpu_memory_utilization=frac, max_logprobs=TOPK,
              **spec.get("vllm", {}))
    print(f"loaded {args.model} in {time.time()-t0:.0f}s", flush=True)
    tok = llm.get_tokenizer()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if args.mode == "chat" else "__cloze"
    dst = OUT / f"{args.model}__{args.task}{tag}.jsonl"
    summary = []
    with dst.open("w", encoding="utf-8") as f:
        for lang in args.langs:
            for script in args.scripts:
                if script == "roman" and not TASKS[args.task]["roman"]:
                    continue
                if script == "roman" and lang == "en":
                    continue
                try:
                    rows = load_split(args.task, lang, script)
                except Exception as e:
                    print(f"  {lang}/{script}: SKIP ({type(e).__name__})")
                    continue
                rows = rows[: args.limit]
                built = [cloze(r, args.task) for r in rows]
                if args.mode == "cloze":
                    prompts = [b[0] for b in built]
                else:
                    prompts = []
                    for ctx, _, _ in built:
                        body = ctx.rsplit("\nAnswer:", 1)[0]
                        msgs = [{"role": "user", "content": body +
                                 "\n\nReply with only the letter of the correct option."}]
                        ids = tok.apply_chat_template(
                            msgs, tokenize=True, add_generation_prompt=True,
                            **spec.get("chat_kwargs", {}).get("chat_template_kwargs", {}))
                        if hasattr(ids, "input_ids"):
                            ids = ids.input_ids
                        elif isinstance(ids, dict):
                            ids = ids["input_ids"]
                        if ids and isinstance(ids[0], (list, tuple)):
                            ids = ids[0]
                        ids = [int(i) for i in ids]
                        # NOTE: the model's prefill is deliberately NOT applied here.
                        # Tried for sarvam-30b (spec: ["<think>", "</think>"]) and it
                        # made things worse -- no-candidate went 33.8% -> ~95%. Closing
                        # the thinking block simply moves the model from reasoning to
                        # prose ("The answer is..."), and first-token scoring needs the
                        # very first token to BE the letter.
                        prompts.append({"prompt_token_ids": ids})

                # One forward pass per item. max_tokens=1 with logprobs gives the full
                # next-token distribution; we never sample, so nothing can truncate.
                sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=TOPK, seed=0)
                t1 = time.time()
                outs = llm.generate(prompts, sp, use_tqdm=False)
                dt = time.time() - t1

                n_ok = n_missing = 0
                for (ctx, gold, n_opts), o in zip(built, outs):
                    lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
                    cand = letter_ids(tok, n_opts)
                    scores = {}
                    for L, ids in cand.items():
                        vals = [lp[i].logprob for i in ids if i in lp]
                        if vals:
                            scores[L] = max(vals)
                    if not scores:
                        # No candidate letter in the top-K: record it rather than
                        # silently guessing. A high rate here means the format is wrong
                        # for this model, not that the model is bad.
                        n_missing += 1
                        pred = None
                    else:
                        pred = max(scores, key=scores.get)
                    ok = pred == gold
                    n_ok += ok
                    f.write(json.dumps({
                        "model": args.model, "task": args.task,
                        "method": f"loglik-{args.mode}",
                        "lang": lang, "script": script, "gold": gold, "pred": pred,
                        "correct": int(ok), "n_opts": n_opts,
                        "scores": {k: round(v, 4) for k, v in scores.items()},
                    }, ensure_ascii=False) + "\n")
                acc = n_ok / max(len(built), 1)
                summary.append(dict(lang=lang, script=script, n=len(built),
                                    acc=round(acc, 4), missing=n_missing,
                                    sec=round(dt, 1)))
                print(f"  {lang}/{script:7} n={len(built):4d} acc={acc:.4f} "
                      f"no-candidate={n_missing:3d}  {dt:5.1f}s", flush=True)

    (OUT / f"{args.model}__{args.task}{tag}__summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
