"""LLM-judge scoring of persisted generations with gpt-oss-120b.

Why this exists: chrF rewards literal string overlap. On a worked example the
model that copied a phrase from the article outscored the model that paraphrased
and covered *both* topics of the gold summary by 10 points. So "fertility does
not predict chrF" cannot become "fertility does not predict quality" until a
semantic measurement agrees.

Design:
  * REFERENCE-BASED — the judge sees article + gold + candidate. Comparing a
    candidate against a gold is a far lower bar for a non-Indic-specialised judge
    than independently assessing Tamil fluency, which is where METAL's warning
    about weak multilingual judges actually bites.
  * gpt-oss-120b — no OpenAI model is in the contestant set, so it is the only
    strong local judge with zero self-preference bias. Its Indic fertility (2.59)
    also beats every contestant but Sarvam, so article+gold+candidate fits.
  * NO guided decoding. Constraining output to the JSON schema forces gpt-oss to
    emit JSON from its first token, so it never reaches its reasoning channel.
    Measured on a case with known ground truth, the constrained judge collapsed
    to coverage=2 for every model and scored one good summary 1/1/1; unconstrained
    it separated them correctly (4/3/2/2) and matched the qualitative read. We let
    it reason and parse the final JSON object out of the tail.

  python src/judge.py --condition native
  python src/judge.py --condition native --judge gpt-oss-20b --sample 0.05
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Device must be pinned before torch initialises, so --gpu is read straight from
# argv rather than via argparse. This box is shared: GPU 4 picked up another
# user's 38GB vLLM job mid-project, so the device is a runtime choice.
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
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results" / "judge"

JUDGES = {
    "gpt-oss-120b": "/home/models/gpt-oss-120b",
    "gpt-oss-20b": "/home/models/gpt-oss-20b",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "coverage": {"type": "integer", "minimum": 1, "maximum": 5},
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "fluency": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["coverage", "faithfulness", "fluency"],
    "additionalProperties": False,
}

# Article is capped so one long source cannot dominate the judge's context; the
# lead of a news article carries the summarisable content.
ARTICLE_CAP = 3000

PROMPT = """You are evaluating a summary of a {language} news article.

ARTICLE:
{article}

REFERENCE SUMMARY (written by a professional journalist):
{reference}

CANDIDATE SUMMARY (to be evaluated):
{candidate}

Rate the CANDIDATE on three independent 1-5 scales.

coverage — does it capture the same key information as the REFERENCE?
  5 = every main point of the reference; 3 = about half; 1 = unrelated content.
  Judge information content only. Different wording that conveys the same facts
  scores 5. Copying the article's phrasing is NOT itself worth credit.

faithfulness — is every claim supported by the ARTICLE?
  5 = fully supported; 3 = one unsupported detail; 1 = substantially fabricated.

fluency — is it well-formed, natural {language} in the correct script?
  5 = natural and grammatical; 3 = understandable but awkward; 1 = broken,
  wrong language, or degenerate repetition.

Reply with JSON only: {{"coverage": n, "faithfulness": n, "fluency": n}}"""


def build(rec):
    return PROMPT.format(
        language=C.LANG_NAME[rec["lang"]],
        article=rec.get("article", "")[:ARTICLE_CAP],
        reference=rec["reference"],
        candidate=rec["gen"] if rec["gen"].strip() else "(empty)",
    )


def load_articles():
    """Generations store the reference but not the source article; rejoin by id+lang."""
    out = {}
    p = ROOT / "data" / "eval" / "xlsum_eval.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["lang"], r["id"])] = r["text"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="native")
    ap.add_argument("--judge", default="gpt-oss-120b", choices=sorted(JUDGES))
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of rows to judge")
    ap.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--gpu", default="0", help="CUDA device (read early from argv)")
    ap.add_argument("--gpu-frac", type=float, default=0.0,
                    help="0 = auto-size to free memory (shared GPU safe)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    articles = load_articles()
    rows = []
    for f in sorted(GEN.glob(f"*__{args.condition}.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["article"] = articles.get((r["lang"], r["id"]), "")
                rows.append(r)
    if args.sample < 1.0:
        rng = random.Random(20260811)
        rows = [r for r in rows if rng.random() < args.sample]
    print(f"judging {len(rows)} rows with {args.judge} (reasoning={args.reasoning})")

    from vllm import LLM, SamplingParams

    frac = args.gpu_frac
    if args.gpu_frac <= 0:
        # GPU 4 is shared with other users' jobs. vLLM measures utilization against
        # TOTAL memory, so a fixed fraction fails whenever a neighbour is resident.
        # Claim 90% of what is actually free instead, and never more than asked.
        import torch
        free, total = torch.cuda.mem_get_info()
        frac = min(0.80, 0.90 * free / total)
        print(f"GPU: {free/2**30:.0f}GiB free of {total/2**30:.0f}GiB -> "
              f"gpu_memory_utilization={frac:.2f}")

    t0 = time.time()
    llm = LLM(model=JUDGES[args.judge], max_model_len=args.max_model_len,
              gpu_memory_utilization=frac)
    print(f"loaded judge in {time.time()-t0:.0f}s")
    tok = llm.get_tokenizer()

    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": build(r)}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      reasoning_effort=args.reasoning)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        ids = [int(i) for i in ids]
        # Truncate from the LEFT of the article only if a prompt still overruns.
        prompts.append({"prompt_token_ids": ids[: args.max_model_len - 600]})

    # Room for the reasoning trace plus the JSON verdict.
    sp = SamplingParams(temperature=0.0, max_tokens=1600, seed=0)

    t1 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=True)
    print(f"judged in {time.time()-t1:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{args.judge}__{args.condition}.jsonl"
    n_bad = 0
    with dst.open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            txt = o.outputs[0].text.strip()
            try:
                # LAST json object: the reasoning trace may contain braces of its own.
                sc = json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])
                cov, fai, flu = int(sc["coverage"]), int(sc["faithfulness"]), int(sc["fluency"])
            except Exception:
                n_bad += 1
                cov = fai = flu = None
            f.write(json.dumps({
                "judge": args.judge, "model": r["model"], "family": r["family"],
                "band": r["band"], "lang": r["lang"], "bucket": r["bucket"], "id": r["id"],
                "coverage": cov, "faithfulness": fai, "fluency": flu,
                "gen_chars": r["gen_chars"], "script": r["script"],
                "finish_reason": o.outputs[0].finish_reason,
            }, ensure_ascii=False) + "\n")
    print(f"wrote {dst} ({len(rows)} rows, {n_bad} unparseable)")


if __name__ == "__main__":
    main()
