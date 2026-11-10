"""T5 — pairwise judging of summaries, same protocol as T4.

Why pairwise and not the pointwise judge in src/judge.py: pointwise 1-5 scores
compressed into 2.23-2.83 across every model, which cannot separate adjacent
systems. Pairwise win rates can.

Why T5 is worth ranking when T4's ranking is not. T4's win rate correlates with
generation truncation at r = -0.778 (Defect 29): our 1024-token indivibe budget cut
off 90% of sarvam-30b's replies, so the judge was scoring our budget. Summarization
has a natural target length and the roster truncates at ~0.2%, so the same confound
does not apply here. The truncation rate is reported alongside the ranking anyway,
because that is the check that would have caught Defect 29 earlier.

Every lesson from the T4 judge is carried over:
  * no guided/structured decoding -- constraining output to JSON stopped gpt-oss
    reaching its reasoning channel and collapsed every score to a constant
  * both orderings judged and averaged, because position bias is worst when two
    systems are close
  * verdict keys are namespaced away from metadata keys. In T4 the verdict dict was
    splatted over the record and its "script" key silently overwrote the prompt
    condition, destroying the native/romanised split
  * chunked writes, so an interruption does not lose the whole run

  python src/judge_pairwise_summ.py --gpu 0 --limit 250
"""

import os
from pathlib import Path
MODELS_ROOT = Path(os.environ.get("MSB_MODELS_ROOT", "./models")).expanduser()
import argparse
import itertools
import json
import os
import random
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
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results" / "judge_pairwise_summ"
JUDGE = str(MODELS_ROOT / 'gpt-oss-120b')
ARTICLE_CAP = 2500
DIMS = ["coverage", "faithfulness", "fluency", "verbosity"]

CMP_PROMPT = """You are comparing two summaries of the same {language} news article.

ARTICLE:
{article}

REFERENCE SUMMARY (written by a professional journalist):
{reference}

SUMMARY A:
{a}

SUMMARY B:
{b}

Judge on four independent dimensions. For each, answer "A", "B", or "tie".

coverage      — which captures more of the key information in the REFERENCE?
                Judge information content only. Different wording conveying the same
                facts is equally good. Copying the article's phrasing earns no credit.
faithfulness  — which makes fewer claims unsupported by the ARTICLE?
fluency       — which is more natural, grammatical {language} in the correct script?
verbosity     — which is more appropriately concise? Penalise padding, and penalise
                being too short to be useful. Do NOT prefer longer.

Think briefly, then end with JSON only:
{{"coverage": "...", "faithfulness": "...", "fluency": "...", "verbosity": "..."}}"""


def load_articles():
    """Generations store the reference but not the source article; rejoin by lang+id."""
    out = {}
    p = ROOT / "data" / "eval" / "xlsum_eval.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["lang"], r["id"])] = r["text"]
    return out


def extract_summary(text):
    """The summary itself, with any reasoning scaffolding stripped.

    Thinking-only models cannot be told not to think, so Param2-17B-A2.4B emits
    markdown-sectioned planning ("## कार्य की समझ" = "## understanding the task")
    and then the summary as a trailing block. Stored raw, its "summary" averaged
    1309 chars against a 146-char reference and would have ranked last for emitting
    a reasoning trace rather than for summarising badly.

    Taking the last non-heading block is the same operation extract_letter performs
    for MCQ: parse the model's final answer out of its output. It is applied to
    EVERY model, and verified to be a no-op for the other seven (0.0% of rows
    changed), so it is a uniform rule rather than special handling for one model.
    """
    t = (text or "").strip()
    if not t:
        return t
    blocks = [b.strip() for b in re.split(r"\n\s*\n", t) if b.strip()]
    tail = [b for b in blocks if not b.lstrip().startswith("#")]
    return tail[-1] if tail else t


def load_gens(roster_only=True):
    """{model: {(lang, id): row}} for native-condition summaries."""
    out = {}
    for f in sorted(GEN.glob("*__native.jsonl")):
        m = f.stem.replace("__native", "")
        if "_" in m and m.split("_")[-1] in ("smoke", "ctxtest", "trunctest"):
            continue
        if roster_only and m not in C.ROSTER:
            continue
        d = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["gen"] = extract_summary(r.get("gen", ""))
                d[(r["lang"], r["id"])] = r
        out[m] = d
    return out


def decode_ids(tok, msgs, reasoning="medium"):
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  reasoning_effort=reasoning)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(i) for i in ids]


def parse_verdict(txt):
    """Last JSON object in the reply; the reasoning trace may contain braces."""
    try:
        v = json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])
    except Exception:
        return None
    out = {}
    for d in DIMS:
        s = str(v.get(d, "")).strip().lower()
        out[d] = "A" if s.startswith("a") else "B" if s.startswith("b") else "tie"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit", type=int, default=250, help="items per pair")
    ap.add_argument("--gpu-frac", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--all-models", action="store_true")
    args = ap.parse_args()

    gens = load_gens(roster_only=not args.all_models)
    models = sorted(gens)
    if len(models) < 2:
        raise SystemExit(f"need >=2 models with summaries, found {models}")
    keys = sorted(set.intersection(*(set(d) for d in gens.values())))
    rng = random.Random(20260817)
    rng.shuffle(keys)
    keys = keys[: args.limit]
    n_pairs = len(models) * (len(models) - 1) // 2
    print(f"{len(models)} models, {len(keys)} items, {n_pairs} pairs, "
          f"{n_pairs * len(keys) * 2} judge calls")
    print("models:", ", ".join(models))

    articles = load_articles()

    from vllm import LLM, SamplingParams
    frac = args.gpu_frac
    if frac <= 0:
        import torch
        free, total = torch.cuda.mem_get_info()
        frac = min(0.80, 0.90 * free / total)
        print(f"GPU {_gpu}: {free/2**30:.0f}/{total/2**30:.0f} GiB free -> util={frac:.2f}")

    t0 = time.time()
    llm = LLM(model=JUDGE, max_model_len=args.max_model_len, gpu_memory_utilization=frac)
    print(f"loaded judge in {time.time()-t0:.0f}s", flush=True)
    tok = llm.get_tokenizer()

    prompts, meta = [], []
    for ma, mb in itertools.combinations(models, 2):
        for k in keys:
            ra, rb = gens[ma][k], gens[mb][k]
            lang, item_id = k
            base = dict(language=C.LANG_NAME[lang],
                        article=articles.get(k, "")[:ARTICLE_CAP],
                        reference=ra.get("reference", ""))
            for flip in (0, 1):
                a, b = (ra, rb) if flip == 0 else (rb, ra)
                txt = CMP_PROMPT.format(a=a["gen"][:1500] or "(empty)",
                                        b=b["gen"][:1500] or "(empty)", **base)
                ids = decode_ids(tok, [{"role": "user", "content": txt}])
                prompts.append({"prompt_token_ids": ids[: args.max_model_len - 1600]})
                meta.append((ma, mb, lang, item_id, flip))

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "pairwise_summ_raw.jsonl"
    # 1400 left 11.3% of verdicts unparseable in the smoke test: these prompts carry a
    # 2500-char article that T4's did not, so the judge reasons longer and was running
    # out of budget before emitting its JSON. Unparsed verdicts are not missing at
    # random -- they concentrate on the hardest comparisons -- so this is raised rather
    # than tolerated. finish_reason is recorded to make the next such case diagnosable.
    sp = SamplingParams(temperature=0.0, max_tokens=2400, seed=0)
    CHUNK = 2000
    n_bad = n_ok = n_cut = 0
    t1 = time.time()
    print(f"judging {len(prompts)} comparisons", flush=True)
    with dst.open("w", encoding="utf-8") as f:
        for i in range(0, len(prompts), CHUNK):
            outs = llm.generate(prompts[i:i + CHUNK], sp, use_tqdm=True)
            for (ma, mb, lang, item_id, flip), o in zip(meta[i:i + CHUNK], outs):
                v = parse_verdict(o.outputs[0].text)
                if v is None:
                    n_bad += 1
                    n_cut += o.outputs[0].finish_reason == "length"
                    continue
                if flip:      # flip=1 presented B first; invert back to (ma, mb)
                    v = {d: ("B" if x == "A" else "A" if x == "B" else "tie")
                         for d, x in v.items()}
                rec = {"model_a": ma, "model_b": mb, "lang": lang,
                       "item": item_id, "flip": flip}
                rec.update(v)          # DIMS never collide with the keys above
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            f.flush()
            print(f"  {min(i+CHUNK, len(prompts))}/{len(prompts)} "
                  f"({n_ok} verdicts, {n_bad} unparseable [{n_cut} hit token cap], {time.time()-t1:.0f}s)",
                  flush=True)
    print(f"wrote {dst}  ({n_ok} verdicts, {n_bad} unparseable, {n_cut} of those hit the token cap)")


if __name__ == "__main__":
    main()
