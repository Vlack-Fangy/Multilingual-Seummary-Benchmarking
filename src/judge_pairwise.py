"""T4 — pairwise LLM-as-judge with win rates, following Sarvam's protocol.

Replaces the pointwise 1-5 judge, whose scores compressed into 2.23-2.83 and could
not separate adjacent models.

Protocol (Sarvam 30B/105B blog):
  * pairwise A-vs-B, reported as win rates
  * four dimensions: fluency, script/language correctness, usefulness, verbosity
  * an ENGLISH reference answer scores only the *usefulness* dimension — judging
    correctness directly in a low-resource language is noisy, and the reference
    sidesteps that. Sarvam used Claude Opus 4 for these; we use gpt-oss-120b.

Two things this file does NOT do, both learned the hard way:
  * no guided/structured decoding — constraining output to JSON stopped gpt-oss
    reaching its reasoning channel and collapsed every score to a constant
  * no single ordering — every pair is judged A-then-B and B-then-A and averaged,
    because position bias is worst exactly when two models are close

  python src/judge_pairwise.py --gpu 2 --limit 250
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

_gpu = "2"
if "--gpu" in sys.argv:
    _gpu = sys.argv[sys.argv.index("--gpu") + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

import common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IV = ROOT / "results" / "indivibe"
OUT = ROOT / "results" / "judge_pairwise"
JUDGE = str(MODELS_ROOT / 'gpt-oss-120b')

REF_PROMPT = ("Answer the following question helpfully and concisely, in English. "
              "This will be used as a reference answer.\n\n{q}")

CMP_PROMPT = """You are comparing two assistant replies to the same {language} user message.

The user wrote in {register}. A good reply stays in that same language and register.

USER MESSAGE:
{prompt}

REFERENCE ANSWER (English, for content only — the replies are NOT expected to be in English):
{reference}

REPLY A:
{a}

REPLY B:
{b}

Judge on four independent dimensions. For each, answer "A", "B", or "tie".

fluency        — which is more natural, grammatical {language}?
script         — which better matches the user's language AND script/register?
                 Replying in English, or switching script, is a failure here.
usefulness     — which better covers the content of the REFERENCE ANSWER?
verbosity      — which has more appropriate length? Penalise padding and
                 penalise being too short to be useful. Do NOT prefer longer.

Think briefly, then end with JSON only:
{{"fluency": "...", "script": "...", "usefulness": "...", "verbosity": "..."}}"""

DIMS = ["fluency", "script", "usefulness", "verbosity"]


def load_gens():
    """{model: {(lang, script, prompt): row}}"""
    out = {}
    for f in sorted(IV.glob("*__chat.jsonl")):
        m = f.stem.replace("__chat", "")
        d = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                d[(r["lang"], r["script"], r["prompt"])] = r
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
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--limit", type=int, default=250, help="items per pair")
    ap.add_argument("--gpu-frac", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    gens = load_gens()
    models = sorted(gens)
    keys = sorted(set.intersection(*(set(d) for d in gens.values())))
    rng = random.Random(20260811)
    rng.shuffle(keys)
    keys = keys[: args.limit]
    print(f"{len(models)} models, {len(keys)} items, "
          f"{len(models)*(len(models)-1)//2} pairs, "
          f"{len(models)*(len(models)-1)//2*len(keys)*2} judge calls")

    from vllm import LLM, SamplingParams
    frac = args.gpu_frac
    if frac <= 0:
        import torch
        free, total = torch.cuda.mem_get_info()
        frac = min(0.80, 0.90 * free / total)
        print(f"GPU {_gpu}: {free/2**30:.0f}/{total/2**30:.0f} GiB free -> util={frac:.2f}")

    t0 = time.time()
    llm = LLM(model=JUDGE, max_model_len=args.max_model_len, gpu_memory_utilization=frac)
    print(f"loaded judge in {time.time()-t0:.0f}s")
    tok = llm.get_tokenizer()

    # --- pass 1: English reference answers (50 unique seeds, cheap) ---
    seeds = sorted({gens[models[0]][k]["original_prompt"] for k in keys
                    if gens[models[0]][k].get("original_prompt")})
    print(f"generating {len(seeds)} English reference answers")
    rp = [{"prompt_token_ids": decode_ids(tok, [{"role": "user",
           "content": REF_PROMPT.format(q=s)}], "low")} for s in seeds]
    ro = llm.generate(rp, SamplingParams(temperature=0.0, max_tokens=512, seed=0),
                      use_tqdm=False)
    refs = {}
    for s, o in zip(seeds, ro):
        t = o.outputs[0].text
        refs[s] = t.split("<|channel|>")[-1].strip()[-1500:] if t else ""

    # --- pass 2: pairwise, both orderings ---
    OUT.mkdir(parents=True, exist_ok=True)
    prompts, meta = [], []
    for ma, mb in itertools.combinations(models, 2):
        for k in keys:
            ra, rb = gens[ma][k], gens[mb][k]
            lang, script, ptxt = k
            ref = refs.get(ra.get("original_prompt", ""), "(none)")
            base = dict(language=C.LANG_NAME[lang],
                        register="romanized Latin script" if script == "romanised"
                                 else f"{C.LANG_SCRIPT[lang]} script",
                        prompt=ptxt[:1200], reference=ref[:1200])
            for flip in (0, 1):
                a, b = (ra, rb) if flip == 0 else (rb, ra)
                txt = CMP_PROMPT.format(a=a["gen"][:1800], b=b["gen"][:1800], **base)
                prompts.append({"prompt_token_ids":
                                decode_ids(tok, [{"role": "user", "content": txt}])})
                meta.append((ma, mb, lang, script, ptxt, flip))

    print(f"judging {len(prompts)} comparisons")
    dst = OUT / "pairwise_raw.jsonl"
    sp = SamplingParams(temperature=0.0, max_tokens=1400, seed=0)

    # Written in chunks rather than one shot: an 18k-call batch that only persists
    # at the end loses everything to a single interruption, and this box is shared.
    CHUNK = 2000
    n_bad = n_ok = 0
    t1 = time.time()
    with dst.open("w", encoding="utf-8") as f:
        for i in range(0, len(prompts), CHUNK):
            outs = llm.generate(prompts[i:i + CHUNK], sp, use_tqdm=True)
            for (ma, mb, lang, script, ptxt, flip), o in zip(meta[i:i + CHUNK], outs):
                v = parse_verdict(o.outputs[0].text)
                if v is None:
                    n_bad += 1
                    continue
                # flip=1 presented B first, so invert the labels back to (ma, mb).
                if flip:
                    v = {d: ("B" if x == "A" else "A" if x == "B" else "tie")
                         for d, x in v.items()}
                # Verdict keys include "script", which collides with the prompt
                # condition. The condition is written as prompt_script so that
                # **v cannot silently overwrite the native/romanised split.
                rec = {"model_a": ma, "model_b": mb, "lang": lang,
                       "prompt_script": script, "item": ptxt[:200], "flip": flip}
                rec.update(v)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            f.flush()
            print(f"  {min(i+CHUNK, len(prompts))}/{len(prompts)} "
                  f"({n_ok} verdicts, {n_bad} unparseable, {time.time()-t1:.0f}s)",
                  flush=True)
    print(f"wrote {dst}  ({n_ok} verdicts, {n_bad} unparseable)")


if __name__ == "__main__":
    main()
