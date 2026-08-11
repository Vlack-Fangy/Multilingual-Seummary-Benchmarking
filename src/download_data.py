"""Fetch eval corpora into data/raw.

XL-Sum still ships a loading script and its HF parquet auto-conversion is broken
("Cannot get the config names"), so load_dataset() is unusable on datasets>=3.
We pull the per-language archives directly and read the JSONL inside — no script,
no trust_remote_code, no version pin.
"""
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# XL-Sum language name -> our language code
XLSUM_LANGS = {
    "hindi": "hi",
    "bengali": "bn",
    "punjabi": "pa",
    "tamil": "ta",
    "telugu": "te",
    "english": "en",
}

# FLORES: both openlanguagedata/flores_plus and facebook/flores are gated and need
# manual terms acceptance on the website. google/IndicGenBench_flores_in is ungated,
# is the same FLORES devtest (1012 sentences), and covers exactly our languages.
# Files are flores_{lang}_en_test.json with an Indic `source` and English `target`;
# FLORES is multi-parallel, so English is identical across files and comes free.
FLORES_LANGS = ["hi", "bn", "pa", "ta", "te"]

CROSSSUM_LANGS = ["hi", "bn", "pa", "ta", "te"]


def get_xlsum():
    out = RAW / "xlsum"
    out.mkdir(parents=True, exist_ok=True)
    for name, code in XLSUM_LANGS.items():
        if (out / f"{code}_test.jsonl").exists():
            print(f"  xlsum {code}: cached")
            continue
        p = hf_hub_download(
            repo_id="csebuetnlp/xlsum",
            filename=f"data/{name}_XLSum_v2.0.tar.bz2",
            repo_type="dataset",
        )
        with tarfile.open(p, "r:bz2") as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".jsonl"):
                    continue
                # members look like <lang>_XLSum_v2.0/<lang>_{train,val,test}.jsonl
                split = Path(m.name).stem.rsplit("_", 1)[-1]
                data = tf.extractfile(m).read()
                (out / f"{code}_{split}.jsonl").write_bytes(data)
        print(f"  xlsum {code}: extracted")


def get_flores():
    import json

    out = RAW / "flores"
    out.mkdir(parents=True, exist_ok=True)

    # Each per-language file holds the same 1012 English sentences but in a
    # different row order, so we key on English to recover the alignment. Without
    # this, "Hindi sentence i" and "Tamil sentence i" would be different content
    # and every cross-language fertility comparison would be confounded.
    by_lang = {}
    for code in FLORES_LANGS:
        p = hf_hub_download(
            repo_id="google/IndicGenBench_flores_in",
            filename=f"flores_{code}_en_test.json",
            repo_type="dataset",
        )
        ex = json.load(open(p, encoding="utf-8"))["examples"]
        clean = lambda s: s.replace("\n", " ").strip()
        by_lang[code] = {clean(e["target"]): clean(e["source"]) for e in ex}
        if len(by_lang[code]) != len(ex):
            raise SystemExit(f"{code}: duplicate English targets, cannot align by key")
        print(f"  flores {code}: {len(ex)} sentences")

    keys = sorted(set.intersection(*(set(d) for d in by_lang.values())))
    if len(keys) != 1012:
        print(f"  note: aligned intersection is {len(keys)} sentences (expected 1012)")
    for code in FLORES_LANGS:
        (out / f"{code}.txt").write_text(
            "\n".join(by_lang[code][k] for k in keys), encoding="utf-8")
    (out / "en.txt").write_text("\n".join(keys), encoding="utf-8")
    print(f"  flores en: {len(keys)} sentences (aligned across all languages)")


def get_crosssum():
    out = RAW / "crosssum_in"
    out.mkdir(parents=True, exist_ok=True)
    for code in CROSSSUM_LANGS:
        for split in ("test", "dev"):
            dst = out / f"{code}_{split}.json"
            if dst.exists():
                continue
            try:
                p = hf_hub_download(
                    repo_id="google/IndicGenBench_crosssum_in",
                    filename=f"crosssum_english-{code}_{split}.json",
                    repo_type="dataset",
                )
                dst.write_bytes(Path(p).read_bytes())
            except Exception as e:  # secondary dataset; never block iteration 1
                print(f"  crosssum {code}/{split}: SKIP ({type(e).__name__})")
        print(f"  crosssum {code}: ok")


if __name__ == "__main__":
    print("FLORES+ (fertility)")
    get_flores()
    print("CrossSum-IN (secondary)")
    get_crosssum()
    print("XL-Sum (primary, ~431MB)")
    get_xlsum()
    print("done")
