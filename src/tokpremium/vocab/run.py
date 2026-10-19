"""Phase B runner: continue Qwen's BPE on a Telugu text file, sweep sizes.

    python -m tokpremium.vocab.run --corpus data/tel_train.txt \
        --heldout data/tel_heldout.txt --english data/eng_sample.txt

Corpus/heldout/english are plain UTF-8, one document per line. The heldout
file must not overlap the training corpus, and FLORES devtest must never be
used here (see configs/vocab.yaml). Writes results/vocab/<lang>/<n>/ and a
summary table at results/vocab/<lang>/sweep.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from ..survey.metrics import normalize
from .continue_bpe import continue_bpe
from .decompose import decompose


def _lines(path: str | Path) -> list[str]:
    return [
        normalize(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def _stats(tok, texts: list[str]) -> dict:
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    n_tok = sum(len(x) for x in ids)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_chars = sum(len(t) for t in texts)
    return {"tokens": n_tok, "bytes_per_token": n_bytes / n_tok, "chars_per_token": n_chars / n_tok}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vocab.yaml")
    ap.add_argument("--corpus", required=True, help="training text, one doc per line")
    ap.add_argument("--heldout", required=True, help="held-out target-language text")
    ap.add_argument("--english", help="English text: merge-learning mix and regression check")
    ap.add_argument("--out", default="results/vocab")
    ap.add_argument("--sizes", type=int, nargs="*", help="override n_new_tokens from config")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--lang", help="override target.lang from the config (e.g. hye_Armn)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    lang = args.lang or cfg["target"]["lang"]
    sizes = args.sizes or cfg["n_new_tokens"]
    base_id = cfg["base"]["tokenizer_id"]

    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast

    base_dir = snapshot_download(base_id, allow_patterns=["tokenizer*", "special_tokens_map.json"])
    base_tok = PreTrainedTokenizerFast.from_pretrained(base_dir)
    base_vocab = base_tok.get_vocab()

    train = _lines(args.corpus)
    held = _lines(args.heldout)
    eng = _lines(args.english) if args.english else []
    # Mix English in at the configured share (by line count) so new merges are
    # not spent on words Qwen already has, and English fertility is exercised.
    mix = cfg["corpus"].get("english_mix", 0.0)
    n_eng = int(len(train) * mix / (1 - mix)) if eng and mix else 0
    texts = train + eng[:n_eng]
    print(f"merge corpus: {len(train):,} {lang} lines + {min(n_eng, len(eng)):,} English lines")

    held_base = _stats(base_tok, held)
    eng_base = _stats(base_tok, eng) if eng else None
    rows = []
    for n in sizes:
        out = Path(args.out) / lang / str(n)
        new = continue_bpe(base_dir, texts, n, out, min_count=args.min_count)
        (out / "decomposition.json").write_text(
            json.dumps({str(k): v for k, v in decompose(new, base_vocab).items()}), encoding="utf-8"
        )
        tok = PreTrainedTokenizerFast.from_pretrained(out)
        h = _stats(tok, held)
        row = {
            "n_requested": n,
            "n_added": len(new),
            "heldout_tokens_base": held_base["tokens"],
            "heldout_tokens_new": h["tokens"],
            "token_ratio": h["tokens"] / held_base["tokens"],
            "bytes_per_token_new": h["bytes_per_token"],
            "roundtrip_ok": all(tok.decode(tok(t, add_special_tokens=False)["input_ids"]) == t
                                for t in held[:200]),
        }
        if eng_base:
            e = _stats(tok, eng)
            row["english_token_ratio"] = e["tokens"] / eng_base["tokens"]
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    (Path(args.out) / lang).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / lang / "sweep.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
