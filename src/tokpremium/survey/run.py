"""Survey runner: FLORES sentences x languages x tokenizers -> one parquet.

    python -m tokpremium.survey.run --config configs/survey.yaml

Writes one row per (language, sentence, tokenizer). It does NOT write
averages; see aggregate.py for why. Rerunning overwrites the output, and the
schema check in aggregate.validate_schema catches an accidental append.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from ..data.flores import check_alignment, load_sentences
from . import tokenization
from .metrics import sentence_row


def run_survey(
    languages: list[str],
    tokenizer_ids: list[str],
    split: str = "devtest",
    cache_dir: str = "data/flores",
    limit: int | None = None,
) -> pd.DataFrame:
    """Measure every (language, sentence, tokenizer) combination."""
    by_lang: dict[str, list[str]] = {}
    for lang in languages:
        print(f"  loading {lang} ...", flush=True)
        sents = load_sentences(lang, split=split, cache_dir=cache_dir)
        by_lang[lang] = sents[:limit] if limit else sents
    check_alignment(by_lang)

    rows = []
    for tid in tokenizer_ids:
        print(f"  tokenizing with {tid} ...", flush=True)
        tok = tokenization.load_tokenizer(tid)
        family = tokenization.tokenizer_family(tok)
        vocab = getattr(tok, "vocab_size", None) or len(tok)
        print(f"    family={family} vocab={vocab:,}", flush=True)
        print("    scanning vocabulary for character fragments ...", flush=True)
        for lang, sents in by_lang.items():
            for sent_id, text in enumerate(sents):
                ids, pieces = tokenization.encode(tok, text)
                flags = tokenization.subchar_flags(tid, ids)
                rows.append(
                    sentence_row(
                        lang=lang,
                        sent_id=sent_id,
                        tokenizer_id=tid,
                        text=text,
                        pieces=pieces,
                        subchar_flags=flags,
                    ).as_dict()
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/survey.yaml")
    ap.add_argument("--out", default="results/survey.parquet")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N sentences per language, for a smoke test",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    languages = [entry["code"] for entry in cfg["languages"]]
    tokenizer_ids = [entry["tokenizer_id"] for entry in cfg["tokenizers"]]
    split = cfg.get("split", "devtest")
    cache_dir = cfg.get("cache_dir", "data/flores")

    print(f"{len(languages)} languages x {len(tokenizer_ids)} tokenizers, split={split}")
    df = run_survey(
        languages, tokenizer_ids, split=split, cache_dir=cache_dir, limit=args.limit
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
