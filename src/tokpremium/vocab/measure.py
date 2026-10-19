"""Run the FLORES survey on base + extended tokenizers and print the comparison.

    python -m tokpremium.vocab.measure

Thin wrapper: all metric code lives in survey/. Extended tokenizers are loaded
from their results/vocab/<lang>/<n>/ directories (tokenization.load_tokenizer
accepts a local path as well as a hub id).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from ..survey.aggregate import failure_mode_table
from ..survey.run import run_survey


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vocab.yaml")
    ap.add_argument("--vocab-dir", default="results/vocab")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--lang", help="override target.lang from the config (e.g. hye_Armn)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    lang = args.lang or cfg["target"]["lang"]
    tids = [cfg["base"]["tokenizer_id"]]
    tids += [str(Path(args.vocab_dir) / lang / str(n)) for n in cfg["n_new_tokens"]]
    langs = ["eng_Latn", lang, *cfg["eval"]["controls"]]
    langs = list(dict.fromkeys(langs))

    df = run_survey(langs, tids, split=args.split, limit=args.limit)
    out = Path(args.vocab_dir) / lang
    df.to_parquet(out / "flores_survey.parquet")
    table = failure_mode_table(df)
    table.to_csv(out / "flores_premium.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
