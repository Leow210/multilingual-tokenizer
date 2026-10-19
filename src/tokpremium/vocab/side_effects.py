"""What does a pre-tokenizer change do to languages we are *not* extending?

    python -m tokpremium.vocab.side_effects --config configs/vocab.yaml

A pre-tokenizer applies to all text, not just the target language. For each
target language and each non-shipped pre-tokenizer in the config, encodes every
FLORES+ devtest language with that pre-tokenizer and *no* new merges, and
compares token ids sentence by sentence against shipped Qwen. Any difference is
a change to how the base model sees text it was pretrained on.

Writes results/vocab/pretokenizer_side_effects.csv, one row per
(target_lang, pretokenizer, lang):

  sentences          devtest sentences in that language
  changed            sentences whose token ids differ from shipped Qwen
  tokens_shipped     total tokens under shipped Qwen
  tokens_variant     total tokens under the variant

The first run downloads every FLORES+ language through data/flores.py's cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from ..data.flores import FLORES_PLUS, load_sentences
from . import continue_bpe


def flores_languages() -> list[str]:
    from datasets import get_dataset_config_names

    return sorted(c for c in get_dataset_config_names(FLORES_PLUS) if c != "default")


def compare(base_spec: dict, variant_spec: dict, by_lang: dict[str, list[str]]) -> pd.DataFrame:
    from tokenizers import Tokenizer

    shipped = Tokenizer.from_str(json.dumps(base_spec))
    variant = Tokenizer.from_str(json.dumps(variant_spec))
    rows = []
    for lang, sents in by_lang.items():
        a = shipped.encode_batch(sents, add_special_tokens=False)
        b = variant.encode_batch(sents, add_special_tokens=False)
        rows.append({
            "lang": lang,
            "sentences": len(sents),
            "changed": sum(x.ids != y.ids for x, y in zip(a, b, strict=True)),
            "tokens_shipped": sum(len(x.ids) for x in a),
            "tokens_variant": sum(len(y.ids) for y in b),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/vocab.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    by_lang = {}
    for lang in flores_languages():
        try:
            by_lang[lang] = load_sentences(lang, split="devtest")
        except Exception as exc:  # a config with no devtest split, say
            print(f"  skipping {lang}: {exc}", flush=True)
    print(f"{len(by_lang)} FLORES+ languages loaded", flush=True)

    base_spec = continue_bpe.load_base_spec(cfg["base"]["tokenizer_id"])
    frames = []
    for target in cfg["languages"]:
        for name in cfg["pretokenizers"]:
            if name == "shipped":
                continue
            variant = continue_bpe.PRETOKENIZERS[name](base_spec, target)
            df = compare(base_spec, variant, by_lang)
            frames.append(df.assign(target_lang=target, pretokenizer=name))
            touched = df[df["changed"] > 0]
            print(f"{target} / {name}: {len(touched)} of {len(df)} languages change "
                  f"({', '.join(touched.sort_values('changed', ascending=False)['lang'][:8])})",
                  flush=True)

    out = Path(cfg["output"]["results_dir"]) / "pretokenizer_side_effects.csv"
    cols = ["target_lang", "pretokenizer", "lang", "sentences", "changed",
            "tokens_shipped", "tokens_variant"]
    pd.concat(frames, ignore_index=True)[cols].to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
