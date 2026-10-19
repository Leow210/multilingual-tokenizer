"""Score extended tokenizers with the survey's own code (plan step B, last part).

    python -m tokpremium.vocab.measure --config configs/vocab.yaml

For every language, pre-tokenizer and sweep size, rebuilds the tokenizer from
results/vocab/<lang>/<pretokenizer>/merges.json into a gitignored folder, runs
survey.run.run_survey on the target language plus the controls, and reduces it
with survey.aggregate. No metric is reimplemented here.

Writes results/vocab/summary.csv, one row per (lang, pretokenizer, n_new):

  premium_median        paired median tokens vs English, FLORES+ devtest
  subchar_rate          share of tokens that are character fragments
  bytes_per_token       secondary efficiency number
  token_ratio           target-language tokens / tokens under shipped Qwen
  eng_token_ratio       same for English; must be exactly 1
  cmn_premium           Chinese control; must stay ~0.96
  roundtrip_ok          share of target sentences that decode back exactly
  lm_head_cost_ratio    embedding/lm-head rows after / before (the plan's cost term)
  per_token_cost_ratio  whole forward pass FLOPs per token after / before
  net_cost_ratio        token_ratio * per_token_cost_ratio; below 1 is a real saving

The cost model counts matmul FLOPs only: attention and MLP weights plus the d x V
output projection (the input embedding is a lookup). It ignores attention over
the context and memory bandwidth, so it understates what a bigger vocabulary
costs at decode time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from ..data.flores import load_sentences
from ..survey import tokenization
from ..survey.aggregate import by_language, paired_premium
from ..survey.metrics import net_cost_ratio
from ..survey.run import run_survey
from . import continue_bpe


def forward_flops(cfg, vocab_rows: int) -> int:
    """Matmul weights touched per token, which is proportional to forward FLOPs."""
    d, hd = cfg.hidden_size, cfg.head_dim
    attn = d * cfg.num_attention_heads * hd * 2 + d * cfg.num_key_value_heads * hd * 2
    mlp = 3 * d * cfg.intermediate_size
    return cfg.num_hidden_layers * (attn + mlp) + d * vocab_rows


def build_sweep(cfg: dict, lang: str) -> list[dict]:
    """Tokenizer folders for every (pretokenizer, n) point. Returns one record per point."""
    base_spec = continue_bpe.load_base_spec(cfg["base"]["tokenizer_id"])
    points = []
    for name in cfg["pretokenizers"]:
        merges = continue_bpe.load_merges(
            continue_bpe.merges_path(cfg["output"]["results_dir"], lang, name)
        )
        for n in [0, *cfg["n_new_tokens"]]:
            out = Path(cfg["output"]["tokenizer_dir"]) / lang / name / str(n)
            if not (out / "tokenizer.json").exists():
                continue_bpe.build_tokenizer(base_spec, name, merges[:n], out)
            points.append({"lang": lang, "pretokenizer": name, "n_new": min(n, len(merges)),
                           "tokenizer_id": str(out)})
    return points


def score(cfg: dict, lang: str) -> pd.DataFrame:
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(cfg["base"]["model_id"])
    pivot = cfg["eval"]["pivot"]
    langs = list(dict.fromkeys([lang, *cfg["eval"]["controls"]]))
    points = build_sweep(cfg, lang)
    shipped = cfg["base"]["tokenizer_id"]

    df = run_survey(langs, [shipped] + [p["tokenizer_id"] for p in points])
    prem = paired_premium(df, pivot_lang=pivot).set_index(["tokenizer_id", "lang"])
    agg = by_language(df).set_index(["tokenizer_id", "lang"])
    target_text = load_sentences(lang)

    base_rows = model_cfg.vocab_size
    flops_before = forward_flops(model_cfg, base_rows)
    rows = []
    for p in points:
        tid = p["tokenizer_id"]
        tok = tokenization.load_tokenizer(tid)
        # Qwen pads its embedding table past the real vocabulary, so the first
        # few hundred new tokens land in rows that already exist.
        rows_after = max(base_rows, max(tok.get_vocab().values()) + 1)
        cost = forward_flops(model_cfg, rows_after) / flops_before
        token_ratio = agg.loc[(tid, lang), "tokens"] / agg.loc[(shipped, lang), "tokens"]
        rows.append({
            **{k: p[k] for k in ("lang", "pretokenizer", "n_new")},
            "premium_median": prem.loc[(tid, lang), "premium_median"],
            "subchar_rate": agg.loc[(tid, lang), "subchar_rate"],
            "bytes_per_token": agg.loc[(tid, lang), "bytes_per_token"],
            "token_ratio": token_ratio,
            "eng_token_ratio": agg.loc[(tid, pivot), "tokens"]
            / agg.loc[(shipped, pivot), "tokens"],
            "cmn_premium": prem.loc[(tid, "cmn_Hans"), "premium_median"],
            "roundtrip_ok": sum(tokenization.roundtrip_ok(tok, s) for s in target_text)
            / len(target_text),
            "lm_head_cost_ratio": rows_after / base_rows,
            "per_token_cost_ratio": cost,
            "net_cost_ratio": net_cost_ratio(token_ratio, cost),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/vocab.yaml")
    ap.add_argument("--lang", action="append", help="limit to these languages (repeatable)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out = Path(cfg["output"]["results_dir"]) / "summary.csv"
    frames = [score(cfg, lang) for lang in args.lang or cfg["languages"]]
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(out, index=False, float_format="%.4f")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
