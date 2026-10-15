"""Aggregations over the sentence-level survey table.

All of these are groupbys or pivots over the parquet written by survey/run.py.
Keeping them here rather than in a notebook means the numbers that go in the
report come from tested code, and a plot can be redrawn without re-running the
survey.

A note on denominators. We report bytes_per_token as the primary efficiency
number rather than chars_per_token, because a code point is not a comparable
unit across scripts: one Han character carries far more information than one
Latin letter, so chars-per-token flatters logographic scripts and penalizes
alphabetic ones. Bytes are not perfect either, since UTF-8 spends more bytes
on non-Latin scripts by design, which is part of what we are measuring. We
report both and lean on the paired premium below, which sidesteps the question
entirely by holding the content constant.
"""

from __future__ import annotations

import pandas as pd

SENTENCE_SCHEMA = {
    "lang": "string",
    "sent_id": "int64",
    "tokenizer_id": "string",
    "n_tokens": "int64",
    "n_chars": "int64",
    "n_bytes": "int64",
    "n_subchar_tokens": "int64",
    "n_fallback_pieces": "int64",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Fail loudly if the survey table is not what everything downstream expects."""
    missing = set(SENTENCE_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"survey table is missing columns: {sorted(missing)}")
    dupes = df.duplicated(subset=["lang", "sent_id", "tokenizer_id"]).sum()
    if dupes:
        raise ValueError(
            f"{dupes} duplicate (lang, sent_id, tokenizer_id) rows; "
            "the survey was probably run twice into the same file"
        )


def by_language(df: pd.DataFrame) -> pd.DataFrame:
    """Per (tokenizer, language) summary.

    Ratios come from summed totals rather than a mean of per-sentence ratios,
    which is right for a corpus-level efficiency figure: it weights by length
    instead of letting one short sentence count as much as one long one.
    """
    validate_schema(df)
    g = df.groupby(["tokenizer_id", "lang"], as_index=False).agg(
        n_sentences=("sent_id", "count"),
        tokens=("n_tokens", "sum"),
        chars=("n_chars", "sum"),
        bytes=("n_bytes", "sum"),
        subchar_tokens=("n_subchar_tokens", "sum"),
        fallback_pieces=("n_fallback_pieces", "sum"),
    )
    g["tokens_per_sentence"] = g["tokens"] / g["n_sentences"]
    g["bytes_per_token"] = g["bytes"] / g["tokens"]
    g["chars_per_token"] = g["chars"] / g["tokens"]
    g["subchar_rate"] = g["subchar_tokens"] / g["tokens"]
    g["fallback_rate"] = g["fallback_pieces"] / g["tokens"]
    return g


def paired_premium(df: pd.DataFrame, pivot_lang: str = "eng_Latn") -> pd.DataFrame:
    """Per-sentence token premium against a pivot language.

    This is the headline number and the reason FLORES is the right corpus:
    sent_id N is a translation of the same source sentence in every language,
    so tokens(lang, N) / tokens(pivot, N) isolates the tokenizer from any
    difference in what is being said.

    The median is the headline rather than the mean, because the ratio
    distribution has a long right tail (a handful of sentences with rare proper
    nouns or digits blow up) and the mean chases it.
    """
    validate_schema(df)
    if pivot_lang not in set(df["lang"]):
        raise ValueError(f"pivot language {pivot_lang!r} is not in the survey table")

    pivot = (
        df[df["lang"] == pivot_lang]
        .loc[:, ["tokenizer_id", "sent_id", "n_tokens"]]
        .rename(columns={"n_tokens": "pivot_tokens"})
    )
    merged = df.merge(pivot, on=["tokenizer_id", "sent_id"], how="inner")
    merged = merged[merged["pivot_tokens"] > 0].copy()
    merged["ratio"] = merged["n_tokens"] / merged["pivot_tokens"]

    out = merged.groupby(["tokenizer_id", "lang"], as_index=False).agg(
        n_paired=("ratio", "count"),
        premium_median=("ratio", "median"),
        premium_mean=("ratio", "mean"),
        premium_p90=("ratio", lambda s: s.quantile(0.90)),
    )
    return out.sort_values(
        ["tokenizer_id", "premium_median"], ascending=[True, False]
    ).reset_index(drop=True)


def premium_matrix(df: pd.DataFrame, pivot_lang: str = "eng_Latn") -> pd.DataFrame:
    """Languages as rows, tokenizers as columns, median premium in the cells.

    This is the table that goes in the report. If a language is expensive in
    every column, the problem is structural rather than one vendor's choice,
    which is the claim we want to be able to make.
    """
    prem = paired_premium(df, pivot_lang=pivot_lang)
    wide = prem.pivot(index="lang", columns="tokenizer_id", values="premium_median")
    return wide.sort_values(by=list(wide.columns), ascending=False)


def failure_mode_table(df: pd.DataFrame, pivot_lang: str = "eng_Latn") -> pd.DataFrame:
    """Premium alongside sub-character rate. This is what target selection uses.

    Two languages can share a premium of 4x for different reasons. A Latin
    script language usually gets there through poor merges, with a
    sub-character rate near zero. A language whose script is barely in the
    vocabulary gets there by being spelled out, with a high sub-character rate.
    Those are different failure modes and the repair may behave differently on
    each, so we pick one of each rather than the two worst.
    """
    prem = paired_premium(df, pivot_lang=pivot_lang)
    lang = by_language(df)[
        ["tokenizer_id", "lang", "bytes_per_token", "subchar_rate", "fallback_rate"]
    ]
    out = prem.merge(lang, on=["tokenizer_id", "lang"], how="left")
    out["failure_mode"] = out["subchar_rate"].apply(
        lambda r: "script-uncovered" if r >= 0.10 else "poor-merges"
    )
    return out
