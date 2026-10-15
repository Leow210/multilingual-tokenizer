"""Tests for the survey aggregations.

The paired premium is the headline number in the report, so it gets the most
attention here. The key property under test is that it compares sentence N
against sentence N, which is what makes the comparison meaningful.
"""

import pandas as pd
import pytest

from tokpremium.survey.aggregate import (
    by_language,
    failure_mode_table,
    paired_premium,
    premium_matrix,
    validate_schema,
)


def make_df():
    """Three languages, four sentences, one tokenizer.

    Constructed so the answers are known by hand: relative to English, 'cheap'
    is 1x on every sentence and 'pricey' is 3x on every sentence.
    """
    rows = []
    eng_tokens = [10, 20, 30, 40]
    for sent_id, n in enumerate(eng_tokens):
        rows.append(
            dict(lang="eng_Latn", sent_id=sent_id, tokenizer_id="tok_a",
                 n_tokens=n, n_chars=n * 4, n_bytes=n * 4,
                 n_subchar_tokens=0, n_fallback_pieces=0)
        )
        rows.append(
            dict(lang="cheap_Latn", sent_id=sent_id, tokenizer_id="tok_a",
                 n_tokens=n, n_chars=n * 4, n_bytes=n * 4,
                 n_subchar_tokens=0, n_fallback_pieces=0)
        )
        rows.append(
            dict(lang="pricey_Ethi", sent_id=sent_id, tokenizer_id="tok_a",
                 n_tokens=n * 3, n_chars=n * 2, n_bytes=n * 6,
                 n_subchar_tokens=int(n * 3 * 0.5), n_fallback_pieces=int(n * 3 * 0.5))
        )
    return pd.DataFrame(rows)


class TestValidateSchema:
    def test_accepts_good_table(self):
        validate_schema(make_df())

    def test_rejects_missing_column(self):
        df = make_df().drop(columns=["n_bytes"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_schema(df)

    def test_rejects_duplicate_rows(self):
        # Catches the classic mistake of appending a second run into one file.
        df = pd.concat([make_df(), make_df()], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            validate_schema(df)


class TestByLanguage:
    def test_one_row_per_language(self):
        out = by_language(make_df())
        assert len(out) == 3
        assert set(out["lang"]) == {"eng_Latn", "cheap_Latn", "pricey_Ethi"}

    def test_bytes_per_token(self):
        out = by_language(make_df()).set_index("lang")
        assert out.loc["eng_Latn", "bytes_per_token"] == pytest.approx(4.0)
        assert out.loc["pricey_Ethi", "bytes_per_token"] == pytest.approx(2.0)

    def test_subchar_rate(self):
        out = by_language(make_df()).set_index("lang")
        assert out.loc["eng_Latn", "subchar_rate"] == pytest.approx(0.0)
        assert out.loc["pricey_Ethi", "subchar_rate"] == pytest.approx(0.5, abs=0.02)


class TestPairedPremium:
    def test_known_ratios(self):
        out = paired_premium(make_df()).set_index("lang")
        assert out.loc["eng_Latn", "premium_median"] == pytest.approx(1.0)
        assert out.loc["cheap_Latn", "premium_median"] == pytest.approx(1.0)
        assert out.loc["pricey_Ethi", "premium_median"] == pytest.approx(3.0)

    def test_pairing_beats_corpus_ratio(self):
        """The property that makes FLORES worth using.

        Per-sentence ratios of 10x and 1x give a median of 5.5, while the ratio
        of the totals is 110/101 = 1.09, which hides the short-sentence blowup
        entirely. Storing per-language averages makes only the second number
        recoverable.
        """
        rows = [
            dict(lang="eng_Latn", sent_id=0, tokenizer_id="t", n_tokens=1,
                 n_chars=4, n_bytes=4, n_subchar_tokens=0, n_fallback_pieces=0),
            dict(lang="eng_Latn", sent_id=1, tokenizer_id="t", n_tokens=100,
                 n_chars=400, n_bytes=400, n_subchar_tokens=0, n_fallback_pieces=0),
            dict(lang="xx_Test", sent_id=0, tokenizer_id="t", n_tokens=10,
                 n_chars=4, n_bytes=4, n_subchar_tokens=0, n_fallback_pieces=0),
            dict(lang="xx_Test", sent_id=1, tokenizer_id="t", n_tokens=100,
                 n_chars=400, n_bytes=400, n_subchar_tokens=0, n_fallback_pieces=0),
        ]
        out = paired_premium(pd.DataFrame(rows)).set_index("lang")
        assert out.loc["xx_Test", "premium_median"] == pytest.approx(5.5)

    def test_median_resists_outliers_that_move_the_mean(self):
        rows = []
        for sent_id in range(10):
            rows.append(dict(lang="eng_Latn", sent_id=sent_id, tokenizer_id="t",
                             n_tokens=10, n_chars=40, n_bytes=40,
                             n_subchar_tokens=0, n_fallback_pieces=0))
            n = 20 if sent_id < 9 else 500      # one pathological sentence
            rows.append(dict(lang="xx_Test", sent_id=sent_id, tokenizer_id="t",
                             n_tokens=n, n_chars=40, n_bytes=40,
                             n_subchar_tokens=0, n_fallback_pieces=0))
        out = paired_premium(pd.DataFrame(rows)).set_index("lang")
        assert out.loc["xx_Test", "premium_median"] == pytest.approx(2.0)
        assert out.loc["xx_Test", "premium_mean"] > 6.0

    def test_rejects_unknown_pivot(self):
        with pytest.raises(ValueError, match="pivot language"):
            paired_premium(make_df(), pivot_lang="zzz_Zzzz")


class TestPremiumMatrix:
    def test_shape_and_values(self):
        wide = premium_matrix(make_df())
        assert list(wide.columns) == ["tok_a"]
        assert wide.loc["pricey_Ethi", "tok_a"] == pytest.approx(3.0)


class TestFailureMode:
    def test_labels_by_subchar_rate(self):
        out = failure_mode_table(make_df()).set_index("lang")
        assert out.loc["pricey_Ethi", "failure_mode"] == "script-uncovered"
        assert out.loc["cheap_Latn", "failure_mode"] == "poor-merges"
