"""Tests for the pure metric functions.

These run without transformers, torch, or a network connection, which is the
point: the numbers that end up in the report come from this code, so it should
be testable in under a second by anyone who just cloned the repo.
"""

import pytest

from tokpremium.survey.metrics import (
    SentenceRow,
    count_byte_fallback_pieces,
    count_bytes,
    count_chars,
    count_subchar_tokens,
    is_byte_fallback_piece,
    net_cost_ratio,
    normalize,
    sentence_row,
)


class TestNormalize:
    def test_composed_and_decomposed_agree(self):
        composed = "é"           # U+00E9
        decomposed = "é"   # e + combining acute
        assert composed != decomposed
        assert normalize(composed) == normalize(decomposed)

    def test_normalization_changes_char_count(self):
        # The reason we normalize at load time: the same visible string can
        # otherwise carry a different code point count depending on the source.
        assert count_chars("é") == 2
        assert count_chars(normalize("é")) == 1


class TestCounts:
    def test_ascii_bytes_equal_chars(self):
        assert count_chars("hello") == count_bytes("hello") == 5

    def test_non_latin_bytes_exceed_chars(self):
        # Amharic: 3 UTF-8 bytes per code point. This gap is a large part of
        # what the survey measures.
        text = "ሰላም"  # 3 syllable characters
        assert count_chars(text) == 3
        assert count_bytes(text) == 9

    def test_han_is_three_bytes_per_char(self):
        assert count_chars("中文") == 2
        assert count_bytes("中文") == 6


class TestByteFallback:
    @pytest.mark.parametrize("piece", ["<0xE1>", "<0x00>", "<0xff>", "<0xAB>"])
    def test_recognizes_fallback_pieces(self, piece):
        assert is_byte_fallback_piece(piece)

    @pytest.mark.parametrize(
        "piece", ["▁hello", "<0xGG>", "0xE1", "<unk>", "<0x1>", "<0xE111>", ""]
    )
    def test_rejects_non_fallback_pieces(self, piece):
        assert not is_byte_fallback_piece(piece)

    def test_counts_only_fallbacks(self):
        pieces = ["▁the", "<0xE1>", "<0x88>", "cat", "<0xB0>"]
        assert count_byte_fallback_pieces(pieces) == 3

    def test_empty_is_zero(self):
        assert count_byte_fallback_pieces([]) == 0


class TestSubchar:
    def test_counts_true_flags(self):
        assert count_subchar_tokens([True, False, True, True]) == 3

    def test_empty_is_zero(self):
        assert count_subchar_tokens([]) == 0


class TestSentenceRow:
    def test_builds_expected_row(self):
        row = sentence_row(
            lang="amh_Ethi",
            sent_id=7,
            tokenizer_id="test/tok",
            text="ሰላም",
            pieces=["<0xE1>", "<0x88>", "<0xB0>", "ላም"],
            subchar_flags=[True, True, True, False],
        )
        assert isinstance(row, SentenceRow)
        assert row.n_tokens == 4
        assert row.n_chars == 3
        assert row.n_bytes == 9
        assert row.n_subchar_tokens == 3
        assert row.n_fallback_pieces == 3
        assert row.as_dict()["lang"] == "amh_Ethi"

    def test_flags_default_to_empty(self):
        row = sentence_row(
            lang="eng_Latn",
            sent_id=0,
            tokenizer_id="test/tok",
            text="hello",
            pieces=["hel", "lo"],
        )
        assert row.n_subchar_tokens == 0
        assert row.n_tokens == 2

    def test_row_is_frozen(self):
        row = sentence_row(
            lang="eng_Latn", sent_id=0, tokenizer_id="t", text="a", pieces=["a"]
        )
        with pytest.raises(Exception):
            row.n_tokens = 99


class TestNetCostRatio:
    def test_larger_vocab_eats_part_of_the_gain(self):
        # 2.5x fewer tokens, 10% more expensive per token, so the real saving
        # is 2.27x rather than 2.5x.
        assert net_cost_ratio(0.40, 1.10) == pytest.approx(0.44)

    def test_can_erase_the_gain_entirely(self):
        assert net_cost_ratio(0.80, 1.25) == pytest.approx(1.0)

    def test_neutral_when_cost_unchanged(self):
        assert net_cost_ratio(0.5, 1.0) == pytest.approx(0.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive(self, bad):
        with pytest.raises(ValueError):
            net_cost_ratio(bad, 1.0)
        with pytest.raises(ValueError):
            net_cost_ratio(1.0, bad)
