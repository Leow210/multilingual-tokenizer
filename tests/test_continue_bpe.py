"""Tests for continued BPE. Needs the `tokenizers` package, not a network.

A tiny byte-level BPE is built in-memory so the invariants can be checked
without downloading Qwen.
"""

import json

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel  # noqa: E402

from tokpremium.vocab.continue_bpe import (  # noqa: E402
    continue_bpe,
    extend_tokenizer_json,
    learn_merges,
    next_free_id,
    word_counts,
)
from tokpremium.vocab.decompose import decompose  # noqa: E402


def _base_tokenizer(tmp_path):
    """Byte-level BPE with the 256 byte symbols, one English merge, one special."""
    alphabet = ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    vocab["he"] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[("h", "e")]))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(r"\p{L}+|\p{M}+| ?[^\s\p{L}\p{M}]+|\s+", "isolated"),
         ByteLevel(add_prefix_space=False, use_regex=False)]
    )
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<eos>"])
    d = tmp_path / "base"
    d.mkdir()
    tok.save(str(d / "tokenizer.json"))
    return d, tok


TELUGU = ["తెలుగు భాష", "తెలుగు వార్తలు", "తెలుగు సినిమా"] * 20


def test_old_merges_are_a_prefix_and_ids_append(tmp_path):
    base, tok = _base_tokenizer(tmp_path)
    old = json.loads((base / "tokenizer.json").read_text())
    new = continue_bpe(base, TELUGU, 8, tmp_path / "ext")
    ext = json.loads((tmp_path / "ext" / "tokenizer.json").read_text())

    n_old = len(old["model"]["merges"])
    assert ext["model"]["merges"][:n_old] == old["model"]["merges"]
    assert len(ext["model"]["merges"]) == n_old + len(new) > n_old
    # Old surface forms keep their ids.
    for t, i in old["model"]["vocab"].items():
        assert ext["model"]["vocab"][t] == i
    # New ids sit above the special token too.
    special = max(a["id"] for a in old["added_tokens"])
    assert min(m.token_id for m in new) == special + 1


def test_special_token_ids_survive_reload(tmp_path):
    # Regression: the loader re-assigns specials that are absent from
    # model.vocab, so ids must be checked on the LOADED tokenizer, not the json.
    base, tok = _base_tokenizer(tmp_path)
    before = tok.token_to_id("<eos>")
    new = continue_bpe(base, TELUGU, 20, tmp_path / "ext")
    ext = Tokenizer.from_file(str(tmp_path / "ext" / "tokenizer.json"))
    assert ext.token_to_id("<eos>") == before
    for m in new:
        assert ext.id_to_token(m.token_id) == m.token


def test_english_encoding_unchanged(tmp_path):
    base, tok = _base_tokenizer(tmp_path)
    continue_bpe(base, TELUGU, 20, tmp_path / "ext")
    ext = Tokenizer.from_file(str(tmp_path / "ext" / "tokenizer.json"))
    s = "he said hello, the hero helped"
    assert ext.encode(s).tokens == tok.encode(s).tokens


def test_telugu_gets_shorter_and_roundtrips(tmp_path):
    base, tok = _base_tokenizer(tmp_path)
    continue_bpe(base, TELUGU, 30, tmp_path / "ext")
    ext = Tokenizer.from_file(str(tmp_path / "ext" / "tokenizer.json"))
    s = "తెలుగు భాష"
    assert len(ext.encode(s).ids) < len(tok.encode(s).ids)
    assert ext.decode(ext.encode(s).ids) == s


def test_decomposition_reencodes_to_same_bytes(tmp_path):
    base, tok = _base_tokenizer(tmp_path)
    old_vocab = tok.get_vocab()
    new = continue_bpe(base, TELUGU, 30, tmp_path / "ext")
    dec = decompose(new, old_vocab)
    inv = {i: t for t, i in old_vocab.items()}
    assert set(dec) == {m.token_id for m in new}
    for m in new:
        assert "".join(inv[i] for i in dec[m.token_id]) == m.token


def test_min_count_stops_early(tmp_path):
    base, tok = _base_tokenizer(tmp_path)
    new = continue_bpe(base, ["అ"], 50, tmp_path / "ext", min_count=2)
    assert new == []


def test_no_duplicate_tokens(tmp_path):
    base, tok = _base_tokenizer(tmp_path)
    new = continue_bpe(base, TELUGU, 40, tmp_path / "ext")
    toks = [m.token for m in new]
    assert len(toks) == len(set(toks))
    assert not set(toks) & set(tok.get_vocab())


def test_extend_accepts_string_style_merges():
    spec = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
        "added_tokens": [{"id": 5, "content": "<s>"}],
    }
    assert next_free_id(spec) == 6
    from tokpremium.vocab.continue_bpe import NewMerge

    out = extend_tokenizer_json(spec, [NewMerge(6, "ab" + "a", "ab", "a", 3)])
    assert out["model"]["merges"][-1] == "ab a"
    assert spec["model"]["merges"] == ["a b"]  # input untouched
