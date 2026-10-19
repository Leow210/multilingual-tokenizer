r"""Tests for BPE continuation (vocab/continue_bpe.py, vocab/decompose.py).

These are the properties .contexts/vocabExtension-plan.md asks to lock before
any extended tokenizer is trusted. They run offline against a tiny tokenizer
built the way Qwen's is: Qwen's split regex, byte-level BPE, NFC, and a special
token whose id sits after the model vocabulary.
"""

import json
import unicodedata

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from tokpremium.vocab.continue_bpe import (
    apply_merges,
    count_chunks,
    extend,
    is_ascii_piece,
    keep_marks,
    keep_marks_target,
    learn_merges,
    piece_bytes,
    script_marks,
    segment,
)
from tokpremium.vocab.decompose import decomposer

QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
ENGLISH = ["Hello world, it's a test.", "The quick brown fox jumps over the lazy dog."]
TELUGU = ["తెలుగు భాష", "తెలుగు ప్రజలు తెలుగు మాట్లాడతారు", "భాష తెలుగు"]


def make_base() -> dict:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tok.train_from_iterator(ENGLISH * 20, trainer)
    tok.add_special_tokens(["<|endoftext|>"])
    return json.loads(tok.to_str())


@pytest.fixture(scope="module")
def base():
    return make_base()


def learn(spec, lines, n, accept=lambda t: not is_ascii_piece(t)):
    words = segment(count_chunks(lines, spec), spec)
    return learn_merges(words, spec["model"]["vocab"], n, accept=accept)


def encode(spec, text):
    return Tokenizer.from_str(json.dumps(spec)).encode(text, add_special_tokens=False).ids


def test_old_merges_prefix_new_and_ids_only_append(base):
    merges = learn(base, TELUGU * 5, 20)
    new, new_ids = extend(base, merges)
    old_merges = base["model"]["merges"]
    assert new["model"]["merges"][: len(old_merges)] == old_merges
    for token, i in base["model"]["vocab"].items():
        assert new["model"]["vocab"][token] == i
    assert new["added_tokens"] == base["added_tokens"]
    special = max(t["id"] for t in base["added_tokens"])
    assert new_ids == list(range(special + 1, special + 1 + len(merges)))


def test_special_tokens_keep_their_ids_after_loading(base):
    """tokenizers re-numbers added tokens on load; the extension must survive that."""
    merges = learn(base, TELUGU * 5, 20)
    new, new_ids = extend(base, merges)
    loaded = Tokenizer.from_str(json.dumps(new))
    for special in base["added_tokens"]:
        assert loaded.token_to_id(special["content"]) == special["id"]
    for (a, b), i in zip(merges, new_ids, strict=True):
        assert loaded.id_to_token(i) == a + b


def test_english_encoding_is_unchanged(base):
    # English in the training text must not buy new merges.
    merges = learn(base, (ENGLISH + TELUGU) * 5, 40)
    new, _ = extend(base, merges)
    assert all(not is_ascii_piece(a + b) for a, b in merges)
    for text in ENGLISH + ["Hello world"]:
        assert encode(new, text) == encode(base, text)


def test_three_byte_character_becomes_one_token(base):
    char = "త"
    assert len(char.encode("utf-8")) == 3
    assert len(encode(base, char)) == 3
    new, _ = extend(base, learn(base, TELUGU * 5, 40))
    assert len(encode(new, char)) == 1


def test_learned_segmentation_matches_the_tokenizer(base):
    """Simulated merges during learning are exactly what the extended tokenizer does."""
    merges = learn(base, TELUGU * 5, 40)
    new_tok = Tokenizer.from_str(json.dumps(extend(base, merges)[0]))
    for chunk, _ in count_chunks(TELUGU, base).items():
        old = [t.value for t in Tokenizer.from_str(json.dumps(base)).model.tokenize(chunk)]
        assert apply_merges(old, merges) == [t.value for t in new_tok.model.tokenize(chunk)]


def test_decomposition_reencodes_to_the_same_bytes(base):
    merges = learn(base, TELUGU * 5, 40)
    new, new_ids = extend(base, merges)
    old_vocab = {i: t for t, i in base["model"]["vocab"].items()}
    split = decomposer(base)
    for a, b in merges:
        ids = split(a + b)
        assert ids and all(i in old_vocab for i in ids)
        assert b"".join(piece_bytes(old_vocab[i]) for i in ids) == piece_bytes(a + b)


def test_nfc_in_nfc_out(base):
    new, _ = extend(base, learn(base, TELUGU * 5, 40))
    tok = Tokenizer.from_str(json.dumps(new))
    text = unicodedata.normalize("NFD", "café తెలుగు")
    assert tok.decode(tok.encode(text, add_special_tokens=False).ids) == unicodedata.normalize(
        "NFC", text
    )


def test_pairs_that_already_exist_are_never_merged():
    words = {("a", "b"): 10, ("b", "c"): 5}
    assert learn_merges(words, {"a", "b", "c", "ab"}, 5) == [("b", "c")]


def test_merges_are_most_frequent_first_and_stop_at_n():
    words = {("x", "y", "z"): 3, ("x", "y"): 4, ("z", "q"): 1}
    merges = learn_merges(words, {"x", "y", "z", "q"}, 2)
    assert merges == [("x", "y"), ("xy", "z")]


def test_keep_marks_keeps_a_telugu_word_whole(base):
    word = "తెలుగు"  # vowel signs are \p{M}
    shipped = Tokenizer.from_str(json.dumps(base)).pre_tokenizer
    marked = Tokenizer.from_str(json.dumps(keep_marks(base))).pre_tokenizer
    assert len(shipped.pre_tokenize_str(word)) > 1
    assert len(marked.pre_tokenize_str(word)) == 1
    # Text without marks pre-tokenizes the same either way.
    for text in ENGLISH:
        assert shipped.pre_tokenize_str(text) == marked.pre_tokenize_str(text)


def chunks(spec, text):
    return [c for c, _ in Tokenizer.from_str(json.dumps(spec)).pre_tokenizer.pre_tokenize_str(text)]


def test_keep_marks_target_only_touches_the_target_script(base):
    telugu = keep_marks_target(base, "tel_Telu")
    assert len(chunks(telugu, "తెలుగు")) == 1
    # Other scripts with combining marks are cut exactly as shipped Qwen cuts them.
    for text in ["हिन्दी", "ขณะนี้", "தமிழ்நாடு", "مَرْحَبًا", *ENGLISH]:
        assert chunks(telugu, text) == chunks(base, text)


def test_keep_marks_target_is_a_no_op_for_a_script_without_marks(base):
    assert script_marks("Armn") == ""
    assert keep_marks_target(base, "hye_Armn") == base


def test_script_marks_lists_only_combining_marks():
    cls = script_marks("Telu")
    assert r"\x{0C3E}" in cls  # vowel sign AA
    assert "0C15" not in cls  # KA is a letter


def test_unknown_script_is_an_error(base):
    with pytest.raises(ValueError, match="SCRIPT_BLOCKS"):
        keep_marks_target(base, "xyz_Zzzz")


def test_subchar_mask_covers_ids_past_the_special_tokens(base, tmp_path):
    """Appended ids sit after the special tokens, so vocab_size is not the id range."""
    from transformers import PreTrainedTokenizerFast

    from tokpremium.survey.tokenization import _subchar_vocab_mask

    char = "త".encode()
    # Force one new token that is two thirds of a character: a fragment by construction.
    vocab = base["model"]["vocab"]
    pieces = [next(t for t in vocab if piece_bytes(t) == bytes([b])) for b in char[:2]]
    new, new_ids = extend(base, [tuple(pieces)])
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(new))).save_pretrained(
        tmp_path
    )
    assert new_ids[0] in _subchar_vocab_mask(str(tmp_path))
