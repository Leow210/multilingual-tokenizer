"""The only module that touches a live tokenizer.

Quarantining transformers here keeps metrics.py and aggregate.py importable
and testable without downloading a model, which matters because those are the
modules whose numbers end up in the report.
"""

from __future__ import annotations

from functools import lru_cache

from .metrics import REPLACEMENT_CHAR


@lru_cache(maxsize=8)
def load_tokenizer(tokenizer_id: str):
    """Load and cache a tokenizer by Hugging Face id.

    Cached because the survey loops languages inside tokenizers, and reloading
    per language turns a two-minute job into a twenty-minute one.

    Built straight from the repo's tokenizer.json rather than via AutoTokenizer.
    In transformers 5.x, AutoTokenizer lets the class named in
    tokenizer_config.json rebuild the pre-tokenizer from its own defaults.
    Xenova/gpt-4o names GPT2Tokenizer, so it got GPT-2's regex, which splits
    combining marks off their base letters and roughly doubled GPT-4o's token
    count for Indic scripts while Latin-script numbers still looked fine.
    """
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(tokenizer_id)


# How each family marks a leading space in its surface pieces. This is the
# reliable signal, because it is a property of the encoding rather than of a
# class name.
BYTE_LEVEL_SPACE = "Ġ"      # 'Ġ' — GPT-2 byte-to-unicode mapping of 0x20
SENTENCEPIECE_SPACE = "▁"   # '▁' — SentencePiece word boundary


def tokenizer_family(tok) -> str:
    """Classify as 'byte_bpe', 'sentencepiece', or 'unknown'.

    This decides how the vocabulary extension has to work. Continuing an
    existing merge list is straightforward for byte-level BPE; a SentencePiece
    model with byte fallback needs a different approach. Confirm the answer
    before locking the base model, not after.

    Detection is behavioral rather than by class name, because class names lie.
    Qwen2Tokenizer, for instance, wraps a BPE model behind a `Sequence`
    pre-tokenizer with ByteLevel nested inside it, so inspecting the outer
    pre-tokenizer's name returns "Sequence" and tells you nothing. Encoding a
    space-prefixed token settles it: byte-level BPE renders the space as 'Ġ'
    and SentencePiece renders it as '▁'.
    """
    try:
        pieces = tok.tokenize(" a")
        joined = "".join(pieces)
        if BYTE_LEVEL_SPACE in joined:
            return "byte_bpe"
        if SENTENCEPIECE_SPACE in joined:
            return "sentencepiece"
    except Exception:
        pass

    # Fall back to introspection for anything the behavioral test could not
    # reach (a tokenizer that errors on tokenize(), say).
    cls = type(tok).__name__.lower()
    if "gpt2" in cls or "bytelevel" in cls:
        return "byte_bpe"
    if getattr(tok, "sp_model", None) is not None:
        return "sentencepiece"
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None:
        model_type = type(getattr(backend, "model", None)).__name__.lower()
        if "unigram" in model_type:
            return "sentencepiece"
        if "bpe" in model_type:
            return "byte_bpe"
    return "unknown"


def encode(tok, text: str) -> tuple[list[int], list[str]]:
    """Token ids and their surface pieces, with no special tokens.

    Special tokens are excluded because BOS/EOS add a constant per sentence,
    which inflates short sentences and makes cross-language premiums depend on
    sentence count rather than content.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    return ids, pieces


@lru_cache(maxsize=32)
def _subchar_vocab_mask(tokenizer_id: str) -> frozenset[int]:
    """Ids that do not decode to a complete character on their own.

    Computed once per tokenizer over the whole vocabulary rather than per
    sentence, which turns a per-token decode into a set lookup. This is the
    startup pause you see at the beginning of each tokenizer in the survey.

    The test is portable across tokenizer families: decode a single id in
    isolation and see whether it yields the Unicode replacement character. A
    SentencePiece byte-fallback piece such as <0xE1> and a byte-level BPE
    fragment of a multi-byte character both fail it, while an ordinary subword
    passes. A token is counted as sub-character only when it cannot stand
    alone, so a legitimate one-byte ASCII token is not flagged.
    """
    tok = load_tokenizer(tokenizer_id)
    bad: set[int] = set()
    # Every id the tokenizer actually has, not range(vocab_size): an extended
    # vocabulary appends ids after the special tokens, past vocab_size.
    for i in sorted(set(tok.get_vocab().values())):
        try:
            s = tok.decode([i])
        except Exception:
            bad.add(i)
            continue
        if REPLACEMENT_CHAR in s:
            bad.add(i)
    return frozenset(bad)


def subchar_flags(tokenizer_id: str, ids: list[int]) -> list[bool]:
    """Per-token flags: True where the token is a character fragment."""
    mask = _subchar_vocab_mask(tokenizer_id)
    return [i in mask for i in ids]


def roundtrip_ok(tok, text: str) -> bool:
    """Encoding then decoding returns the original text.

    Run this on target-language text before trusting any measurement of it. A
    tokenizer that silently drops a combining mark still produces a
    plausible-looking token count.
    """
    ids, _ = encode(tok, text)
    return tok.decode(ids) == text
