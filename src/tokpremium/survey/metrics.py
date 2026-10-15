"""Pure measurement functions for the token-cost survey.

Everything here works on plain Python data (strings, lists, ints) with no
dependency on transformers or torch. That is deliberate: these are the
functions the whole survey's numbers rest on, so they must be testable in
under a second by anyone who just cloned the repo.

Anything needing a live tokenizer lives in `tokenization.py` instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

# SentencePiece byte-fallback pieces look like <0xE1>. Byte-level BPE has no
# equivalent surface form, so it is detected by decoding instead; see
# tokenization.subchar_flags().
BYTE_FALLBACK_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

# The Unicode replacement character. A token that decodes to this on its own is
# a fragment of a character rather than a character, which is the
# tokenizer-agnostic signal that a script is being spelled out byte by byte.
REPLACEMENT_CHAR = "�"


def normalize(text: str) -> str:
    """NFC-normalize text before any measurement.

    Without this, the same visible string can carry a different code point
    count depending on where the corpus came from, which silently corrupts
    every per-character metric. Applied once, at load time, to every language.
    """
    return unicodedata.normalize("NFC", text)


def count_chars(text: str) -> int:
    """Unicode code points."""
    return len(text)


def count_bytes(text: str) -> int:
    """UTF-8 bytes.

    Prefer this as the denominator for cross-script comparison: a code point is
    not a comparable unit of information between, say, Han script and the Latin
    alphabet.
    """
    return len(text.encode("utf-8"))


def is_byte_fallback_piece(piece: str) -> bool:
    """True for a SentencePiece byte-fallback piece such as '<0xE1>'."""
    return bool(BYTE_FALLBACK_PIECE_RE.match(piece))


def count_byte_fallback_pieces(pieces: list[str]) -> int:
    """How many pieces are SentencePiece byte fallbacks.

    Exact and cheap for SentencePiece tokenizers, always zero for byte-level
    BPE. Use count_subchar_tokens() for the portable version.
    """
    return sum(1 for p in pieces if is_byte_fallback_piece(p))


def count_subchar_tokens(flags: list[bool]) -> int:
    """How many tokens failed to decode to a complete character on their own.

    `flags` comes from tokenization.subchar_flags(). This is the portable
    measure of "the tokenizer is spelling this script out instead of
    tokenizing it", and it works for both tokenizer families.
    """
    return sum(1 for f in flags if f)


@dataclass(frozen=True)
class SentenceRow:
    """One measurement: one sentence, one language, one tokenizer.

    The survey stores these, NOT per-language averages. Averaging at write
    time throws away the sentence-level pairing that makes FLORES worth using,
    and a paired premium cannot be recovered from two means. Every aggregate we
    report is a groupby over this table.
    """

    lang: str
    sent_id: int
    tokenizer_id: str
    n_tokens: int
    n_chars: int
    n_bytes: int
    n_subchar_tokens: int
    n_fallback_pieces: int

    def as_dict(self) -> dict:
        return asdict(self)


def sentence_row(
    *,
    lang: str,
    sent_id: int,
    tokenizer_id: str,
    text: str,
    pieces: list[str],
    subchar_flags: list[bool] | None = None,
) -> SentenceRow:
    """Build one survey row. `text` must already be normalized."""
    if subchar_flags is None:
        subchar_flags = []
    return SentenceRow(
        lang=lang,
        sent_id=sent_id,
        tokenizer_id=tokenizer_id,
        n_tokens=len(pieces),
        n_chars=count_chars(text),
        n_bytes=count_bytes(text),
        n_subchar_tokens=count_subchar_tokens(subchar_flags),
        n_fallback_pieces=count_byte_fallback_pieces(pieces),
    )


def net_cost_ratio(token_count_ratio: float, per_token_cost_ratio: float) -> float:
    """Net compute ratio after accounting for a larger vocabulary.

    A bigger vocabulary shrinks the token count but enlarges the embedding and
    output projection matrices, which raises the cost of processing each
    individual token. Reporting the token reduction alone overstates the win.

    Args:
        token_count_ratio: tokens_after / tokens_before. Below 1.0 is better,
            so 0.4 means the text got 2.5x shorter.
        per_token_cost_ratio: cost_per_token_after / cost_per_token_before.
            Above 1.0, since the vocabulary only grows.

    Returns:
        Net cost ratio. Below 1.0 is a real saving; above 1.0 means the larger
        vocabulary ate the gain.

    >>> round(net_cost_ratio(0.40, 1.10), 4)
    0.44
    """
    if token_count_ratio <= 0:
        raise ValueError("token_count_ratio must be positive")
    if per_token_cost_ratio <= 0:
        raise ValueError("per_token_cost_ratio must be positive")
    return token_count_ratio * per_token_cost_ratio
