"""New token -> sequence of ORIGINAL token ids. Input to mean-init (phase C).

A new merge's parts may themselves be new tokens, so we expand recursively
until every leaf is a token that existed before extension.
"""

from __future__ import annotations

from .continue_bpe import NewMerge


def decompose(new: list[NewMerge], old_vocab: dict[str, int]) -> dict[int, list[int]]:
    """Map each new token id to the old ids it is made of, in order."""
    by_token = {m.token: m for m in new}
    memo: dict[str, list[int]] = {}

    def expand(tok: str) -> list[int]:
        if tok in memo:
            return memo[tok]
        if tok in old_vocab:
            memo[tok] = [old_vocab[tok]]
        else:
            m = by_token[tok]
            memo[tok] = expand(m.left) + expand(m.right)
        return memo[tok]

    return {m.token_id: expand(m.token) for m in new}
