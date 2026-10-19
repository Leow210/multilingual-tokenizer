"""New token -> the old token ids it replaces. Input to mean-init (plan step C).

Continuation only ever merges two tokens that already exist, so every new token
is a fixed string of bytes that the *old* BPE model can still split. That split
is the source-piece list: E_new[t] = mean(E_old[i] for i in source_ids(t)).

Splitting with the old model rather than recursing through the two merged
halves matters once merges stack: a new token built from another new token
must still decompose into ids the checkpoint actually has rows for.
"""

from __future__ import annotations

import json
from collections.abc import Callable


def decomposer(old_spec: dict) -> Callable[[str], list[int]]:
    """A function mapping a byte-level token to the old ids it is made of.

    Built once per tokenizer: parsing a 150k-entry tokenizer.json per token
    would dominate the run.
    """
    from tokenizers import Tokenizer

    model = Tokenizer.from_str(json.dumps(old_spec)).model
    return lambda token: [t.id for t in model.tokenize(token)]
