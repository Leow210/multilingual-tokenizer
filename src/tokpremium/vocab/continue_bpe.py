"""Continue a byte-level BPE merge list on a new corpus. No model, no GPU.

Plan: .contexts/vocabExtension-plan.md, phase B.

The invariants (each one is locked by a test):

* The existing merge list is a strict prefix of the new one. Old merges are
  never re-ranked, so text that was already well tokenized stays identical.
* Ids are append-only. New tokens get ids above every id already in the
  tokenizer, special tokens included.
* Every new token is the concatenation of exactly two tokens that existed when
  it was created, so it decomposes into old tokens (see decompose.py).

HuggingFace's BpeTrainer retrains from scratch, so this is a small loop over
pair counts on top of the existing model instead.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NewMerge:
    """One appended merge: `left + right -> token`, assigned `token_id`."""

    token_id: int
    token: str
    left: str
    right: str
    count: int  # corpus frequency of the pair at the moment it was merged


def _merge_pair(merge) -> tuple[str, str]:
    """tokenizer.json stores merges as "a b" (older) or ["a", "b"] (newer)."""
    if isinstance(merge, str):
        left, right = merge.split(" ", 1)
        return left, right
    return merge[0], merge[1]


def _apply(seq: list[str], left: str, right: str, merged: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def word_counts(texts: Iterable[str], backend) -> Counter:
    """Count pre-tokenized words (in byte-level alphabet) over `texts`.

    `backend` is the `tokenizers.Tokenizer`. We run its own normalizer and
    pre-tokenizer so counting matches exactly what encoding will see.
    """
    counts: Counter = Counter()
    norm = backend.normalizer
    pre = backend.pre_tokenizer
    for text in texts:
        if norm is not None:
            text = norm.normalize_str(text)
        for word, _ in pre.pre_tokenize_str(text):
            counts[word] += 1
    return counts


def learn_merges(
    words: Counter,
    backend,
    n_new: int,
    existing_vocab: set[str],
    first_id: int,
    min_count: int = 2,
) -> list[NewMerge]:
    """Greedy BPE from the current segmentation, `n_new` merges at most.

    Each unique word is first segmented with the *existing* model, then pairs
    are merged by frequency. Stops early when the best pair occurs fewer than
    `min_count` times (merging singletons memorizes the corpus).
    """
    seqs: list[list[str]] = []
    freqs: list[int] = []
    for word, c in words.items():
        seqs.append([t.value for t in backend.model.tokenize(word)])
        freqs.append(c)

    pair_count: Counter = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for w, seq in enumerate(seqs):
        for pair in zip(seq, seq[1:]):
            pair_count[pair] += freqs[w]
            where[pair].add(w)

    # Lazy max-heap: entries go stale when counts change; re-check on pop.
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)

    taken = set(existing_vocab)
    out: list[NewMerge] = []
    next_id = first_id
    while len(out) < n_new and heap:
        neg, pair = heapq.heappop(heap)
        cur = pair_count.get(pair, 0)
        if cur <= 0:
            continue
        if -neg != cur:
            heapq.heappush(heap, (-cur, pair))
            continue
        if cur < min_count:
            break
        left, right = pair
        merged = left + right
        if merged in taken:
            # Already a token reachable some other way; adding it again would
            # create a duplicate vocab key. Retire the pair and move on.
            pair_count[pair] = 0
            continue

        taken.add(merged)
        out.append(NewMerge(next_id, merged, left, right, cur))
        next_id += 1

        for w in list(where[pair]):
            seq = seqs[w]
            f = freqs[w]
            for p in zip(seq, seq[1:]):
                pair_count[p] -= f
            new_seq = _apply(seq, left, right, merged)
            seqs[w] = new_seq
            for p in zip(new_seq, new_seq[1:]):
                pair_count[p] += f
                where[p].add(w)
                heapq.heappush(heap, (-pair_count[p], p))
        pair_count[pair] = 0
        del where[pair]
    return out


def extend_tokenizer_json(spec: dict, new: list[NewMerge]) -> dict:
    """Return a copy of a tokenizer.json dict with `new` appended.

    Only `model.vocab` and `model.merges` change. Pre-tokenizer, normalizer,
    decoder, post-processor and added (special) tokens are carried over as-is.
    """
    out = json.loads(json.dumps(spec))
    model = out["model"]
    if model["type"] != "BPE":
        raise ValueError(f"expected a BPE model, got {model['type']!r}")
    as_lists = bool(model["merges"]) and not isinstance(model["merges"][0], str)
    # The `tokenizers` loader ignores the ids on added_tokens and re-assigns
    # any special token missing from model.vocab to len(vocab) + i. Qwen keeps
    # its specials out of model.vocab, so growing the vocab would silently
    # shift <|im_end|> etc. up by n_new and collide with real tokens. Listing
    # each special in model.vocab at its original id makes the loader keep it.
    taken_ids = set(model["vocab"].values())
    for a in out.get("added_tokens", []):
        if a["content"] not in model["vocab"] and a["id"] not in taken_ids:
            model["vocab"][a["content"]] = a["id"]
    for m in new:
        if m.token in model["vocab"]:
            raise ValueError(f"token already in vocab: {m.token!r}")
        model["vocab"][m.token] = m.token_id
        model["merges"].append([m.left, m.right] if as_lists else f"{m.left} {m.right}")
    return out


def next_free_id(spec: dict) -> int:
    """One past the highest id in use, counting special/added tokens."""
    ids = list(spec["model"]["vocab"].values())
    ids += [a["id"] for a in spec.get("added_tokens", [])]
    return max(ids) + 1


def continue_bpe(
    tokenizer_dir: str | Path,
    texts: Iterable[str],
    n_new: int,
    out_dir: str | Path,
    min_count: int = 2,
) -> list[NewMerge]:
    """Extend the tokenizer in `tokenizer_dir` and write the result to `out_dir`.

    `tokenizer_dir` must hold tokenizer.json (a local snapshot or a hub id
    resolved by the caller). Also writes `new_merges.json`, the record that
    decompose.py and the embedding init consume.
    """
    from tokenizers import Tokenizer

    tokenizer_dir, out_dir = Path(tokenizer_dir), Path(out_dir)
    spec = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    backend = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    words = word_counts(texts, backend)
    new = learn_merges(
        words,
        backend,
        n_new,
        existing_vocab=set(spec["model"]["vocab"]),
        first_id=next_free_id(spec),
        min_count=min_count,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    extended = extend_tokenizer_json(spec, new)
    (out_dir / "tokenizer.json").write_text(
        json.dumps(extended, ensure_ascii=False), encoding="utf-8"
    )
    # Config files the HF loader wants next to tokenizer.json.
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        src = tokenizer_dir / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())
    (out_dir / "new_merges.json").write_text(
        json.dumps(
            [m.__dict__ for m in new],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return new
