"""Monolingual training text for merge learning, streamed once and cached on disk.

Only a prefix of each source is read. Merges are ranked by pair frequency, and a
few tens of millions of characters already settle the first few thousand pairs,
so streaming the whole of IndicCorp or FineWeb-2 buys nothing at this stage.
The cache lives under data/ (gitignored) and is NFC-normalized the same way
data/flores.py normalizes FLORES, so training and evaluation text agree.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..survey.metrics import normalize


def corpus_path(lang: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{lang}.txt"


def fetch(lang: str, source: dict, max_chars: int, cache_dir: str | Path) -> Path:
    """Stream up to max_chars of `source` into the cache, one paragraph per line.

    `source` is a load_dataset spec from configs/vocab.yaml: `repo`, plus
    either `config` (FineWeb-2 style) or `data_files` (a raw text file such as
    IndicCorp's). Delete the cached file to force a refresh.
    """
    path = corpus_path(lang, cache_dir)
    if path.exists():
        return path

    from datasets import load_dataset

    ds = load_dataset(
        source["repo"],
        source.get("config"),
        data_files=source.get("data_files"),
        split="train",
        streaming=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name first so an interrupted download is not mistaken for a cache hit.
    tmp = path.with_suffix(".partial")
    total = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ds:
            for line in row["text"].split("\n"):
                line = normalize(line.strip())
                if line:
                    f.write(line + "\n")
                    total += len(line)
            if total >= max_chars:
                break
    tmp.replace(path)
    return path


def read_lines(path: str | Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")
