"""FLORES+ loading, cached, with NFC normalization applied once at the door.

FLORES+ is the maintained successor to FLORES-200, run by the Open Language
Data Initiative. Configs are named <iso639_3>_<Script>, e.g. eng_Latn, and the
dataset is gated, so a first run needs the terms accepted on the dataset page
and `huggingface-cli login`.

The property the whole survey depends on: row N of every language config is a
translation of the same source sentence. We therefore use the row index as
sent_id and never shuffle or filter per language, because doing so would break
the pairing silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..survey.metrics import normalize

FLORES_PLUS = "openlanguagedata/flores_plus"


def _cache_path(cache_dir: Path, lang: str, split: str) -> Path:
    return cache_dir / f"{lang}.{split}.json"


def load_sentences(
    lang: str,
    split: str = "devtest",
    cache_dir: str | Path = "data/flores",
) -> list[str]:
    """Return the sentences for one language, normalized and index-aligned.

    Cached to disk as JSON so a survey rerun does not need the network. Delete
    the cache directory to force a refresh.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(cache_dir, lang, split)
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    sentences = [normalize(s) for s in _download(lang, split)]
    cached.write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    return sentences


def _download(lang: str, split: str) -> list[str]:
    from datasets import load_dataset

    try:
        ds = load_dataset(FLORES_PLUS, lang, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"could not load {lang}/{split} from {FLORES_PLUS}: {exc}\n"
            "If this is an auth error: accept the terms on the dataset page, "
            "then run `huggingface-cli login`."
        ) from exc

    # FLORES+ uses 'text'; older releases used 'sentence'.
    col = "text" if "text" in ds.column_names else "sentence"
    return list(ds[col])


def check_alignment(by_lang: dict[str, list[str]]) -> None:
    """Assert every language has the same number of sentences.

    Cheap insurance. If one config is short, the paired premium silently drops
    the tail of the corpus for that language and the number still looks fine.
    """
    lengths = {lang: len(s) for lang, s in by_lang.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            "FLORES configs are not aligned; sentence counts differ: "
            f"{sorted(lengths.items(), key=lambda kv: kv[1])}"
        )
