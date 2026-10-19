r"""Continue Qwen's byte-level BPE merge list on a new language (plan step B).

    python -m tokpremium.vocab.continue_bpe --config configs/vocab.yaml

The corpus is pre-tokenized and segmented by the *current* tokenizer, and BPE
pair counting continues from that segmentation: the most frequent adjacent pair
is merged, counts are updated, repeat. That is what makes the new merges a true
continuation rather than a fresh tokenizer unioned onto the old one. Each new
token is exactly one merge of two tokens that already exist, so it has a
canonical decomposition into old ids, which mean-init needs.

Appending only at the end of the merge list keeps every existing merge at
higher priority, and a new token can only ever be built from pieces the old
merges produced. So text the new merges do not touch tokenizes exactly as
before. Merges whose result is pure ASCII are refused outright, which pins
English down even when the training corpus carries English.

Three pre-tokenizers are supported:

  shipped            Qwen's regex, unchanged.
  keep_marks_target  Combining marks of the target language's own script are
                     allowed inside a letter run. Qwen's letter class is \p{L}+,
                     which excludes the vowel signs and viramas Brahmic scripts
                     are written with, so a Tamil or Telugu word is cut into
                     several chunks before BPE ever runs and no merge can span
                     a vowel sign. Continuation under `shipped` therefore
                     plateaus on those scripts however many tokens it adds.
                     Scoped to one Unicode block, every other script tokenizes
                     exactly as shipped (see side_effects.py).
  keep_marks         Every combining mark (\p{M}) allowed inside a letter run,
                     as GPT-4o's o200k regex does. Reported for comparison
                     only: with no new merges at all it already changes how
                     Qwen tokenizes 39 FLORES+ languages, Thai (-43% tokens)
                     and Arabic among them, because Qwen's vocabulary holds
                     whole-word tokens its own regex can never reach.

Output per (language, pre-tokenizer): results/vocab/<lang>/<pretokenizer>/merges.json,
holding the ordered new merges up to the largest size in the sweep. Merges are
greedy, so the first n entries are exactly the n-token extension; the full
tokenizer.json (~11 MB) is rebuilt from it by build_tokenizer() instead of
being committed.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import yaml

from . import corpus, decompose

# Qwen's letter-run alternatives, and their replacements with `{marks}` (a regex
# character-class body) allowed inside a run and kept out of the punctuation run.
MARKS_PATCH = [
    (r"[^\r\n\p{L}\p{N}]?\p{L}+", r"[^\r\n\p{L}{marks}\p{N}]?[\p{L}{marks}]+"),
    (r" ?[^\s\p{L}\p{N}]+", r" ?[^\s\p{L}{marks}\p{N}]+"),
]

# Unicode block per ISO 15924 script code, for scoping keep_marks_target. A
# block is per script, not per language: Telugu's marks also serve any other
# language written in Telugu script.
SCRIPT_BLOCKS = {
    "Armn": (0x0530, 0x058F), "Hebr": (0x0590, 0x05FF), "Arab": (0x0600, 0x06FF),
    "Deva": (0x0900, 0x097F), "Beng": (0x0980, 0x09FF), "Guru": (0x0A00, 0x0A7F),
    "Gujr": (0x0A80, 0x0AFF), "Orya": (0x0B00, 0x0B7F), "Taml": (0x0B80, 0x0BFF),
    "Telu": (0x0C00, 0x0C7F), "Knda": (0x0C80, 0x0CFF), "Mlym": (0x0D00, 0x0D7F),
    "Sinh": (0x0D80, 0x0DFF), "Thai": (0x0E00, 0x0E7F), "Laoo": (0x0E80, 0x0EFF),
    "Tibt": (0x0F00, 0x0FFF), "Mymr": (0x1000, 0x109F), "Ethi": (0x1200, 0x137F),
    "Khmr": (0x1780, 0x17FF), "Olck": (0x1C50, 0x1C7F),
}


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: byte-level display character -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


BYTE_DECODER = _byte_decoder()


def piece_bytes(piece: str) -> bytes:
    """The raw bytes behind a byte-level BPE piece such as 'Ġhello'."""
    return bytes(BYTE_DECODER[ch] for ch in piece)


def is_ascii_piece(piece: str) -> bool:
    return all(b < 128 for b in piece_bytes(piece))


def keep_marks(spec: dict, marks: str = r"\p{M}") -> dict:
    """Copy of a tokenizer spec whose split regex keeps `marks` inside words.

    `marks` is a regex character-class body; the default is every combining
    mark. An empty class leaves the spec exactly as it was.
    """
    spec = copy.deepcopy(spec)
    if not marks:
        return spec
    pre = spec["pre_tokenizer"]
    split = next(p for p in pre.get("pretokenizers", [pre]) if p["type"] == "Split")
    regex = split["pattern"]["Regex"]
    for old, new in MARKS_PATCH:
        if regex.count(old) != 1:
            raise ValueError(f"split regex does not have the expected shape; missing {old!r}")
        regex = regex.replace(old, new.replace("{marks}", marks))
    split["pattern"]["Regex"] = regex
    return spec


def script_marks(script: str) -> str:
    r"""Character-class body listing every combining mark in a script's Unicode block.

    Listed code point by code point (as \x{....} ranges) rather than as an
    intersection like [\p{M}&&...], so the class is explicit in the regex and
    does not depend on the regex engine's set-operation support.
    """
    if script not in SCRIPT_BLOCKS:
        raise ValueError(f"no Unicode block for script {script!r}; add it to SCRIPT_BLOCKS")
    lo, hi = SCRIPT_BLOCKS[script]
    ranges: list[list[int]] = []
    for cp in range(lo, hi + 1):
        if unicodedata.category(chr(cp)).startswith("M"):
            if ranges and cp == ranges[-1][1] + 1:
                ranges[-1][1] = cp
            else:
                ranges.append([cp, cp])
    return "".join(rf"\x{{{a:04X}}}" if a == b else rf"\x{{{a:04X}}}-\x{{{b:04X}}}"
                   for a, b in ranges)


def keep_marks_target(spec: dict, lang: str) -> dict:
    """keep_marks scoped to the combining marks of `lang`'s script (e.g. tel_Telu -> Telu)."""
    return keep_marks(spec, script_marks(lang.split("_")[1]))


# name -> (base spec, target language) -> spec with that pre-tokenizer.
PRETOKENIZERS: dict[str, Callable[[dict, str], dict]] = {
    "shipped": lambda spec, lang: copy.deepcopy(spec),
    "keep_marks_target": keep_marks_target,
    "keep_marks": lambda spec, lang: keep_marks(spec),
}


def _tokenizer(spec: dict):
    from tokenizers import Tokenizer

    return Tokenizer.from_str(json.dumps(spec))


def count_chunks(lines: Iterable[str], spec: dict) -> Counter[str]:
    """Pre-tokenizer chunks (byte-level strings) with their corpus frequency."""
    tok = _tokenizer(spec)
    counts: Counter[str] = Counter()
    for line in lines:
        text = tok.normalizer.normalize_str(line) if tok.normalizer else line
        counts.update(chunk for chunk, _ in tok.pre_tokenizer.pre_tokenize_str(text))
    return counts


def segment(chunks: Counter[str], spec: dict) -> dict[tuple[str, ...], int]:
    """Each chunk as the current BPE model splits it, frequencies pooled per split."""
    tok = _tokenizer(spec)
    words: Counter[tuple[str, ...]] = Counter()
    for chunk, freq in chunks.items():
        words[tuple(t.value for t in tok.model.tokenize(chunk))] += freq
    return dict(words)


def apply_merges(seq: list[str], merges: Iterable[tuple[str, str]]) -> list[str]:
    """Apply merges in order, each left to right without overlap, as BPE does."""
    for a, b in merges:
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return seq


def learn_merges(
    words: dict[tuple[str, ...], int],
    vocab: Iterable[str],
    n_new: int,
    accept: Callable[[str], bool] = lambda token: True,
) -> list[tuple[str, str]]:
    """Greedy BPE continuation over an existing segmentation.

    Repeatedly merges the most frequent adjacent pair whose result is not
    already a token and passes `accept`, until n_new merges are learned or no
    pair is left. Ties break on the pair itself, so the output is deterministic.
    Counts are updated incrementally; only words containing the merged pair are
    touched.
    """
    seqs = [list(w) for w in words]
    freqs = list(words.values())
    vocab = set(vocab)
    counts: Counter[tuple[str, str]] = Counter()
    where: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for i, seq in enumerate(seqs):
        for pair in zip(seq, seq[1:], strict=False):
            counts[pair] += freqs[i]
            where[pair].add(i)

    heap = [(-c, pair) for pair, c in counts.items()]
    heapq.heapify(heap)
    refused: set[tuple[str, str]] = set()
    merges: list[tuple[str, str]] = []
    while heap and len(merges) < n_new:
        neg, pair = heapq.heappop(heap)
        # Lazy deletion: every count change pushes a fresh entry, so stale ones are skipped.
        if pair in refused or -neg != counts[pair] or counts[pair] <= 0:
            continue
        token = pair[0] + pair[1]
        if token in vocab or not accept(token):
            refused.add(pair)
            continue
        merges.append(pair)
        vocab.add(token)

        touched: set[tuple[str, str]] = set()
        for i in list(where[pair]):
            old = seqs[i]
            new = apply_merges(old, [pair])
            for p in zip(old, old[1:], strict=False):
                counts[p] -= freqs[i]
                where[p].discard(i)
                touched.add(p)
            for p in zip(new, new[1:], strict=False):
                counts[p] += freqs[i]
                where[p].add(i)
                touched.add(p)
            seqs[i] = new
        for p in touched:
            if counts[p] > 0 and p not in refused:
                heapq.heappush(heap, (-counts[p], p))
    return merges


def extend(spec: dict, merges: list[tuple[str, str]]) -> tuple[dict, list[int]]:
    """Copy of `spec` with `merges` appended. Returns the spec and the new token ids.

    Append-only: existing tokens keep their ids, existing merges keep their
    order, and special tokens do not move. New ids start after the highest id
    in use, special tokens included.

    Special tokens are pinned into the model vocab at their current ids first.
    On load, `tokenizers` re-numbers any added token missing from the model
    vocab to model_vocab_size + k, ignoring the id written in tokenizer.json.
    Qwen keeps its 26 special tokens outside the model vocab, so without the
    pin, appending N tokens slides every special token N slots up, onto the new
    tokens' ids: decode(new_id) returns '<|endoftext|>'. Encoding still counts
    correctly, which is what makes this easy to miss. A special token in the
    model vocab is never produced by a merge, so BPE output is unaffected.
    """
    spec = copy.deepcopy(spec)
    model = spec["model"]
    vocab, existing = model["vocab"], model["merges"]
    for special in spec.get("added_tokens", []):
        vocab.setdefault(special["content"], special["id"])
    string_merges = bool(existing) and isinstance(existing[0], str)
    next_id = max(vocab.values()) + 1
    new_ids = []
    for a, b in merges:
        if a + b in vocab:
            raise ValueError(f"merge {a!r} + {b!r} would duplicate an existing token")
        vocab[a + b] = next_id
        existing.append(f"{a} {b}" if string_merges else [a, b])
        new_ids.append(next_id)
        next_id += 1
    return spec, new_ids


def build_tokenizer(
    base_spec: dict, pretokenizer: str, lang: str, merges: list[tuple[str, str]], out_dir
):
    """Write a loadable Hugging Face tokenizer folder for one point of the sweep."""
    from transformers import PreTrainedTokenizerFast

    spec, _ = extend(PRETOKENIZERS[pretokenizer](base_spec, lang), merges)
    tok = PreTrainedTokenizerFast(tokenizer_object=_tokenizer(spec))
    tok.save_pretrained(str(out_dir))
    return out_dir


def load_base_spec(tokenizer_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(tokenizer_id, "tokenizer.json"), encoding="utf-8") as f:
        return json.load(f)


def merges_path(results_dir: str | Path, lang: str, pretokenizer: str) -> Path:
    return Path(results_dir) / lang / pretokenizer / "merges.json"


def load_merges(path: str | Path) -> list[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [tuple(t["merge"]) for t in data["new_tokens"]]


def run(cfg: dict, langs: list[str], force: bool = False) -> None:
    base_spec = load_base_spec(cfg["base"]["tokenizer_id"])
    n_max = max(cfg["n_new_tokens"])
    for lang in langs:
        source = cfg["corpus"]["sources"][lang]
        path = corpus.fetch(lang, source, cfg["corpus"]["max_chars"], cfg["corpus"]["cache_dir"])
        print(f"{lang}: corpus {path}", flush=True)
        for name in cfg["pretokenizers"]:
            out = merges_path(cfg["output"]["results_dir"], lang, name)
            if out.exists() and not force:
                print(f"  {name}: {out} exists, skipping (--force to relearn)", flush=True)
                continue
            spec = PRETOKENIZERS[name](base_spec, lang)
            print(f"  {name}: counting chunks ...", flush=True)
            chunks = count_chunks(corpus.read_lines(path), spec)
            words = segment(chunks, spec)
            print(f"  {name}: {len(chunks):,} distinct chunks; learning {n_max:,} merges ...",
                  flush=True)
            merges = learn_merges(words, spec["model"]["vocab"], n_max,
                                  accept=lambda t: not is_ascii_piece(t))
            _, new_ids = extend(spec, merges)
            source_ids = decompose.decomposer(spec)
            entries = [
                {
                    "id": new_id,
                    "token": a + b,
                    "text": piece_bytes(a + b).decode("utf-8", errors="replace"),
                    "merge": [a, b],
                    "source_ids": source_ids(a + b),
                }
                for new_id, (a, b) in zip(new_ids, merges, strict=True)
            ]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "base": cfg["base"]["tokenizer_id"],
                "lang": lang,
                "pretokenizer": name,
                "corpus": {**source, "chars_cap": cfg["corpus"]["max_chars"]},
                "new_tokens": entries,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {name}: wrote {len(entries):,} merges to {out}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/vocab.yaml")
    ap.add_argument("--lang", action="append", help="limit to these languages (repeatable)")
    ap.add_argument("--force", action="store_true", help="relearn merges that already exist")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg, args.lang or cfg["languages"], force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
