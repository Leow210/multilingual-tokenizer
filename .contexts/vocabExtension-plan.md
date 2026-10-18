# Vocabulary extension

Outline only. No training or merge-learning code lives here yet.

**Target:** Telugu (`tel_Telu`) on `Qwen/Qwen3-0.6B`. Script-uncovered: Qwen median premium **7.04×**, **80.6%** of tokens are character fragments. Armenian (`hye_Armn`) is the backup (poor-merges); same pipeline later, different expected first merges.

**Recipe we are adopting:** in-place tokenizer expansion (Smith et al., 2026, [arXiv:2607.15232](https://arxiv.org/abs/2607.15232)). Continue Qwen’s byte-level BPE merge list, initialize new embedding rows as the mean of the source pieces they replace, adapt in two stages. Churchill and Skiena ([arXiv:2601.13328](https://arxiv.org/abs/2601.13328)) is the survey-shaped paper we are *not* copying for the repair: they add character tokens with derived embeddings, no retraining, no downstream eval. That gap is the point of this half.

Knobs for the first experiment: `configs/vocab.yaml`.

---

## Why this recipe, for this failure mode

Qwen is byte-level BPE (`tokenizer_family` in `survey/tokenization.py`). Existing merges can be held fixed and new ones appended. Every new token then has an exact decomposition into old tokens, which is what mean-init needs.

Telugu is spelled out at the byte level today. The first new merges should glue UTF-8 fragments into Telugu characters; later merges should form syllables and common subwords. That is the dramatic case. Armenian, later, already has characters in the vocab, so continuation would mostly learn better merges, not invent the script.

Do **not** train a fresh Telugu tokenizer and union the vocabs (Chinese-LLaMA style). Independent BPE gives new tokens with no canonical old-token decomposition, and mean-init becomes a guess.

---

## What we are not doing at Liquid scale

Liquid AI grew 65K → 128K and trained 600B embedding-only tokens plus 400B full CPT. We cannot. `check.py` already frames the course-sized budget: **+4,096 tokens**. Qwen3-0.6B is ~152k vocab; 4k new rows is a few million parameters, not a new model.

So we steal the *invariants* of the recipe, not the token counts:

| Keep | Scale down |
|---|---|
| Continue existing merges; do not reshuffle old ids | Add 1k / 2k / 4k tokens, not +64k |
| Mean-init new rows; copy old rows | Stage 1/2 on megabytes–gigabytes of Telugu, not 0.6T tokens |
| Freeze old rows in stage 1 | Skip mid-training / 128k context |
| Stage 2 mix must include the target language **and** English | Bits-per-byte + FLORES premium, not Global-MMLU |

Yamaguchi et al. (2026) report useful CVA from on the order of **0.01 GB** of continued pre-training. That is the regime we should try first.

---

## Pipeline

Four phases. Stop after A if the tokenizer-only numbers are not worth training.

```
corpus ──► continue BPE ──► mean-init embeddings ──► stage 1 (new rows)
                                      │                      │
                                      ▼                      ▼
                               FLORES premium           stage 2 (all)
                               subchar rate                  │
                               English control               ▼
                                                    bits/byte, net cost,
                                                    English generation check
```

### 0. Preconditions

Run `python check.py` and refuse to proceed unless:

- `FAMILY` is `byte_bpe`
- `tie_word_embeddings` is known (Qwen small models usually tie; if untied, init **both** input and lm-head with the same mean rule)
- FLORES `devtest` is never used to learn merges or tune

Confirm `roundtrip_ok` on a handful of Telugu sentences before trusting any new tokenizer.

### A. Corpus (no GPU)

Prefer **IndicCorp v2 Telugu** (731M tokens, news, LID-filtered). FineWeb-2 `tel_Telu` (14.4 GB UTF-8) is the volume fallback.

Mix in a small English slice (~10%) while learning merges so we do not spend the new budget on English words Qwen already has. NFC-normalize the same way `data/flores.py` does.

Hold out FLORES `devtest` entirely. If we need a merge-training diagnostic, use FLORES `dev` or a slice of IndicCorp, not the survey split.

### B. Continue BPE (no GPU) — implement this first

Load Qwen’s `tokenizer.json` via `PreTrainedTokenizerFast` / the `tokenizers` backend, same as the survey. Then:

1. Keep the pre-tokenizer, decoder, and special tokens exactly as Qwen shipped them.
2. Keep the existing merge list **in order**. Never re-rank old merges.
3. Run BPE pair counting on the Telugu (+ English) corpus **from the current vocabulary**, and **append** the next most frequent pairs until `n_new_tokens` is reached.
4. **Append-only ids.** Old surface forms keep their integer ids. New tokens get ids `vocab_size, vocab_size+1, …`. Special-token slots do not move. (Liquid reassigned ids and mapped by string. Append-only is safer on a 0.6B model we did not pretrain.)
5. Write a new `tokenizer.json` under `results/vocab/tel_Telu/<n>/`.

HuggingFace `BpeTrainer` retrains from scratch. Do not use it as-is. Continuation is a small loop over pair counts on top of the existing `BPE` model (or an equivalent toolkit). Tests should lock: old merges prefix the new merge list; encoding English `"Hello world"` is unchanged; a Telugu character that used to be three byte-tokens becomes one token after enough merges.

Sweep `n_new_tokens ∈ {1024, 2048, 4096}` before picking one. Report all three. BPE compression has diminishing returns and every new row costs decode bandwidth; `metrics.net_cost_ratio` is how we say whether 4k beat 2k.

**Tokenizer-only success (Telugu, FLORES `devtest`, vs stock Qwen):**

- median premium drops a lot (directionally toward GPT-4o’s ~1.9×, not necessarily all the way)
- `subchar_rate` collapses from ~0.81 toward 0 (this is the script-uncovered signature)
- English and Chinese premiums stay at ~1.0 / ~0.96
- `roundtrip_ok` still holds

If English fertility moves, the continuation is wrong. If Telugu premium drops but `subchar_rate` stays high, we learned the wrong merges.

Reuse `survey/run.py` + `aggregate.failure_mode_table` on the new tokenizer id. Do not duplicate metric code.

### C. Embedding init (needs the checkpoint, still no training)

For each new token `t` with source-piece decomposition `(t1, …, tN)` under the **old** tokenizer:

```
E_new[t] = mean(E_old[t1], …, E_old[tN])
```

Mean, not sum: keep row norm comparable. `N=1` is a copy. Nothing is random.

Old rows copy by id (append-only makes this a no-op slice). If embeddings are tied, this matrix is also the lm-head. If not, apply the same rule to both.

Resize `model.resize_token_embeddings(new_vocab_size)` then overwrite the new slice. Save under `checkpoints/` (gitignored).

Zero-shot after init will look worse on Telugu generation. That dip is expected; it is not a reason to skip stage 1.

### D. Two-stage adaptation (optional extra `[train]`, GPU)

Liquid’s negative findings are the constraints:

1. **Stage 1 trains only the new rows.** The body and the copied rows stay frozen. Training the full embedding against a frozen body broke generation (looping) while MMLU looked fine.
2. **Stage 2 unfreezes everything** on a **Telugu + English** mix. English-only CPT also broke generation while MCQ held. Do not evaluate this project on multiple-choice alone.

Course-scale stand-in, not a copy of 600B/400B:

- Stage 1: Telugu-heavy next-token prediction, short sequences, higher LR, new rows only.
- Stage 2: mix Telugu and English (and maybe a little Chinese as the “already cheap” control), lower LR, all params.
- Stop early. We care that Telugu bits-per-byte falls and English bits-per-byte does not jump, not that we match Qwen3-4B.

`train/` will own the loops later. This folder only needs to export: the expanded tokenizer, the id mapping, and the per-new-token source-piece lists that init uses.

---

## How we will score a run

Never report token reduction alone.

| Metric | Why |
|---|---|
| Paired median premium vs English (FLORES `devtest`) | Headline, same as the survey |
| Subchar rate | Did we actually cover the script? |
| Bytes / token | Secondary efficiency |
| Bits per byte, Telugu **and** English | Vocab-size-safe LM quality. Perplexity is illegal here: different tokenizers, different denominators |
| `net_cost_ratio(token_count_ratio, per_token_cost_ratio)` | 4k extra rows make every token a bit more expensive |
| A short English and Telugu generation check | Liquid: MCQ can look healthy while the model loops |

`per_token_cost_ratio` for a tied model is roughly `(vocab_new × hidden) / (vocab_old × hidden)` on the embedding/lm-head term, not on the whole 0.6B. Be explicit in the report about what the cost model includes.

Controls: English (must not regress) and Chinese (already cheaper than English; must not become expensive).

---

## Module split (when we implement)

Stay in the same style as `survey/`: pure functions tested without GPU, transformers quarantined.

| Module | Job | Needs GPU |
|---|---|---|
| `vocab/continue_bpe.py` | Append merges, write tokenizer | no |
| `vocab/decompose.py` | New token → old id sequence | no |
| `vocab/measure.py` | Thin wrapper around survey metrics on a new tokenizer | no |
| `vocab/init_embeddings.py` | Mean-init + save checkpoint | no* |
| `train/` (later) | Stage 1 / 2 | yes |

\*Loads the 0.6B weights; still no backward pass.

Tests that should exist before we trust a tokenizer:

- NFC in, NFC out
- Old merge list is a prefix of the new one
- English encoding of a fixed sentence is byte-identical
- A synthetic Telugu string that was 3 byte-tokens becomes 1 token after a forced merge
- Decomposition of every new token re-encodes to the same bytes under the old tokenizer
- Duplicate `(lang, sent_id, tokenizer_id)` still rejected when we re-run the survey

---

## Order of work

1. Implement **B only**, sweep 1k/2k/4k, write a small table next to `results/vocab/`.
2. If Telugu `subchar_rate` drops hard and English is flat, implement **C**.
3. Only then **D**, starting with stage 1 on a tiny slice to prove new rows move and old rows do not.
4. Armenian is a second B run with the same code, not a rewrite.

If B fails to cut `subchar_rate`, stop and inspect the pre-tokenizer / byte mapping before touching the model.
