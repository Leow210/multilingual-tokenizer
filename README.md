# Multilingual Tokenizer Premium

CSCI 544, Group 9.

Some languages cost far more to run through a language model than others. The
model breaks text into tokens, and the same sentence can take several times as
many tokens in one language as in another, which means several times the
compute, several times the API bill, and a fraction of the usable context
window. This repo measures that gap across 20 languages and then tries to close
it for one of them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # 40 tests, under a second
huggingface-cli login              # FLORES+ is a gated dataset
python check.py                    # pre-flight: tokenizers, base model, language codes
```

Activate the venv in every new terminal, or none of the commands below will
find their packages.

## Running the survey

```bash
python -m tokpremium.survey.run --limit 20    # smoke test, under a minute
python -m tokpremium.survey.run               # the real run, a few minutes
python -m tokpremium.survey.figures
```

No GPU needed for any of this. Outputs land in `results/`.

The one slow part is a pause at the start of each tokenizer while it scans the
vocabulary for character fragments. It does that once per tokenizer rather than
once per sentence, which is why the survey loops tokenizers on the outside.

## What is here

```
check.py                        pre-flight checks; run this first
configs/survey.yaml             which 20 languages, which 4 tokenizers
src/tokpremium/
  data/flores.py                FLORES+ loading, cached, NFC-normalized
  survey/metrics.py             pure measurement functions, fully tested
  survey/tokenization.py        the only module that loads a real tokenizer
  survey/run.py                 CLI: produces results/survey.parquet
  survey/aggregate.py           parquet to the numbers in the report
  survey/figures.py             parquet to the plots
tests/                          40 tests, no GPU, under a second
results/                        committed: the parquet, tables, figures
data/                           gitignored, re-downloadable
```

The vocabulary-extension and training code (`vocab/`, `train/`, `eval/`) comes
later in the week. The survey half is what the presentation needs.

## Four decisions baked in

**Sentence-level rows, never averages.** The survey writes one row per
(language, sentence, tokenizer). FLORES sentence N is a translation of the same
source sentence in every language, so comparing N to N isolates the tokenizer
from the content. Averaging at write time destroys that pairing permanently,
and a paired premium cannot be recovered from two means. There is a test
demonstrating a case where the paired median and the corpus-level ratio
disagree by a factor of five.

**Bytes, not characters, as the denominator.** One Han character carries more
information than one Latin letter, so characters-per-token flatters some
scripts and penalizes others.

**Bits per byte, never perplexity** (for the later training half). Two models
with different vocabularies normalize perplexity over different token counts,
so the numbers cannot be compared. Comparing before and after a vocabulary
change is the whole project.

**Token reduction reported net of the larger vocabulary.** A bigger vocabulary
shrinks the token count but enlarges the embedding and output matrices, raising
the per-token cost. `metrics.net_cost_ratio()` does the netting.

## Picking the target language

Open `results/figures/failure_modes.csv` after the survey. Choose on three
things together:

- Premium under **Qwen** of at least 3.0. Qwen is the model we extend, so
  Qwen's column is the one we can act on. The other tokenizers tell the story.
- At least 200MB of audited clean text, 500MB comfortable.
- Prefer a high character-fragment rate, since that is the dramatic case.

Rule out Shan regardless of its premium. It will likely top the chart and has
nowhere near enough monolingual text to train on.

## Related work

- *In-Place Tokenizer Expansion for Pre-trained LLMs*, arXiv:2607.15232.
  The recipe we adopt: continue the existing merge list, initialize new rows as
  the mean of the pieces they replace, adapt in two stages. Read their negative
  results section before writing any vocabulary code.
- *Reducing Tokenization Premiums for Low-Resource Languages*, arXiv:2601.13328.
  Covers much of the cross-language survey. Their evidence that the change is
  harmless is that internal states look similar afterwards, with no retraining
  and no downstream evaluation, which is the gap this project fills.
