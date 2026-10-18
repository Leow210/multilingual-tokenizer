# Target language candidates

Decision notes for the vocabulary-extension half. Qwen/Qwen3-0.6B is the model we extend, so **Qwen’s column decides**. Llama and GPT-4o are only for “is this structural?”.

## Criteria (from the original README)

1. **Qwen median premium ≥ 3.0** vs English on FLORES+ `devtest` (1,012 paired sentences).
2. **Clean text:** ≥ 200 MB audited, 500 MB comfortable. Below we treat FineWeb-2 train UTF-8 size as a public lower bound, and MADLAD-400-clean / IndicCorp v2 as the more audited sources.
3. **Prefer a high character-fragment rate** (script-uncovered, subchar ≥ 0.10) as the dramatic case.
4. **Also keep a poor-merges partner** (premium high, subchar near 0): characters are already in the vocab, merges are just bad. Repair may behave differently.
5. **Rule out Shan** regardless of premium: not enough monolingual text.

Failure mode is copied from `results/figures/failure_modes.csv`: `script-uncovered` if Qwen subchar rate ≥ 0.10, else `poor-merges`.

## Recommendation

If we train **one** language: **Telugu**.

If we train **one of each failure mode** (what the survey was designed for): **Telugu + Armenian**.

Do not pick Telugu and Tamil as the pair. They fail the same way.

---

## Ranked list (best → worst)

### 1. Telugu (`tel_Telu`) — best primary

| | |
|---|---|
| Failure mode | script-uncovered |
| Qwen premium | **7.04×** median (mean 7.14, p90 9.15) |
| Qwen subchar rate | **0.806** |
| Qwen bytes/token | 1.83 |
| Llama / GPT-4o premium | 8.28× / 1.93× |
| Survey tier / script | mid / Telugu |

**Data:** passes 500 MB easily.

- FineWeb-2 train: 891M words, 2.0M docs, **14.4 GB** UTF-8
- MADLAD-400-clean: **5.1B characters**
- IndicCorp v2: **731M tokens** (news-heavy, LID + offensive-word filtered)
- CulturaX: **1.57B tokens**

**Why first.** Highest *viable* Qwen premium and the highest fragment rate in the survey. GPT-4o already tokenizes Telugu at ~1.9×, which is useful: the language is not inherently expensive, Qwen just never spent vocabulary on it. IndicCorp + FineWeb + FLORES + Indic evals (IndicXTREME / Belebele) exist.

**Risk.** Same Dravidian/Brahmic cluster as Tamil; if we only get one uncovered language, this is it.

---

### 2. Tamil (`tam_Taml`) — best backup uncovered

| | |
|---|---|
| Failure mode | script-uncovered |
| Qwen premium | **6.20×** (mean 6.23, p90 7.81) |
| Qwen subchar rate | **0.632** |
| Qwen bytes/token | 2.50 |
| Llama / GPT-4o premium | 7.69× / 1.96× |
| Survey tier / script | mid / Tamil |

**Data:** passes 500 MB easily.

- FineWeb-2 train: 1.94B words, 5.5M docs, **37.0 GB** UTF-8 (more than Telugu)
- MADLAD-400-clean: **10.6B characters**
- IndicCorp v2: **476M tokens**
- CulturaX: **4.38B tokens**

**Why second.** Same shape as Telugu, slightly less dramatic, more web text. Strong if Telugu data quality disappoints on inspection. Do not pick *both* Te and Ta unless the point is a same-family replication.

---

### 3. Armenian (`hye_Armn`) — best poor-merges partner

| | |
|---|---|
| Failure mode | **poor-merges** |
| Qwen premium | **5.25×** (mean 5.35, p90 6.83) |
| Qwen subchar rate | **0.0018** |
| Qwen bytes/token | 1.85 |
| Llama / GPT-4o premium | 8.58× (Llama *uncovered*) / 1.77× |
| Survey tier / script | low / Armenian |

**Data:** passes 500 MB.

- FineWeb-2 train: 634M words, 1.8M docs, **7.2 GB** UTF-8
- MADLAD-400-clean: **1.5B characters** (audit note: “ok”)
- CulturaX: **2.40B tokens**, 3.0M docs
- No IndicCorp (not Indic)

**Why third, and the contrast pick.** This is the cleanest “characters are in the vocab, merges are bad” row under Qwen. A 5.25× premium with a 0.18% fragment rate is the opposite of Telugu. Unique script, so new merges will not secretly ride on Hindi/Arabic. Llama fails Armenian as script-uncovered, which is a nice report footnote but does not change the Qwen decision.

**Risk.** Fewer Indic-style downstream suites; FLORES+ and Belebele still cover it. Smaller web crawl than the Indic set, still far above 500 MB.

---

### 4. Bengali (`ben_Beng`)

| | |
|---|---|
| Failure mode | script-uncovered |
| Qwen premium | **5.04×** (mean 5.10, p90 6.37) |
| Qwen subchar rate | **0.676** |
| Qwen bytes/token | 2.51 |
| Llama / GPT-4o premium | 5.83× / 1.71× |
| Survey tier / script | mid / Bengali |

**Data:** huge.

- FineWeb-2 train: 6.15B words, 15.2M docs, **87.0 GB** UTF-8
- MADLAD-400-clean: **4.3B characters**
- IndicCorp v2: **926M tokens**
- CulturaX: **9.57B tokens**

**Why not higher.** Same failure mode as Telugu/Tamil, lower premium. Data is not the bottleneck. Fine as a third uncovered language, not as the pair partner.

---

### 5. Hindi (`hin_Deva`)

| | |
|---|---|
| Failure mode | script-uncovered |
| Qwen premium | **4.44×** (mean 4.49, p90 5.60) |
| Qwen subchar rate | **0.514** |
| Qwen bytes/token | 2.77 |
| Llama / GPT-4o premium | 2.52× (Llama already poor-merges) / 1.57× |
| Survey tier / script | mid / Devanagari |

**Data:** largest of the Indic set.

- FineWeb-2 train: 11.2B words, 22.1M docs, **121 GB** UTF-8
- MADLAD-400-clean: **20.1B characters**
- IndicCorp v2: **6.11B tokens**
- CulturaX: **16.8B tokens**

**Why lower.** Premium is real but the least dramatic uncovered case that still clears 3.0. Llama already tokenizes Hindi reasonably (2.5×). High chance Qwen already spent some vocab here; extension may look like a smaller win. Best as English-adjacent control data, not as the target.

---

### 6. Urdu (`urd_Arab`) — second poor-merges option

| | |
|---|---|
| Failure mode | **poor-merges** |
| Qwen premium | **3.19×** (mean 3.20, p90 4.00) |
| Qwen subchar rate | **0.0020** |
| Qwen bytes/token | 2.66 |
| Llama / GPT-4o premium | 3.07× (Llama *uncovered*) / 1.65× |
| Survey tier / script | mid / Arabic |

**Data:** passes 500 MB.

- FineWeb-2 train: 2.73B words, 4.8M docs, **19.9 GB** UTF-8
- MADLAD-400-clean: **2.7B characters**
- IndicCorp v2: **667M tokens**
- CulturaX: **2.70B tokens**

**Why below Armenian.** Same failure-mode shape, much weaker premium (barely over 3.0). Arabic script likely shares vocabulary with Arabic/Persian, so “new merges” may be entangled with languages Qwen already serves. Prefer Armenian unless we specifically want an Indic poor-merges contrast to Telugu.

---

### 7. Amharic (`amh_Ethi`)

| | |
|---|---|
| Failure mode | script-uncovered |
| Qwen premium | **4.07×** (mean 4.11, p90 5.14) |
| Qwen subchar rate | **0.596** |
| Qwen bytes/token | 2.02 |
| Llama / GPT-4o premium | 7.62× / **5.78×** (still bad on GPT-4o) |
| Survey tier / script | low / Ethiopic |

**Data:** passes 500 MB, smallest among the viable uncovered set.

- FineWeb-2 train: 240M words, 428k docs, **2.49 GB** UTF-8 (848 MB parquet)
- MADLAD-400-clean: **509M characters** (audit: “ok”)
- CulturaX: **358M tokens**, 243k docs
- No IndicCorp

**Why seventh.** Unique script and still expensive even under GPT-4o, which is scientifically interesting. Data and downstream evals are thinner than Indic. If we already have Telugu, Amharic is a second uncovered language only if we want a non-Indic replication.

---

### 8. Yiddish (`ydd_Hebr`) — fails premium gate

| | |
|---|---|
| Failure mode | poor-merges |
| Qwen premium | **2.77×** (mean 2.78, p90 3.35) — **below 3.0** |
| Qwen subchar rate | **0.000** |
| Qwen bytes/token | 3.39 |
| Llama / GPT-4o premium | 5.57× (Llama uncovered) / 2.00× |
| Survey tier / script | low / Hebrew |

**Data:** meets 200 MB, short of comfortable 500 MB.

- FineWeb-2 train: 94M words, 135k docs, **893 MB** UTF-8 (259 MB parquet); wiki_ratio 0.058
- MADLAD-400-clean: **353M characters**
- CulturaX: **218M tokens**, 141k docs

**Why listed.** Near-miss on the premium cutoff; useful as a Hebrew-script control next to Hebrew (1.47×) in the survey writeup, not as a training target.

---

### 9. Shan (`shn_Mymr`) — ruled out (data)

| | |
|---|---|
| Failure mode | script-uncovered |
| Qwen premium | **10.5×** (mean 10.7, p90 14.4) — worst in the survey |
| Qwen subchar rate | **0.765** |
| Qwen bytes/token | 1.81 |
| Llama / GPT-4o premium | 14.8× / 7.92× |
| Survey tier / script | low / Myanmar |

**Data:** fails 200 MB.

- FineWeb-2 train: 4.0M words, 19k docs, **120 MB** UTF-8 (23 MB parquet); wiki_ratio 0.49
- MADLAD-400-clean: **5.7M characters**, 788 clean docs
- Not in CulturaX or IndicCorp

Excellent survey row (tops every tokenizer). Not a viable extension target: there is nowhere near enough text to learn merges or adapt embeddings.

---

## Scoreboard (Qwen, FLORES+ `devtest`)

| Rank | Language | Premium | Subchar | Mode | FineWeb-2 UTF-8 | ≥200 MB | ≥500 MB | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Telugu | 7.04 | 0.81 | uncovered | 14.4 GB | yes | yes | **pick this** |
| 2 | Tamil | 6.20 | 0.63 | uncovered | 37.0 GB | yes | yes | same cluster as Te |
| 3 | Armenian | 5.25 | 0.002 | poor-merges | 7.2 GB | yes | yes | **pick with Te** |
| 4 | Bengali | 5.04 | 0.68 | uncovered | 87.0 GB | yes | yes | extra uncovered |
| 5 | Hindi | 4.44 | 0.51 | uncovered | 121 GB | yes | yes | least dramatic |
| 6 | Urdu | 3.19 | 0.002 | poor-merges | 19.9 GB | yes | yes | weaker than hy |
| 7 | Amharic | 4.07 | 0.60 | uncovered | 2.5 GB | yes | yes | viable, thinner evals |
| 8 | Yiddish | 2.77 | 0.00 | poor-merges | 893 MB | yes | borderline | fails premium ≥ 3 |
| 9 | Shan | 10.5 | 0.77 | uncovered | 120 MB | **no** | no | ruled out |

Sources: `results/figures/failure_modes.csv` and `by_language.csv` (this repo); FineWeb-2 language table; MADLAD-400 appendix Table 8 (clean character counts); IndicCorp v2 Table 3; CulturaX language table.

## Suggested pair for the next step

- **Telugu** — learn new pieces for an uncovered Brahmic script (high fragment rate).
- **Armenian** — continue merges over characters that Qwen already has.

Training text: start with IndicCorp v2 Telugu (cleaner news) plus a FineWeb-2 / MADLAD-400-clean slice; Armenian from FineWeb-2 or MADLAD-400-clean. Keep FLORES `dev` for any tuning; `devtest` stays the premium test set.
