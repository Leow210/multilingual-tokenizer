"""
confirming to make sure that tokenizers load in correctly and that languages are loaded in 
"""

from __future__ import annotations

TOKENIZERS = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "gpt-4o": "Xenova/gpt-4o",
}



LANGUAGES = [
    "eng_Latn",                                      #added additional languages
    "spa_Latn", "fra_Latn", "deu_Latn",             
    "cmn_Hans", "hin_Deva", "ben_Beng", "tam_Taml",  # 12 chosen from study
    "tel_Telu", "urd_Arab", "vie_Latn", "heb_Hebr",
    "hye_Armn", "ydd_Hebr", "amh_Ethi", "shn_Mymr",
]


def check_tokenizers() -> dict[str, str]:
    "Load in tokenizers"
    from transformers import AutoTokenizer

    print("=== tokenizers ===")
    ok = {}
    for label, repo in TOKENIZERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            print(f"  OK    {label:14s} vocab={tok.vocab_size:>7,}  {type(tok).__name__}")
            ok[label] = repo
        except Exception as exc:
            print(f"  FAIL  {label:14s} {type(exc).__name__}: {str(exc)[:70]}")
    return ok


def check_base_model() -> None:
  
    from transformers import AutoConfig, AutoTokenizer

    from tokpremium.survey.tokenization import tokenizer_family

    print("base model: Qwen/Qwen3-0.6B")
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as exc:
        print(f"  could not load: {exc}")
        return

    backend = getattr(tok, "backend_tokenizer", None)
    pre = type(getattr(backend, "pre_tokenizer", None)).__name__ if backend else "?"
    model = type(getattr(backend, "model", None)).__name__ if backend else "?"
    family = tokenizer_family(tok)

    print(f"  tokenizer class : {type(tok).__name__}")
    print(f"  backend model   : {model}")
    print(f"  pre-tokenizer   : {pre}")
    print(f"  FAMILY          : {family}")

    # The surface form of a leading space is the evidence behind that verdict.
    print(f"  tokenize(' a')  : {tok.tokenize(' a')}")
    print(f"  tokenize('中')   : {tok.tokenize('中')}")

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    hidden = getattr(cfg, "hidden_size", None)
    vocab = getattr(cfg, "vocab_size", None) or tok.vocab_size
    print(f"  tie_word_embeddings : {getattr(cfg, 'tie_word_embeddings', None)}")
    print(f"  hidden_size         : {hidden}")
    if hidden and vocab:
        emb = hidden * vocab
        print(f"  embedding matrix    : {vocab:,} x {hidden:,} = {emb / 1e6:.1f}M params")
        print(f"  adding 4,096 tokens : +{4096 * hidden / 1e6:.1f}M params")

    if family != "byte_bpe":
        print("\n  >>> NOT byte-level BPE. Tell the group before writing any")
        print("  >>> vocabulary-extension code; the merge-continuation plan changes.")

#check to see if languages exist
def check_languages() -> None:
    from datasets import get_dataset_config_names

    print("\n=== FLORES+ language codes ===")
    try:
        names = set(get_dataset_config_names("openlanguagedata/flores_plus"))
    except Exception as exc:
        print(f"  could not list configs: {type(exc).__name__}: {str(exc)[:90]}")
        print("  If this is an auth error: accept the terms on the dataset page,")
        print("  then run `huggingface-cli login`.")
        return

    print(f"  {len(names)} configs available")
    missing = [lang for lang in LANGUAGES if lang not in names]
    if missing:
        print(f"  MISSING: {missing}")
        print("  Drop or rename these in configs/survey.yaml before running.")
        # Show near neighbours so a renamed code is easy to spot.
        for lang in missing:
            stem = lang.split("_")[0][:2]
            near = sorted(n for n in names if n.startswith(stem))
            print(f"    codes starting '{stem}': {near[:12]}")
    else:
        print(f"  all {len(LANGUAGES)} requested codes exist")


def main() -> None:
    ok = check_tokenizers()
    check_base_model()
    check_languages()

    print("\n=== summary ===")
    print(f"  {len(ok)}/{len(TOKENIZERS)} tokenizers usable: {sorted(ok)}")
    if len(ok) < len(TOKENIZERS):
        missing = sorted(set(TOKENIZERS) - set(ok))
        print(f"  not usable yet: {missing}")


if __name__ == "__main__":
    main()
