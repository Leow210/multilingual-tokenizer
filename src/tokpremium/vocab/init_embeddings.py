"""Mean-init the new embedding rows and save a checkpoint. Phase C, no training.

    E_new[t] = mean(E_old[t1], ..., E_old[tN])

where (t1..tN) is the token's decomposition under the ORIGINAL tokenizer
(decompose.py). Mean, not sum, so row norms stay comparable; N=1 is a copy.
Old rows are untouched. If the model ties input and output embeddings, the one
matrix is the lm-head too; if it does not, the same rule is applied to both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def mean_init_rows(weight: torch.Tensor, decomp: dict[int, list[int]], old_rows: int) -> None:
    """Overwrite rows `decomp.keys()` of `weight` in place with source means.

    `old_rows` guards the invariant: sources must all be original ids. Computed
    in float32 and cast back, since bf16 accumulation is lossy for long means.
    """
    with torch.no_grad():
        for new_id, srcs in decomp.items():
            if not srcs or max(srcs) >= old_rows or new_id < old_rows:
                raise ValueError(f"bad decomposition for id {new_id}: {srcs}")
            rows = weight[torch.tensor(srcs)].float()
            weight[new_id] = rows.mean(dim=0).to(weight.dtype)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--vocab-dir", required=True, help="results/vocab/<lang>/<n>")
    ap.add_argument("--out", required=True, help="checkpoint dir (gitignored)")
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    vdir = Path(args.vocab_dir)
    decomp = {int(k): v for k, v in json.loads((vdir / "decomposition.json").read_text()).items()}
    tok = PreTrainedTokenizerFast.from_pretrained(vdir)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    old_rows = model.get_input_embeddings().weight.shape[0]
    # Ids in the stock tokenizer's range (incl. padding rows) are original.
    first_new = min(decomp)
    needed = max(max(decomp) + 1, len(tok))
    if needed > old_rows:
        model.resize_token_embeddings(needed)
    inp = model.get_input_embeddings().weight
    # Sources are always <= the stock tokenizer's top id; rows below first_new are "old".
    mean_init_rows(inp, decomp, old_rows=first_new)
    tied = model.config.tie_word_embeddings
    if not tied:
        mean_init_rows(model.get_output_embeddings().weight, decomp, old_rows=first_new)

    out = Path(args.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out}: rows {old_rows} -> {inp.shape[0]}, tied={tied}, "
          f"new={len(decomp)}, hidden={inp.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
