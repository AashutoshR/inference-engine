"""WikiText-2 perplexity, to check that quantization keeps quality (Phase 4 gate).

    python bench/perplexity.py --model models/Llama-3.2-1B-Instruct
    python bench/perplexity.py --model models/Llama-3.2-1B-Instruct --quant int4

The test split is joined with blank lines, tokenized once, and cut into non-overlapping
windows of --seq-len tokens. Perplexity is exp(mean next-token NLL) over every
predicted token. The LM head runs in slices so full-window logits (seq_len x 128K vocab
in FP32) never sit in memory at once. Compare runs only at the same --seq-len.

Writes bench/results/ppl-<model>-<quant>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).parent / "results"


def wikitext2_ids(tokenizer) -> list[int]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
    )  # fmt: skip
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return tokenizer.encode(text, add_special_tokens=False)


@torch.inference_mode()
def window_nll(model, ids: torch.Tensor, head_chunk: int = 256) -> tuple[float, int]:
    """Sum of next-token NLL over one window [1, T], and the number of predictions."""
    t = ids.shape[1]
    pos = torch.arange(t, device=ids.device)[None]
    cos, sin = model.rotary(pos)
    x = model.model.embed_tokens(ids)
    for layer in model.model.layers:
        x = layer(x, cos, sin)
    x = model.model.norm(x)[0, :-1]  # positions that predict a next token
    targets = ids[0, 1:]
    total = 0.0
    for start in range(0, t - 1, head_chunk):
        logits = model.lm_head(x[start : start + head_chunk]).float()
        total += torch.nn.functional.cross_entropy(
            logits, targets[start : start + head_chunk], reduction="sum"
        ).item()
    return total, t - 1


def main() -> None:
    from transformers import AutoTokenizer

    from engine.model.loader import load_model

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant", choices=["none", "int4"], default="none")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=0, help="0 = the whole test set")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = wikitext2_ids(tok)
    quant = None if args.quant == "none" else args.quant
    model = load_model(args.model, device="cuda", quantization=quant, group_size=args.group_size)

    n_windows = len(ids) // args.seq_len
    if args.max_windows:
        n_windows = min(n_windows, args.max_windows)
    start = time.perf_counter()
    nll, count = 0.0, 0
    for w in range(n_windows):
        window = torch.tensor([ids[w * args.seq_len : (w + 1) * args.seq_len]], device="cuda")
        s, n = window_nll(model, window)
        nll, count = nll + s, count + n
    ppl = math.exp(nll / count)

    result = {
        "model": Path(args.model).name,
        "quant": args.quant if args.quant == "none" else f"{args.quant}-g{args.group_size}",
        "seq_len": args.seq_len,
        "windows": n_windows,
        "tokens_scored": count,
        "perplexity": round(ppl, 4),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "seconds": round(time.perf_counter() - start, 1),
    }
    print(json.dumps(result, indent=2))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"ppl-{result['model']}-{result['quant']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
