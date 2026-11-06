"""Batch-1 decode speed for every runtime variant (Phase 3 and 4 targets).

    python bench/bench_decode.py --model models/Llama-3.2-1B-Instruct --mode graphs
    python bench/bench_decode.py --model models/Llama-3.2-3B-Instruct --mode graphs --quant int4
    python bench/bench_decode.py --model M --mode spec --spec ngram
    python bench/bench_decode.py --model M --mode spec --spec models/Llama-3.2-1B-Instruct

Modes:
  static   Phase 1 loop (contiguous cache, SDPA)
  eager    paged engine, PyTorch reference attention, no CUDA graphs
  triton   paged engine, Triton decode kernel, no CUDA graphs
  graphs   paged engine, Triton decode kernel, CUDA graphs
  spec     speculative decoding (n-gram lookup or a draft model directory)

Greedy, a 128-token prompt, --new-tokens output tokens, median of --repeats runs.
Appends a row to bench/results/decode-<model>.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from engine.model.loader import load_model
from engine.runtime.engine import LLMEngine
from engine.runtime.generate import generate
from engine.runtime.params import SamplingParams

RESULTS = Path(__file__).parent / "results"
PROMPT = (
    "Write a detailed, well-organized guide to keeping a sourdough starter alive. Cover "
    "feeding schedules, flour types, hydration, temperature, signs of a healthy starter, "
    "common problems like hooch and mold, how to revive a neglected starter, and how to "
    "store it while travelling. Use headings and short paragraphs."
)


def run_once(mode: str, model, tok, prompt: list[int], n: int, engine=None, spec=None) -> float:
    params = SamplingParams(temperature=0, max_tokens=n)
    torch.cuda.synchronize()
    start = time.perf_counter()
    if mode == "static":
        count = sum(1 for _ in generate(model, tok, [prompt], params, ()))
    elif mode == "spec":
        count = sum(1 for _ in spec.generate(tok, prompt, params, ()))
    else:
        count = len(engine.generate({f"r{time.perf_counter_ns()}": prompt}, params).popitem()[1])
    torch.cuda.synchronize()
    return count / (time.perf_counter() - start)


def main() -> None:
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["static", "eager", "triton", "graphs", "spec"], required=True)
    p.add_argument("--quant", choices=["none", "int4", "fp8"], default="none")
    p.add_argument("--spec", default="ngram", help="'ngram' or a draft model directory")
    p.add_argument("--spec-k", type=int, default=4)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    quant = None if args.quant == "none" else args.quant
    model = load_model(args.model, device="cuda", quantization=quant)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    prompt = (ids * (128 // len(ids) + 1))[:128]

    engine = spec = None
    stats = {}
    if args.mode in ("eager", "triton", "graphs"):
        engine = LLMEngine(
            model, tok, (), max_num_seqs=8,
            num_blocks=(128 + args.new_tokens) // 16 * 4 + 16,
            use_triton=args.mode != "eager", enable_cuda_graphs=args.mode == "graphs",
        )  # fmt: skip
    elif args.mode == "spec":
        from engine.runtime.speculative import SpeculativeDecoder

        draft = None
        if args.spec != "ngram":
            draft = load_model(args.spec, device="cuda", quantization=quant)
        spec = SpeculativeDecoder(model, draft, num_speculative_tokens=args.spec_k)

    run_once(args.mode, model, tok, prompt, 16, engine, spec)  # warm-up
    if spec is not None:
        spec.stats = type(spec.stats)()
    speeds = [
        run_once(args.mode, model, tok, prompt, args.new_tokens, engine, spec)
        for _ in range(args.repeats)
    ]
    if spec is not None:
        stats = {
            "acceptance_rate": round(spec.stats.acceptance_rate, 3),
            "tokens_per_target_pass": round(spec.stats.tokens_per_round, 2),
        }

    row = {
        "mode": args.mode,
        "quant": args.quant,
        "spec": args.spec if args.mode == "spec" else None,
        "decode_tokens_per_s": round(statistics.median(speeds), 1),
        "itl_ms": round(1000 / statistics.median(speeds), 2),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        **stats,
    }
    print(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"decode-{Path(args.model).name}.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    rows = [r for r in rows if (r["mode"], r["quant"], r.get("spec")) !=
            (row["mode"], row["quant"], row["spec"])] + [row]  # fmt: skip
    out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
