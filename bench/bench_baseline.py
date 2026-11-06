"""Phase 1 baseline: batch-1 decode speed, prefill speed and peak memory (Week 3).

    python bench/bench_baseline.py --model models/Llama-3.2-1B-Instruct

Writes bench/results/phase1-<model>.json. Commit the results: Phase 2 is measured against them.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from engine.config import load_stop_token_ids
from engine.model.loader import load_model
from engine.runtime.chat import chat_prompt_ids
from engine.runtime.generate import generate
from engine.runtime.params import SamplingParams

RESULTS = Path(__file__).parent / "results"


def timed(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def bench_prefill(model, prompt_len: int, repeats: int) -> float:
    ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device="cuda")
    positions = torch.arange(prompt_len, device="cuda").unsqueeze(0)
    model(ids, positions, last_only=True)  # warm-up
    times = [timed(lambda: model(ids, positions, last_only=True)) for _ in range(repeats)]
    return prompt_len / statistics.median(times)


def bench_decode(model, tokenizer, stop_ids, new_tokens: int, repeats: int) -> float:
    prompt = chat_prompt_ids(
        tokenizer, [{"role": "user", "content": "Tell me a long story about a lighthouse keeper."}]
    )
    params = SamplingParams(temperature=0, max_tokens=new_tokens)

    def run() -> None:
        for _ in generate(model, tokenizer, [prompt], params, stop_token_ids=()):  # never stop
            pass

    run()  # warm-up
    times = [timed(run) for _ in range(repeats)]
    return new_tokens / statistics.median(times)


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device="cuda", dtype=torch.bfloat16)
    stop_ids = load_stop_token_ids(args.model)
    torch.cuda.reset_peak_memory_stats()

    result = {
        "model": Path(args.model).name,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "dtype": "bfloat16",
        "prefill_tokens_per_s": {
            n: round(bench_prefill(model, n, args.repeats)) for n in (512, 2000)
        },
        "decode_tokens_per_s_batch1": round(
            bench_decode(model, tokenizer, stop_ids, args.new_tokens, args.repeats), 1
        ),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"phase1-{result['model']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
