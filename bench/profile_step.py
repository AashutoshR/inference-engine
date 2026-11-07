"""Where does an engine step spend its time with N requests always in flight?

Drives LLMEngine directly (no HTTP) the way the server is loaded in the benchmarks: N
requests running at once, each finished request replaced by a new prompt, greedy, 128
tokens each. Reports steps per second, how many steps could use a CUDA graph, the time
per step kind, and a cProfile of the hottest Python functions.

    python bench/profile_step.py --concurrency 64 --steps 400
"""

from __future__ import annotations

import argparse
import cProfile
import io
import itertools
import json
import pstats
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "tests" / "data" / "prompts.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Llama-3.2-1B-Instruct")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--profile", action="store_true", help="also run cProfile (slows steps)")
    p.add_argument("--no-coalesce", action="store_true", help="admit prompts every step")
    p.add_argument("--coalesce", action="store_true", help="hold prompts (default: engine's)")
    p.add_argument("--no-mixed-graphs", action="store_true", help="mixed steps run eagerly")
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer

    from engine.config import load_stop_token_ids
    from engine.model.loader import load_model
    from engine.runtime.chat import chat_prompt_ids
    from engine.runtime.engine import LLMEngine
    from engine.runtime.params import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in PROMPTS.read_text("utf-8").splitlines() if line]
    prompts = itertools.cycle([chat_prompt_ids(tok, r["messages"]) for r in rows])
    model = load_model(args.model, device="cuda", dtype=torch.bfloat16)
    coalesce = False if args.no_coalesce else True if args.coalesce else None
    engine = LLMEngine(model, tok, load_stop_token_ids(args.model), coalesce_prefill=coalesce,
                       enable_mixed_graphs=False if args.no_mixed_graphs else None)  # fmt: skip
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    counter = itertools.count()

    def top_up() -> None:
        while len(engine.scheduler.running) + len(engine.scheduler.waiting) < args.concurrency:
            engine.add_request(f"r{next(counter)}", next(prompts), params)

    for _ in range(args.warmup):
        top_up()
        engine.step()
    torch.cuda.synchronize()

    prof = cProfile.Profile()
    tokens, start = 0, time.perf_counter()
    first = engine.steps_total
    if args.profile:
        prof.enable()
    for _ in range(args.steps):
        top_up()
        tokens += len(engine.step())
    prof.disable()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    log = [s for s in engine.step_log if s["step"] > first]
    graph = [s["ms"] for s in log if s["graph"]]
    mixed_graph = [s["ms"] for s in log if s["mixed_graph"]]
    eager = [s["ms"] for s in log if not s["graph"] and not s["mixed_graph"]]
    steps = engine.steps_total - first
    print(f"{args.concurrency} in flight: {steps} steps in {elapsed:.2f} s = "
          f"{steps / elapsed:.1f} steps/s, {tokens / elapsed:.0f} tokens/s "
          f"(step kinds below from the last {len(log)} steps)")  # fmt: skip
    for name, ms in (("graph (decode only)", graph), ("mixed graph (prompt tokens)", mixed_graph),
                     ("eager", eager)):  # fmt: skip
        if ms:
            print(f"  {name:26} {len(ms):4} steps ({len(ms) / len(log):.0%}), median "
                  f"{statistics.median(ms):.1f} ms, total {sum(ms) / 1000:.2f} s")  # fmt: skip
    mixed = [s["prefill_tokens"] for s in log if not s["graph"]]
    print(f"  prompt tokens per step that reads prompts: {statistics.mean(mixed or [0]):.0f}")
    if not args.profile:
        return
    out = io.StringIO()
    pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(args.top)
    print(out.getvalue())


if __name__ == "__main__":
    main()
