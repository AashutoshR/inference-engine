"""When a prompt's answer changes under load, is it a bug or rounding?

Under load the same greedy prompt can come back with a different answer, because the
batch it shares a step with changes the shapes of the matrix multiplies (and so the order
numbers are added in), and because a prefix-cache hit reuses keys and values computed in
a different batch. Neither should change the math, only the rounding.

This runs each prompt two ways through LLMEngine on the GPU, with every feature on:

    alone    one request at a time, prefix cache off: the reference
    loaded   all prompts twice, arriving one per step while others run, prefix cache on,
             so steps mix prompt chunks with decoding and the second copies hit the cache

and compares the greedy tokens. In FP32, rounding is about 65,000 times finer than BF16,
so a correct engine should give the same tokens both ways except at a genuine near-tie.
For every split it records the top-2 logit gap at that token in the reference run.

    python bench/batch_invariance.py --model models/Llama-3.2-1B-Instruct --dtype float32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "tests" / "data" / "prompts.jsonl"
RESULTS = Path(__file__).parent / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Llama-3.2-1B-Instruct")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
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
    rows = rows[: args.max_prompts]
    prompts = {r["id"]: chat_prompt_ids(tok, r["messages"]) for r in rows}
    model = load_model(args.model, device="cuda", dtype=getattr(torch, args.dtype))
    stops = load_stop_token_ids(args.model)
    greedy = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    traced = SamplingParams(temperature=0, max_tokens=args.max_tokens, logprobs=True,
                            top_logprobs=2)  # fmt: skip

    t0 = time.perf_counter()
    solo_engine = LLMEngine(model, tok, stops, enable_prefix_caching=False)
    alone, gaps = {}, {}
    for rid, ids in prompts.items():
        solo_engine.add_request(rid, ids, traced)
        outs = []
        while solo_engine.has_unfinished():
            outs += solo_engine.step()
        alone[rid] = [o.token_id for o in outs]
        gaps[rid] = [
            (o.top_logprobs[0][1] - o.top_logprobs[1][1]) if len(o.top_logprobs) > 1 else None
            for o in outs
        ]
    del solo_engine
    torch.cuda.empty_cache()

    engine = LLMEngine(model, tok, stops, enable_prefix_caching=True)
    queue = [(f"{rid}#{k}", ids) for k in (1, 2) for rid, ids in prompts.items()]
    loaded = {name: [] for name, _ in queue}
    max_batch = 0
    while queue or engine.has_unfinished():
        if queue:
            name, ids = queue.pop(0)
            engine.add_request(name, ids, greedy)
        for out in engine.step():
            loaded[out.request_id].append(out.token_id)
        max_batch = max(max_batch, len(engine.scheduler.running))
    seconds = time.perf_counter() - t0

    splits = []
    for name, got in loaded.items():
        rid = name.split("#")[0]
        want = alone[rid]
        if got == want:
            continue
        at = next((i for i, (x, y) in enumerate(zip(got, want, strict=False)) if x != y),
                  min(len(got), len(want)))  # fmt: skip
        gap = gaps[rid][at] if at < len(gaps[rid]) else None
        splits.append({"request": name, "diverged_at": at,
                       "top2_gap": None if gap is None else round(gap, 4)})  # fmt: skip

    known = [s["top2_gap"] for s in splits if s["top2_gap"] is not None]
    result = {
        "model": Path(args.model).name,
        "dtype": args.dtype,
        "prompts": len(prompts),
        "loaded_requests": len(loaded),
        "max_batch": max_batch,
        "identical": len(loaded) - len(splits),
        "differed": len(splits),
        "max_split_gap": max(known, default=None),
        "cached_prompt_tokens": engine.cached_prompt_tokens_total,
        "seconds": round(seconds, 1),
        "splits": splits,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "splits"}, indent=2))
    for s in splits:
        print(s)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"invariance-{result['model']}-{args.dtype}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
