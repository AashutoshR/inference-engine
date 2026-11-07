"""Load and soak test against any OpenAI-compatible server (Gate 2, and Phase 3's client).

    engine serve --model models/Llama-3.2-1B-Instruct &
    python bench/load_test.py --concurrency 64 --duration 1800        # Gate 2: 30 minutes
    python bench/load_test.py --concurrency 8 --duration 60           # quick check

Each of --concurrency workers loops: pick a prompt, stream a chat completion, record
TTFT and inter-token gaps. Correctness: every request is greedy, so each prompt must
produce the same text every time; a different answer means outputs got mixed between
sequences. At the end the server's KV usage must fall back to zero (no leaked blocks).

Writes bench/results/load-<model>-c<concurrency>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

RESULTS = Path(__file__).parent / "results"
PROMPTS = Path(__file__).parent.parent / "tests" / "data" / "prompts.jsonl"


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(int(q * len(values)), len(values) - 1)]


class Stats:
    def __init__(self) -> None:
        self.ttft: list[float] = []
        self.itl: list[float] = []
        self.latency: list[float] = []
        self.tokens = 0
        self.ok = 0
        self.errors: dict[str, int] = {}
        self.mismatches: list[str] = []
        self.pairs: dict[str, tuple[str, str]] = {}  # first two different answers per prompt
        self.answers: dict[str, str] = {}

    def error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1


async def one_request(client: httpx.AsyncClient, model: str, prompt: dict, args, stats: Stats):
    body = {
        "model": model,
        "messages": prompt["messages"],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    last = None
    text = []
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                stats.error(f"http_{r.status_code}")
                await r.aread()
                return
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    stats.error("stream_error")
                    return
                if event.get("usage"):
                    stats.tokens += event["usage"]["completion_tokens"]
                for choice in event.get("choices", []):
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        now = time.perf_counter()
                        if last is None:
                            stats.ttft.append(now - start)
                        else:
                            stats.itl.append(now - last)
                        last = now
                        text.append(piece)
    except httpx.HTTPError as exc:
        stats.error(type(exc).__name__)
        return
    stats.latency.append(time.perf_counter() - start)
    stats.ok += 1
    answer = "".join(text)
    seen = stats.answers.setdefault(prompt["id"], answer)
    if seen != answer:
        stats.mismatches.append(prompt["id"])
        stats.pairs.setdefault(prompt["id"], (seen, answer))


async def split_gap(client, model: str, prompt: dict, pair: tuple[str, str], args) -> dict:
    """Where two greedy answers to one prompt part ways, how close was the model's call?

    Asks once more (server idle, batch of one) with top_logprobs=2. This third answer
    departs from at least one of the two earlier ones somewhere; at that token, the gap
    between the top two choices says how close the call was. A gap near zero is a near-tie
    that rounding can flip (batch shape and cached vs recomputed K/V change BF16 rounding);
    a large gap would point at the engine computing something wrong. The token mapping is
    trusted only when the tokens' text joins back into the answer exactly."""

    def common(x: str, y: str) -> int:
        return next((i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q),
                    min(len(x), len(y)))  # fmt: skip

    body = {"model": model, "messages": prompt["messages"], "temperature": 0,
            "max_tokens": args.max_tokens, "logprobs": True, "top_logprobs": 2}  # fmt: skip
    choice = (await client.post("/v1/chat/completions", json=body)).json()["choices"][0]
    text, content = choice["message"]["content"], choice["logprobs"]["content"]
    a, b = pair
    depart = min(common(text, a) if text != a else 10**9, common(text, b) if text != b else 10**9)
    exact = "".join(t["token"] for t in content) == text
    pos = 0
    for i, tok in enumerate(content):
        pos += len(tok["token"])
        if pos > depart:
            top = sorted((t["logprob"] for t in tok["top_logprobs"]), reverse=True)
            gap = top[0] - top[1] if len(top) > 1 else None
            return {"split_char": depart, "split_token": i, "exact_mapping": exact,
                    "top2_gap": round(gap, 4) if gap is not None and exact else None}  # fmt: skip
    return {"split_char": depart, "split_token": None, "exact_mapping": exact, "top2_gap": None}


async def worker(client, model, prompts, args, stats, deadline, rng):
    while time.monotonic() < deadline:
        await one_request(client, model, rng.choice(prompts), args, stats)


async def server_gauges(client: httpx.AsyncClient) -> dict[str, float]:
    text = (await client.get("/metrics")).text
    out = {}
    for line in text.splitlines():
        for name in ("engine_kv_blocks_used_ratio", "engine_running_sequences",
                     "engine_waiting_sequences", "engine_preemptions_total"):  # fmt: skip
            if line.startswith(name + " "):
                out[name] = float(line.split()[1])
    return out


async def main_async(args) -> dict:
    prompts = [json.loads(line) for line in PROMPTS.read_text("utf-8").splitlines() if line]
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    timeout = httpx.Timeout(args.request_timeout)
    async with httpx.AsyncClient(base_url=args.url, timeout=timeout, limits=limits) as client:
        model = (await client.get("/v1/models")).json()["data"][0]["id"]
        stats = Stats()
        start = time.monotonic()
        deadline = start + args.duration
        rngs = [random.Random(args.seed + i) for i in range(args.concurrency)]

        async def report():
            while time.monotonic() < deadline:
                await asyncio.sleep(args.report_every)
                g = await server_gauges(client)
                elapsed = time.monotonic() - start
                print(
                    f"[{elapsed:6.0f}s] ok={stats.ok} err={sum(stats.errors.values())} "
                    f"mismatch={len(stats.mismatches)} tok/s={stats.tokens / elapsed:.0f} "
                    f"running={g.get('engine_running_sequences', 0):.0f} "
                    f"kv={g.get('engine_kv_blocks_used_ratio', 0):.0%} "
                    f"preempt={g.get('engine_preemptions_total', 0):.0f}",
                    flush=True,
                )

        reporter = asyncio.create_task(report())
        await asyncio.gather(
            *(
                worker(client, model, prompts, args, stats, deadline, rngs[i])
                for i in range(args.concurrency)
            )  # fmt: skip
        )
        reporter.cancel()
        elapsed = time.monotonic() - start
        await asyncio.sleep(1.0)  # let the engine finish its last step
        final = await server_gauges(client)
        by_id = {p["id"]: p for p in prompts}
        splits = {pid: await split_gap(client, model, by_id[pid], pair, args)
                  for pid, pair in sorted(stats.pairs.items())}  # fmt: skip

    return {
        "model": model,
        "concurrency": args.concurrency,
        "duration_s": round(elapsed, 1),
        "requests_ok": stats.ok,
        "errors": stats.errors,
        "mismatched_prompts": sorted(set(stats.mismatches)),
        "mismatch_splits": splits,
        "max_split_gap": max(
            (v["top2_gap"] for v in splits.values() if v["top2_gap"] is not None), default=None
        ),  # fmt: skip
        "output_tokens_per_s": round(stats.tokens / elapsed, 1),
        "requests_per_s": round(stats.ok / elapsed, 2),
        "ttft_s": {"p50": round(pct(stats.ttft, 0.5), 4), "p99": round(pct(stats.ttft, 0.99), 4)},
        "itl_s": {"p50": round(pct(stats.itl, 0.5), 4), "p99": round(pct(stats.itl, 0.99), 4)},
        "latency_s": {"mean": round(statistics.fmean(stats.latency), 3) if stats.latency else None},
        "final_kv_used": final.get("engine_kv_blocks_used_ratio"),
        "preemptions": final.get("engine_preemptions_total"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--duration", type=float, default=60, help="seconds")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-prompts", type=int, default=0, help="use only the first N prompts")
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--report-every", type=float, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="results file stem (default load-<model>-c<N>)")
    args = p.parse_args()

    result = asyncio.run(main_async(args))
    print(json.dumps(result, indent=2))
    RESULTS.mkdir(exist_ok=True)
    stem = args.name or f"load-{result['model']}-c{args.concurrency}"
    out = RESULTS / f"{stem}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    passed = (not result["errors"] and not result["mismatched_prompts"]
              and result["requests_ok"] > 0 and result["final_kv_used"] == 0)  # fmt: skip
    print(f"saved {out}\nGATE {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
