"""Command line: ``engine info``, ``engine generate``; ``engine serve`` arrives in Phase 2."""

from __future__ import annotations

import argparse
import sys
import time

from engine.config import (
    ModelConfig,
    estimate_param_count,
    kv_bytes_per_token,
    load_stop_token_ids,
    max_kv_tokens,
)
from engine.runtime.params import SamplingParams

GIB = 2**30


def cmd_info(args: argparse.Namespace) -> int:
    """Print the model shape and whether it fits on a GPU of --gpu-mem GiB. No torch needed."""
    cfg = ModelConfig.from_pretrained(args.model)
    params = estimate_param_count(cfg)
    print(f"architecture        {cfg.architecture}")
    print(f"layers / hidden     {cfg.num_hidden_layers} / {cfg.hidden_size}")
    print(
        f"heads (q / kv)      {cfg.num_attention_heads} / {cfg.num_key_value_heads}"
        f"  (GQA groups: {cfg.num_kv_groups}, head dim {cfg.head_dim})"
    )
    print(f"parameters          {params / 1e9:.2f} B")
    print(f"KV cache per token  {kv_bytes_per_token(cfg) / 1024:.0f} KiB (bf16)")
    print(f"\nOn a {args.gpu_mem:g} GiB GPU (1 GiB reserved for activations and CUDA context):")
    for label, bytes_per_param in (("bf16", 2.0), ("int8", 1.0), ("int4", 0.5)):
        weights_gib = params * bytes_per_param / GIB
        tokens = max_kv_tokens(cfg, args.gpu_mem, bytes_per_param)
        verdict = f"{tokens:>9,} KV tokens fit" if tokens else "      does not fit"
        print(f"  {label}  weights {weights_gib:5.1f} GiB  ->{verdict}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    # Heavy imports stay inside the command so `engine info` works without torch.
    from transformers import AutoTokenizer

    from engine.model.loader import load_model
    from engine.runtime.chat import chat_prompt_ids
    from engine.runtime.generate import generate

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant = None if args.quantization == "none" else args.quantization
    model = load_model(args.model, device=args.device, quantization=quant)
    prompt_ids = chat_prompt_ids(tokenizer, [{"role": "user", "content": args.prompt}])
    params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
    )

    start, n_tokens = time.perf_counter(), 0
    for out in generate(model, tokenizer, [prompt_ids], params, load_stop_token_ids(args.model)):
        n_tokens += 1
        print(out.text, end="", flush=True)
    elapsed = time.perf_counter() - start
    print(
        f"\n\n[{n_tokens} tokens in {elapsed:.2f}s, {n_tokens / elapsed:.1f} tokens/s]",
        file=sys.stderr,
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """OpenAI-compatible server: paged KV cache, continuous batching, SSE streaming."""
    import logging
    from pathlib import Path

    import uvicorn
    from transformers import AutoTokenizer

    from engine.model.loader import load_model
    from engine.runtime.engine import LLMEngine
    from engine.server.api import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    import torch

    quant = None if args.quantization == "none" else args.quantization
    model = load_model(
        args.model,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        quantization=quant,
        group_size=args.group_size,
    )
    engine = LLMEngine(
        model,
        tokenizer,
        load_stop_token_ids(args.model),
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=not args.no_prefix_caching,
        enable_chunked_prefill=not args.no_chunked_prefill,
        enable_cuda_graphs=False if args.no_cuda_graphs else None,
        coalesce_prefill=False if args.no_prefill_coalescing else None,
    )
    cap = engine.scheduler.capacity_tokens
    print(
        f"KV cache: {engine.block_manager.allocator.num_blocks} blocks x {args.block_size} = "
        f"{cap:,} tokens; loaded in {time.perf_counter() - start:.1f}s",
        file=sys.stderr,
    )
    name = args.served_model_name or Path(args.model).name
    app = create_app(
        engine, tokenizer, name, max_waiting=args.max_waiting, request_timeout=args.timeout
    )
    print(f"dashboard: http://{args.host}:{args.port}/dashboard", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show model shape and GPU memory fit")
    info.add_argument("--model", required=True, help="local model directory")
    info.add_argument("--gpu-mem", type=float, default=8.0, help="GPU memory in GiB")
    info.set_defaults(func=cmd_info)

    gen = sub.add_parser("generate", help="chat completion for one prompt, streamed")
    gen.add_argument("--model", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--max-tokens", type=int, default=256)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--quantization", choices=["none", "int4", "fp8"], default="none")
    gen.set_defaults(func=cmd_generate)

    serve = sub.add_parser("serve", help="OpenAI-compatible server")
    serve.add_argument("--model", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default=None, help="default: model folder name")
    serve.add_argument("--device", default="cuda")
    serve.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    serve.add_argument("--max-num-seqs", type=int, default=64, help="max running sequences")
    serve.add_argument("--max-num-batched-tokens", type=int, default=4096)
    serve.add_argument("--max-waiting", type=int, default=256, help="queue limit; then HTTP 429")
    serve.add_argument("--timeout", type=float, default=600.0, help="per-request seconds")
    serve.add_argument("--block-size", type=int, default=16)
    serve.add_argument("--num-blocks", type=int, default=None, help="default: fill free GPU memory")
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    serve.add_argument("--no-prefix-caching", action="store_true")
    serve.add_argument("--no-chunked-prefill", action="store_true")
    serve.add_argument("--quantization", choices=["none", "int4", "fp8"], default="none")
    serve.add_argument("--group-size", type=int, default=128, help="int4 group size")
    serve.add_argument("--no-cuda-graphs", action="store_true")
    serve.add_argument(
        "--no-prefill-coalescing",
        action="store_true",
        help="admit new prompts every step instead of reading them together",
    )
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
