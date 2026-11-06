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
    model = load_model(args.model, device=args.device)
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
    print(
        "`engine serve` is Phase 2 (paged KV cache, continuous batching, OpenAI API).",
        file=sys.stderr,
    )
    return 2


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
    gen.set_defaults(func=cmd_generate)

    serve = sub.add_parser("serve", help="OpenAI-compatible server (Phase 2)")
    serve.add_argument("--model", required=True)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
