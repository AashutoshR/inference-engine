# Results

Measured on **NVIDIA GeForce RTX 4060 Laptop GPU** (8.0 GiB), PyTorch 2.15.0+cu132, CUDA 13.2, 2026-11-07. Every number here comes from a JSON file in [`bench/results/`](bench/results), produced by `python bench/run_all.py` and drawn by `python bench/report.py`.

A browsable version of this page: [`docs/results.html`](docs/results.html).

## Gates

| Check | Result | Detail |
| --- | --- | --- |
| Gate 1 · 1B FP32 exact | ✅ pass | 200 exact, 0 near-ties, 0 failed of 200 |
| Gate 1 · 3B BF16 near-tie | ❌ fail | 182 exact, 11 near-ties, 7 failed of 200 |
| Gate 2 · 64 streams soak | ❌ fail | 30 min, 52,728 requests, 0 errors, KV left in use 0%, repeat answers differed: 160 prompts; of 88 measured, 85 split where the top two choices were within 0.25 (largest gap 3.625) |
| Gate 3 · 70% of vLLM | ✅ pass | 113% of vLLM at 64 users: 1,982 vs 1,756 tokens/s |
| Unit tests (CPU) | ✅ pass | 100 passed, 0 failed, 24.8 s |
| GPU tests | ✅ pass | 36 passed, 0 failed, 13.2 s |

## Headline numbers

| | | |
| --- | --- | --- |
| One user, 1B | **194 tok/s** | 3.3x the Phase 1 loop |
| Many users, 1B | **1,982 tok/s** | 64 concurrent streams |
| Matches Hugging Face | **393 / 400** | prompts, 38,166 tokens compared |
| INT4 quality cost, 3B | **+0.96** | perplexity, 6.31 to 2.45 GiB |

## Correctness: identical output to Hugging Face

Each square is one of the 200 test prompts. The engine and Hugging Face both generate 128 tokens greedily; green means every token matched. In BF16, amber marks a prompt where the two picked different tokens at a point where the model's top two choices were within 0.01 of each other (a coin flip that rounding can decide); anything else would be red.

![Gate 1, 1B](docs/results/gate1-1b.svg)

![Gate 1, 3B](docs/results/gate1-3b.svg)

**Llama-3.2-3B-Instruct bfloat16: the 7 failed prompts.** Where the two implementations first chose a different token, and how far apart Hugging Face's top two logits were there.

| prompt | prompt tokens | diverged at token | top-2 logit gap |
| --- | ---: | ---: | ---: |
| long-00 | 302 | 82 | 0.125 |
| long-04 | 280 | 28 | 0.125 |
| long-21 | 1068 | 74 | 0.125 |
| long-24 | 955 | 21 | 0.125 |
| long-25 | 1189 | 23 | 0.125 |
| long-27 | 1167 | 98 | 0.125 |
| long-31 | 1415 | 39 | 0.25 |

## One user: how fast does a single reply stream?

Each bar adds one technique on top of the one above it. Batch size 1, a 128-token prompt, 256 generated tokens, greedy, median of three runs.

![decode speed Llama-3.2-1B-Instruct](docs/results/decode-Llama-3.2-1B-Instruct.svg)

| Variant | Tokens/s | ms per token | Peak GPU memory |
| --- | ---: | ---: | ---: |
| Phase 1 loop (no paging) | 58.5 | 17.09 | 2.33 GiB |
| Paged engine, PyTorch attention | 42.2 | 23.7 | 2.37 GiB |
| + Triton attention kernel | 63.2 | 15.82 | 2.37 GiB |
| + CUDA graphs | 91.4 | 10.94 | 2.5 GiB |
| + CUDA graphs · FP8 | 118.0 | 8.48 | 1.53 GiB |
| + CUDA graphs · INT4 | 194.2 | 5.15 | 1.18 GiB |
| Speculative, n-gram lookup | 54.5 | 18.36 | 2.41 GiB |

![decode speed Llama-3.2-3B-Instruct](docs/results/decode-Llama-3.2-3B-Instruct.svg)

| Variant | Tokens/s | ms per token | Peak GPU memory |
| --- | ---: | ---: | ---: |
| Phase 1 loop (no paging) | 29.0 | 34.5 | 6.04 GiB |
| + CUDA graphs | 36.5 | 27.42 | 6.34 GiB |
| + CUDA graphs · FP8 | 52.0 | 19.23 | 3.57 GiB |
| + CUDA graphs · INT4 | 86.7 | 11.54 | 2.49 GiB |
| Speculative, n-gram lookup | 28.4 | 35.23 | 6.12 GiB |
| Speculative, 1B draft · INT4 | 25.0 | 40.0 | 3.31 GiB |

## Many users: what happens as load grows?

The real server (`engine serve`, 1B, BF16) under N clients that each stream chat completions back to back for 60 seconds, 128 tokens per reply.

![throughput](docs/results/load-throughput.svg)

![latency](docs/results/load-latency.svg)

| Users | Output tok/s | Requests/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | Errors | Preemptions | Repeat answers differed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 82 | 0.95 | 28 ms | 204 ms | 11.4 ms | 19.8 ms | 0 | 0 | 2 prompts; of 2 measured, 2 split where the top two choices were within 0.25 (largest gap 0.125) |
| 4 | 290 | 3.14 | 38 ms | 343 ms | 12.2 ms | 33.3 ms | 0 | 0 | 22 prompts; of 14 measured, 13 split where the top two choices were within 0.25 (largest gap 3.625) |
| 16 | 888 | 9.23 | 53 ms | 319 ms | 14.3 ms | 69.2 ms | 0 | 0 | 81 prompts; of 54 measured, 53 split where the top two choices were within 0.25 (largest gap 3.625) |
| 32 | 1,419 | 14.56 | 85 ms | 417 ms | 17.6 ms | 69.3 ms | 0 | 0 | 63 prompts; of 42 measured, 40 split where the top two choices were within 0.25 (largest gap 3.625) |
| 64 | 1,982 | 20.85 | 184 ms | 1779 ms | 23.8 ms | 134.4 ms | 0 | 0 | 125 prompts; of 74 measured, 73 split where the top two choices were within 0.25 (largest gap 1.625) |

**Soak (Gate 2):** 64 streams for 30 minutes, 52,728 requests, 2,752 tokens/s, 0 errors, repeat answers differed: 160 prompts; of 88 measured, 85 split where the top two choices were within 0.25 (largest gap 3.625), KV memory in use at the end: 0%.

## Against vLLM (Gate 3)

Gate 3 runs the same load test against vLLM v0.32.0, the most widely used open source LLM server, in Docker on the same GPU, same model (Llama 3.2 1B, BF16), same 64-sequence limit, with its defaults otherwise (prefix caching, chunked prefill and CUDA graphs on). The gate: 70% of vLLM's throughput at 64 users.

![engine vs vLLM](docs/results/vs-vllm.svg)

| users | engine tok/s | vLLM tok/s | engine / vLLM | first token p50, engine | first token p50, vLLM | per token p50, engine | per token p50, vLLM | repeat answers differed, vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 82 | 86 | 96% | 28 ms | 31 ms | 11.4 ms | 10.7 ms | 2 prompts |
| 4 | 290 | 319 | 91% | 38 ms | 50 ms | 12.2 ms | 11.3 ms | 20 prompts; of 12 measured, 12 split where the top two choices were within 0.25 (largest gap 0.125) |
| 16 | 888 | 990 | 90% | 53 ms | 88 ms | 14.3 ms | 12.2 ms | 83 prompts; of 58 measured, 56 split where the top two choices were within 0.25 (largest gap 3.875) |
| 32 | 1,419 | 1,360 | 104% | 85 ms | 168 ms | 17.6 ms | 13.9 ms | 90 prompts; of 58 measured, 55 split where the top two choices were within 0.25 (largest gap 3.875) |
| 64 | 1,982 | 1,756 | 113% | 184 ms | 414 ms | 23.8 ms | 19.8 ms | 117 prompts; of 71 measured, 70 split where the top two choices were within 0.25 (largest gap 3.875) |

## Batching changes rounding, not math

Under load, the same greedy prompt can come back with a different answer. Is that a bug or rounding? bench/batch_invariance.py runs 64 prompts through the engine on the GPU two ways: alone, one at a time with the prefix cache off, and loaded, all of them twice over with one new arrival per step, mixed prompt-and-decode steps and the prefix cache on. If batching or caching changed the math, FP32 would show it; if it only changes rounding, FP32 (65,000 times finer than BF16) should match almost everywhere.

| weights | loaded requests | largest batch | identical | differed | largest top-2 gap at a split |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 128 | 64 | 128 | 0 |  |
| bfloat16 | 128 | 64 | 71 | 57 | 0.25 |

## Quality: does quantization hurt?

Perplexity on the WikiText-2 test set (about 290,000 tokens, 2,048-token windows). Lower is better; the gap to BF16 is the price of smaller weights.

![perplexity](docs/results/perplexity.svg)

| model | weights | perplexity | vs BF16 | peak GPU memory |
| --- | --- | ---: | ---: | ---: |
| Llama 3.2 1B | BF16 | 13.17 |  | 2.62 GiB |
| Llama 3.2 1B | FP8 | 13.33 | +0.16 | 1.71 GiB |
| Llama 3.2 1B | INT4, groups of 128 | 15.84 | +2.67 | 1.29 GiB |
| Llama 3.2 3B | BF16 | 11.05 |  | 6.31 GiB |
| Llama 3.2 3B | FP8 | 11.14 | +0.09 | 3.68 GiB |
| Llama 3.2 3B | INT4, groups of 128 | 12.01 | +0.96 | 2.45 GiB |

## Reading prompts (prefill)

| Model | 512-token prompt | 2,000-token prompt | Peak memory |
| --- | ---: | ---: | ---: |
| Llama 3.2 1B | 11,480 tok/s | 11,969 tok/s | 2.42 GiB |
| Llama 3.2 3B | 4,561 tok/s | 4,314 tok/s | 6.12 GiB |

## Reproduce

```bash
python bench/run_all.py      # about two hours
python bench/report.py       # rebuilds this page
```
