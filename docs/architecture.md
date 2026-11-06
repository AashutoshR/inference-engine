# Architecture

`engine` is a single-node LLM inference server for Llama-architecture models. One
process owns one GPU. This document follows a request from HTTP to tokens and back,
then covers memory, correctness and the performance features.

```
HTTP clients  -->  FastAPI (engine/server/api.py)               asyncio event loop
                     |  validate, apply chat template, 400 / 429 / 504
                     v
                   AsyncEngine (engine/server/async_engine.py)   one background thread
                     |  admit new requests, apply aborts, step(), route outputs
                     v
                   LLMEngine.step() (engine/runtime/engine.py)
                     +- Scheduler.schedule()       which sequences, how many tokens each
                     |    +- BlockManager          KV blocks, prefix cache, refcounts
                     +- build PagedBatch           flattened tokens + slot mapping
                     +- model forward              decode: CUDA graph replay
                     |    +- Attention._paged      decode: Triton kernel, else PyTorch
                     +- sample + OutputProcessor   tokens to text deltas, stop rules
```

## Request path

1. **API** (`engine/server/api.py`). `/v1/chat/completions` applies the model's chat
   template; `/v1/completions` takes text or token ids. Pydantic validates the body;
   errors use OpenAI's error shape. The handler asks the `AsyncEngine` for a stream and
   awaits the first output before sending headers, so a full queue (HTTP 429), a bad
   request (400) or a timeout (504) is still a clean JSON error. Streaming responses are
   server-sent events in OpenAI's chunk format, ending with `data: [DONE]`.

2. **AsyncEngine** (`engine/server/async_engine.py`). The engine is not thread-safe, so
   exactly one thread touches it. Handlers enqueue `(id, prompt, params)` and get an
   async iterator backed by an `asyncio.Queue`; the engine thread delivers outputs with
   `call_soon_threadsafe`. Leaving the iterator early (client disconnect, timeout,
   exception) queues an abort, and the next loop iteration frees the request's KV
   blocks. A failed engine step (e.g. a CUDA error) fails the in-flight requests and the
   loop keeps serving.

3. **LLMEngine.step()** (`engine/runtime/engine.py`) runs one scheduler step, one model
   forward pass and one round of sampling. Greedy rows are sampled with a single
   `argmax` and one device sync for the whole batch.

## Scheduling

`engine/runtime/scheduler.py` implements continuous batching: sequences join and
leave the batch at every step. Each step has a token budget (`max_num_batched_tokens`):

- Running sequences go first, oldest first. A decoding sequence costs one token; a
  sequence part-way through its prompt gets the next chunk (**chunked prefill**), so a
  2,000-token prompt never stalls the decodes around it.
- Leftover budget admits waiting sequences, at most `max_num_seqs` running.
- If a running sequence needs a KV block and none is free, the newest running
  sequence is **preempted**: its blocks are freed and it goes back to the front of the
  queue to be recomputed later. Its full blocks stay in the prefix cache, so the
  recompute is often cheap.

The invariant that makes this simple: `seq.num_computed` tokens of a sequence have
K/V in the cache. A step feeds `all_ids[num_computed : num_computed + n]`, and a token
is sampled only when that reaches the end of the sequence.

## KV memory

**Paged cache** (`engine/runtime/paged_cache.py`). Per layer, one flat array of slots,
`[num_blocks * block_size, kv_heads, head_dim]`. A sequence owns a *block table*, the
list of its block ids; token `i` lives in slot `table[i // B] * B + i % B`. Nothing is
reserved ahead of time, so fragmentation is at most one partly filled block per
sequence. One extra block past the allocator's range is scratch space for CUDA-graph
padding rows.

**Block manager** (`engine/runtime/block_manager.py`). A free list plus reference
counts. **Prefix caching:** a full block is named by a hash of its tokens chained with
the previous block's hash, so equal hashes mean equal prefixes. Full blocks are
registered under their hash. A new sequence first walks its prompt block by block and
takes references to matching cached blocks; only the rest is computed (at least the
last token always is, since its logits are needed). When the last owner frees a hashed
block it becomes *evictable*: still valid and findable, reclaimed least recently used
first when free blocks run out. Sequences in the waiting queue never hold blocks,
which rules out a starvation deadlock after preemption.

**Sizing.** `LLMEngine` fills the GPU memory left after loading weights (minus
headroom for activations, logits and CUDA graphs) with KV blocks. `engine serve` prints
the resulting capacity.

## Attention

All attention runs over the flattened tokens of a step, `[1, N]`.

- **Decode steps** (one new token per sequence) use the Triton kernel in
  `engine/kernels/paged_attention.py`: one program per (sequence, query head) walks the
  context through the block table in tiles, with an online softmax, reading K/V
  straight from the paged cache. GQA maps query head `h` to KV head `h // groups`.
- **Prefill, chunked and mixed steps** use the PyTorch reference: gather each
  sequence's context through its block table into a padded view, then one
  `scaled_dot_product_attention` call with a mask that is causal per sequence. The
  kernel is tested against an independent PyTorch implementation.

## CUDA graphs

`engine/runtime/cuda_graphs.py` captures the whole decode forward pass once per
batch-size bucket (1, 2, 4, 8, 16, 24, 32, 48, 64, …). Inputs live in static buffers;
a step of B sequences copies its data in, pads to the next bucket, and replays. Padding
rows read and write only the scratch block. The Triton kernel reads each sequence's
length from GPU memory, so one graph serves every context length.

## Model and weights

`engine/model/` is a plain PyTorch Llama: RMSNorm, RoPE with Llama 3 frequency
scaling, grouped-query attention, SwiGLU. Module names match Hugging Face, so a
checkpoint loads with `load_state_dict(strict=True)` and no renaming. The loader builds
the model on the meta device, then streams tensors from safetensors on the CPU, casting
each one before moving it to the GPU, so peak memory equals the final size.

**Quantization** (`engine/quant/`), applied tensor by tensor while loading:

- `int4`: group-wise round-to-nearest, 4-bit codes plus a BF16 scale and zero per group
  of 128 input channels, multiplied by PyTorch's fused tinygemm kernel
  (`_weight_int4pack_mm`), which unpacks inside the matmul.
- `fp8`: e4m3 weights and dynamically quantized activations through `_scaled_mm`, on
  GPUs with FP8 tensor cores (Ada, Hopper).

Embeddings and the LM head stay in BF16.

## Speculative decoding

`engine/runtime/speculative.py`, single sequence. A proposer (a smaller draft model
or n-gram lookup over the context) guesses k tokens; the target scores all of them in
one forward pass. Acceptance follows Leviathan et al.: accept guess x with probability
min(1, p(x)/q(x)), otherwise sample from norm(max(0, p - q)). This keeps the target's
output distribution exactly (verified by a chi-square test on 10,000 samples), and
greedy speculative output equals plain greedy output token for token.

## Correctness strategy

Every optimization is tested for *identical output* against something simpler:

| Component | Oracle |
| --- | --- |
| Forward pass | Hugging Face logits with the same weights (1e-4) |
| Whole model, real weights | Gate 1: HF greedy tokens on 200 prompts (FP32 exact) |
| KV cache | A no-cache forward pass recomputed every step |
| Paged engine, batching, preemption, chunking, prefix caching | The Phase 1 static loop, token for token, including ~200 random workloads (Hypothesis) |
| Block allocator | Invariant checks after every operation and every engine step |
| Triton kernel | A per-sequence PyTorch implementation |
| CUDA graphs | The same engine without graphs |
| Speculative decoding | Plain greedy (exact) and a chi-square test (sampled) |
| HTTP API | The engine called directly; streaming text equals non-streaming |

## Token confidence

With `logprobs` (and optionally `top_logprobs`, up to 20) a request gets, for every
generated token, its log-probability under the model's own distribution (before
temperature and top-k/p) and the most likely alternatives. The engine computes them with
one batched log-softmax per step, only for the rows that asked. The API returns them in
OpenAI's chat format (`logprobs.content`) or the legacy completions format, streamed one
entry per token. The dashboard playground uses them to tint each token by certainty.

## Observability

`/dashboard` is a single HTML page served by the engine. It polls `/debug/state`, a
snapshot the engine thread takes at most every 250 ms (`LLMEngine.snapshot()`): one
character per KV block (free, cached, owned, shared) with its owner, every running and
waiting sequence, totals, and a log of the last 120 steps (tokens, prefill/decode mix,
milliseconds, whether a CUDA graph ran). Taking the snapshot on the engine thread means
no locks in the hot path and a consistent picture.

`/metrics` (Prometheus): requests by outcome, prompt and generated tokens, TTFT, ITL,
step time, running and waiting sequences, KV blocks in use, preemptions and the
prefix-cache hit ratio. `/health` reports whether the engine thread is alive.
