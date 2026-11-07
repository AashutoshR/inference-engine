# Architecture

`engine` is a single-node LLM inference server for Llama-architecture models. One
engine process owns one GPU; `engine serve` puts HTTP in a second process. This document
follows a request from HTTP to tokens and back, then covers memory, correctness and the
performance features.

```
HTTP clients  -->  FastAPI (engine/server/api.py)               HTTP process, asyncio
                     |  validate, apply chat template, 400 / 429 / 504
                     v
                   ProcessEngine (engine/server/process_engine.py)
                     |  streams, timeouts, aborts; two pipes to the engine process
                     v
                   engine process loop                          its own process and GIL
                     |  admit new requests, apply aborts, step(), send outputs
                     v
                   LLMEngine.step() (engine/runtime/engine.py)
                     +- Scheduler.schedule()       which sequences, how many tokens each
                     |    +- BlockManager          KV blocks, prefix cache, refcounts
                     +- build PagedBatch           flattened tokens + slot mapping
                     +- model forward              decode: one CUDA graph replay
                     |                             mixed: a graph per layer, attention between
                     |    +- Attention._paged      Triton kernel for decode rows,
                     |                             FlashAttention varlen for prompt chunks
                     +- sample + OutputProcessor   tokens to text deltas, stop rules
```

## Request path

1. **API** (`engine/server/api.py`). `/v1/chat/completions` applies the model's chat
   template; `/v1/completions` takes text or token ids. Pydantic validates the body;
   errors use OpenAI's error shape. The handler asks the `AsyncEngine` for a stream and
   awaits the first output before sending headers, so a full queue (HTTP 429), a bad
   request (400) or a timeout (504) is still a clean JSON error. Streaming responses are
   server-sent events in OpenAI's chunk format, ending with `data: [DONE]`.

2. **ProcessEngine** (`engine/server/process_engine.py`), built on **AsyncEngine**
   (`engine/server/async_engine.py`). The engine is not thread-safe, so exactly one
   thread touches it. Handlers get an async iterator backed by an `asyncio.Queue`.
   Leaving the iterator early (client disconnect, timeout, exception) sends an abort,
   and the engine frees the request's KV blocks before its next step. A failed engine
   step (e.g. a CUDA error) fails the in-flight requests and the loop keeps serving.

   `engine serve` runs the engine in a **process of its own** (spawned, never forked).
   In one process the engine thread and the HTTP thread share one GIL: streaming a
   token costs the HTTP side a few hundred microseconds of Python (JSON, Starlette, a
   socket write), and with 16 to 64 streams that work kept taking the GIL in the middle
   of the engine's step. Now requests go over one pipe and each step comes back as one
   message holding all its outputs (token ids and detokenized text) and the metrics. If
   the HTTP process dies, the engine process sees its pipe close and exits, so nothing
   keeps holding the GPU. `--in-process-engine` keeps the engine in a thread instead
   (most tests use that); either way a step's outputs reach the event loop with one
   wake-up, not one per token.

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
- **Prompt coalescing** (on only when mixed-step graphs are off, e.g. with FP8): a
  step that reads any prompt tokens cannot replay a decode graph, and an eager step
  costs about four graph steps however few tokens it reads, because it is bound by
  launching ~770 small GPU operations from Python. So while sequences are decoding, a
  new prompt may wait up to one step per 8 running sequences (8 steps at 64) and is
  then read together with the others that arrived meanwhile. With mixed-step graphs
  (below) a step that reads prompts costs little more than a decode step, so holding
  prompts would only delay them, and coalescing is off. `--no-prefill-coalescing`
  turns it off in every case.
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
  `engine/kernels/paged_attention.py`: each program walks a sequence's context through
  the block table in tiles, with an online softmax, reading K/V straight from the paged
  cache. In BF16/FP16 there is one program per (sequence, KV head): the query heads
  that share that KV head (4 in Llama 3.2 1B) are the rows of one small matrix, so each
  K/V tile is read once and both products run on the matrix units (`tl.dot`), with the
  softmax weights rounded to BF16 before multiplying V as FlashAttention does. In FP32
  there is one program per (sequence, query head), all in float32, which keeps the FP32
  reference mode as exact as before.
- **Mixed steps** (several sequences, some reading prompt tokens) send the decode rows
  through the same Triton kernel and all prompt chunks through **one FlashAttention
  varlen call** (`torch.nn.attention.varlen`): every chunk's context is gathered back to
  back with one index operation, and the causal mask is aligned bottom-right so each
  chunk's queries sit at the end of their context, after any prefix-cache hit. In FP32,
  which FlashAttention does not support, each chunk gets its own SDPA call instead.
  Nothing is padded, so memory grows with the tokens actually present.
- **A single sequence's prompt** uses the PyTorch reference: gather its context through
  the block table, then one `scaled_dot_product_attention` call with a causal mask.
  The kernel and the varlen path are tested against independent PyTorch implementations.

## CUDA graphs

`engine/runtime/cuda_graphs.py` has two kinds.

**Decode graphs** capture the whole decode forward pass once per batch-size bucket
(1, 2, 4, 8, 16, 24, 32, 48, 64, …). Inputs live in static buffers; a step of B
sequences copies its data in, pads to the next bucket, and replays. Padding rows read
and write only the scratch block. The Triton kernel reads each sequence's length from
GPU memory, so one graph serves every context length.

**Mixed-step graphs** (`MixedGraphs`) cover steps that read prompt tokens. Their
attention changes shape every step (which rows decode, how long each chunk is), but the
rest of the network treats the step's N tokens row by row. So each layer is split at
attention: one graph runs the previous layer's o_proj, residual and MLP plus this
layer's norm, q/k/v projections, RoPE and KV-cache write; attention runs eagerly in
between, reading q from and writing its output to fixed addresses. That is 17 replays
(16 layers + 1) and a few attention launches per layer instead of ~770 launches, which
on this Windows laptop cost about 44 µs each: a step reading 40 prompt tokens next to
15 decodes did 14 ms of GPU work in 77 ms, and now takes about 25 ms. N is padded to a
token bucket (16, 32, ... 2,048; longer steps are bound by GPU math, not launches, and
stay eager); padding rows embed token 0, write K/V to the scratch block, and are never
attended to or sampled. Off for FP8, whose per-step activation scale padding rows
could change. `--no-mixed-graphs` turns it off.

## Fused kernels

`engine/kernels/fused.py` replaces three chains of small PyTorch ops with Triton
kernels: RoPE for q and for k (12 ops, now 2 kernels), SiLU times the up projection
(2, now 1), and the element-wise part of RMSNorm (add, rsqrt, multiply, cast, scale:
now 1 kernel, after PyTorch's own cast, square and mean). That removes about 17
kernels per layer, roughly 270 per step. Each kernel rounds where PyTorch rounds (and
turns off multiply-add fusion, which would skip a rounding), so all three are bitwise
identical to the unfused ops in every dtype.

RMSNorm's mean stays in PyTorch on purpose. A first version fused it too, with its sum
of squares added in a different order. Tests on random data looked fine (99.9% of BF16
outputs equal, the rest one rounding step apart), but on Llama 3.2 3B in BF16 it
changed 27 of Gate 1's 200 greedy answers. `ENGINE_FUSED_KERNELS=0` turns them all off.

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
| Mixed-step graphs | The same engine running those steps eagerly (FP32 and BF16, with chunking and prefix hits) |
| Fused kernels | The PyTorch ops they replace: bitwise equal in FP32, BF16 and FP16 |
| Engine in its own process | The in-process engine through the same API: same answers, streamed and not |
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
snapshot the engine loop takes at most every 250 ms (`LLMEngine.snapshot()`) and, in
process mode, sends to the HTTP process: one
character per KV block (free, cached, owned, shared) with its owner, every running and
waiting sequence, totals, and a log of the last 120 steps (tokens, prefill/decode mix,
milliseconds, whether a decode or mixed-step CUDA graph ran). Taking the snapshot on the
engine's own thread means no locks in the hot path and a consistent picture.

`/metrics` (Prometheus): requests by outcome, prompt and generated tokens, TTFT, ITL,
step time, running and waiting sequences, KV blocks in use, preemptions and the
prefix-cache hit ratio. `/health` reports whether the engine (its process, and the
thread that reads its results) is alive.
