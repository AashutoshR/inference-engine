# engine

**A large language model server, built from scratch, that you can watch think.**

`engine` takes a Llama model and serves it to many people at once through the same API
as OpenAI, so any OpenAI client or SDK works with it unchanged. It is the same kind of
software as vLLM or Hugging Face TGI, the layer that sits between a model's weights and
every chat app built on top of them, written from first principles to run on a laptop
with an 8 GB GPU.

What makes it different is that nothing is hidden. Open `/dashboard` while it runs and
you see the GPU's memory being handed out block by block, requests joining and leaving
the batch on every step, prompts sharing memory because they start with the same words,
and the scheduler pausing a request when memory runs out. And every speed trick in it is
tested for correctness: the model must pick the same words as Hugging Face's reference
implementation on 200 test prompts, and the batched, paged, cached engine must produce
exactly the same tokens as a plain one-request loop.

![The live dashboard while 64 users stream at once](docs/images/dashboard.png)

*The live dashboard during a 64-user run: 55 requests generating in the same steps, 67%
of prompt tokens served from the prefix cache, nothing paused for memory. In the step
chart, flat teal steps are CUDA-graph replays; the tall amber spikes are steps where
several waiting prompts are read together. The screenshot predates mixed-step graphs
(below), which made those steps much shorter.*

**Results:** every gate, every benchmark and every chart is in [RESULTS.md](RESULTS.md),
with the headline numbers [below](#results).

## Why this exists

Running a language model for one person is easy: load the weights, feed in the prompt,
pick the next word, repeat. Serving it to hundreds of people at once on one GPU is a
different problem, and almost none of it is about the model. It is about memory and
scheduling.

**The memory problem.** To avoid recomputing the whole conversation for every new word,
a model keeps the *keys and values* of every past token in GPU memory: the KV cache. For
Llama 3.2 3B that is 112 KB per token, so one 8,000-token conversation needs about 900
MB. A simple server reserves the maximum length for every request up front, and most of
that memory sits empty. Fewer requests fit, and the expensive GPU idles.

**The scheduling problem.** Generating text is one word at a time, and each step for a
single request barely uses the GPU. The way to go fast is to run many requests in the
same step. But requests arrive at random times, have different lengths and finish at
different moments. If the server waits for a whole batch to finish before starting new
ones, short answers wait behind long ones and the GPU idles again.

**The cost problem.** Each generated word requires reading every model weight from
memory once. For a 3 billion parameter model in 16-bit numbers that is 6 GB of reading
per word. Speed is capped by memory bandwidth, not math.

Production servers solve these with a handful of well-known techniques. This project
builds every one of them by hand, explains each one, and shows them working live.

## What it does

| Problem | Technique in `engine` | Effect |
| --- | --- | --- |
| KV memory wasted on reserved space | **Paged KV cache**: memory in 16-token blocks, handed out as needed | Waste limited to one partly filled block per request |
| Same system prompt recomputed for every user | **Prefix caching**: blocks named by a hash of their content and shared | Repeated prompt prefixes cost nothing after the first time |
| GPU idle between requests | **Continuous batching**: requests join and leave on every step | The batch is always as full as memory allows |
| One long prompt freezes everyone | **Chunked prefill**: long prompts are read in slices alongside other work | Steady speed for running requests |
| Memory runs out mid-generation | **Preemption**: pause the newest request, free its memory, resume it later | No crashes, no out-of-memory errors |
| Attention reads scattered memory | **Triton paged attention kernel**: reads keys and values straight from the blocks | No copying into a contiguous buffer |
| Thousands of tiny GPU launches per word | **CUDA graphs**: each decode step recorded once and replayed | Python and launch overhead removed |
| New prompts knock every step off the fast path | **Mixed-step graphs**: steps that read prompts replay one graph per layer, with attention (one FlashAttention varlen call for all prompt chunks) in between | A step reading new prompts: 80 ms down to 25 ms |
| Dozens of tiny kernels per layer | **Fused Triton kernels** for RMSNorm, RoPE and SiLU, and a decode attention kernel that reads each KV head once for all the query heads sharing it | Decode 7% faster with CUDA graphs, 21% without; bit for bit the same output |
| HTTP work slows the engine down | **Engine in its own process**: the web server and the engine never wait on each other's Python | 16-user throughput 532 to 888 tokens/s, together with the two rows above |
| Weights too big or too slow to read | **INT4 and FP8 weights**: 4-bit or 8-bit weights unpacked inside the matmul | Llama 3B: 6.3 GiB to 2.5 GiB, 2.4x faster |
| One word per weight read | **Speculative decoding**: guess several words cheaply, check them all at once | Several words per pass, same output distribution |
| Clients expect OpenAI | **OpenAI-compatible API** with streaming, cancellation, timeouts and load shedding | Works with existing tools |
| Servers are black boxes | **Live dashboard** and Prometheus metrics | Every decision visible |
| You cannot see why the model chose a word | **Token confidence**: OpenAI `logprobs` and `top_logprobs`, drawn in the playground | Every token tinted by certainty, alternatives on hover |

## See it work

```bash
engine serve --model models/Llama-3.2-1B-Instruct
# open http://127.0.0.1:8000/dashboard
```

The dashboard has four live panels:

- **KV cache, block by block.** Every square is one block of GPU memory. Each request
  gets its own color. White squares are shared by several requests that started with
  the same prompt. Dark teal squares belong to finished requests and are kept as a cache
  in case the same prefix comes back.
- **Scheduler, step by step.** One bar per engine step: amber for prompt tokens being
  read, teal for new tokens being generated, a line for how long the step took.
- **Requests in flight.** Each request's phase, progress, how much of its prompt came
  from the cache, and how many times it was paused for memory.
- **Playground.** Send a prompt and watch the answer stream in, each token tinted by how
  sure the model was (green sure, red unsure); hover a token to see the other words it
  considered. Or press **Flood** to fire 16 requests that share a long system prompt and
  watch batching and prefix caching happen in the panels above.

## How it works

One request, from HTTP to text:

```
client  -->  FastAPI (engine/server/api.py)
               validates the request, applies the chat template, streams the reply
                |
                v
             ProcessEngine (engine/server/process_engine.py)
               the engine runs in a process of its own that owns the GPU;
               requests go in over a pipe, each step's tokens come back in one message
                |
                v
             LLMEngine.step()  (engine/runtime/engine.py), in a loop:
               1. Scheduler     which requests run this step, and how many tokens each
               2. BlockManager  hands out KV blocks, finds shared prefixes
               3. model         one forward pass over every request's new tokens
               4. sampler       picks each request's next token
               5. output        turns tokens into text, checks stop rules
                |
                v
client  <--  server-sent events, one chunk per token
```

**The step.** The engine runs in steps. In each step the scheduler gathers the next
tokens of every running request (one token for a request that is generating, a slice of
the prompt for one that is still reading), flattens them into a single sequence, and runs
the model once. A step might hold 40 requests generating plus a 300-token chunk of a new
prompt.

**Paged memory.** The KV cache is one large array of slots, split into blocks of 16
tokens. Each request owns a *block table*: the list of blocks that hold its tokens, in
order, anywhere in memory. Token `i` lives in block `table[i // 16]`, slot `i % 16`. A
request grows one block at a time, so nothing is reserved that is not used. This is the
idea behind vLLM's PagedAttention, the same trick an operating system uses for virtual
memory.

**Sharing prefixes.** When a block fills up, it gets a name: a hash of its 16 tokens
combined with the name of the block before it. Two requests that start with the same
words produce the same chain of names, so the second request simply points at the first
request's blocks and skips that part of the computation. When the last owner finishes, a
named block is not erased. It stays findable until the memory is needed, oldest first.

**Attention over pages.** Generating a token means comparing it against every earlier
token's keys. The Triton kernel in `engine/kernels/paged_attention.py` walks each
request's block table and reads keys and values directly from wherever the blocks are,
computing the softmax as it goes. Llama shares each key/value head between 4 query
heads, so in BF16 one GPU program handles all 4 at once and reads each block only once.

**Fewer launches.** A generation step for a small model is mostly overhead: hundreds of
tiny GPU kernel launches driven from Python, about 44 µs each on this Windows laptop.
The engine records each step once per batch size as a CUDA graph and replays it, so a
whole forward pass becomes one launch. Steps that also read new prompts cannot be one
graph, because their attention changes shape every time, so they are cut at attention:
one graph per layer for everything else, attention run directly in between. And three
chains of small operations (RMSNorm, rotary embeddings, SiLU) are fused into Triton
kernels that round exactly where PyTorch rounds, bit for bit.

**Two processes.** `engine serve` runs the engine in its own process. In one process,
the web server's per-token work (JSON, sockets) and the engine's per-step work take
turns on Python's single interpreter lock, and the engine's step kept waiting for it.

**Smaller weights.** With `--quantization int4`, each weight matrix is stored as 4-bit
numbers with a scale and offset for every group of 128. The GPU kernel unpacks them
inside the matrix multiply, so a quarter of the bytes cross the memory bus. On Llama
3.2 3B that takes peak memory from 6.3 GiB to 2.5 GiB and speed from 36 to 87 tokens per
second. The same arithmetic says Llama 3.1 8B (15 GB in 16-bit) fits an 8 GB card in
INT4; `engine info` prints the estimate, but 8B has not been benchmarked here yet.

**Guess and check.** Speculative decoding (`engine/runtime/speculative.py`) lets a cheap
proposer (a small draft model, or a lookup that finds the current phrase earlier in the
text) guess the next few tokens. The big model checks all guesses in one pass, which
costs about the same as generating one token. A mathematically exact accept/reject rule
keeps the output distribution identical to the big model alone.

Full design notes: [docs/architecture.md](docs/architecture.md).

## How we know it is correct

Speed tricks are easy to get subtly wrong: a block written to the wrong slot, a request
reading another request's memory, an off-by-one in a position. So each optimization is
tested for **identical output** against something simpler.

| What | Checked against |
| --- | --- |
| The model itself | Hugging Face's Llama with the same weights: logits match to 1e-4 |
| Model and sampler, real Llama 3.2 1B | Hugging Face `generate` on 200 prompts, 128 tokens each, FP32: every token identical (Gate 1) |
| Real Llama 3.2 3B in BF16 | Same 200 prompts; any difference must be a near-tie (top two choices within 0.01) |
| Paged cache, batching, chunking, preemption, prefix caching | The simple one-request loop, token for token, plus ~200 randomly generated workloads with arrivals, cancellations and tiny memory |
| Mixed steps on the GPU (kernel decode rows plus prompt chunks) | The simple one-request loop, with staggered arrivals, chunking and prefix hits |
| Batching on the real model | Each prompt run alone, in FP32: 128 of 128 batched requests identical |
| Block allocator | Leak and double-free checks after every operation and every engine step |
| Triton kernel | A plain PyTorch implementation, 12 shapes and dtypes |
| CUDA graphs | The same engine without graphs |
| Mixed-step graphs | The same engine running those steps without graphs, FP32 and BF16, with chunking and prefix hits |
| Fused kernels | The PyTorch operations they replace: bit for bit identical |
| Engine in its own process | The engine in a thread, through the same HTTP API |
| Speculative decoding | Plain greedy (exact), and a chi-square test on 10,000 samples for sampling |
| HTTP API | The engine called directly; streamed text equals non-streamed text |

```bash
pytest                       # 100 CPU tests, about 25 seconds
pytest -m gpu                # 36 GPU tests: Triton kernels, varlen, both kinds of CUDA graph, INT4, FP8
ENGINE_MODEL_DIR=models/Llama-3.2-1B-Instruct pytest -m slow     # the 200-prompt gate
```

Building it this way caught real bugs: a batch losing track of rows when one request
finished mid-step, a deadlock where a waiting request held memory needed by the one in
front of it, and a CUDA graph reading memory that had been freed after capture. The
benchmarks caught one more: when a long prompt shared a step with many generating
requests, attention padded every request to the prompt's length. With 24 requests and a
4,000-token chunk that is gigabytes of scratch memory per layer, enough to push the GPU
into system RAM and trip Windows' 2-second GPU watchdog. Each prompt chunk now gets its
own attention call and generating requests go through the Triton kernel, so memory grows
with the tokens actually present.

## Results

Measured on one **RTX 4060 Laptop GPU (8 GB)**, Windows 11, PyTorch 2.15, CUDA 13.2, on
6 and 7 November 2026. The full report with every table and chart is
**[RESULTS.md](RESULTS.md)** (also as a standalone page,
[docs/results.html](docs/results.html)); every number comes from a JSON file in
[`bench/results/`](bench/results) and can be regenerated with `python bench/run_all.py`.

Something on this machine changed between the evening of 6 November and the evening of
7 November: kernel launches got cheaper and Hugging Face's own outputs shifted in their
last bits (see *What did not work*). The engine's numbers below were all measured on
the evening of 7 November, after the change. vLLM's are from 02:00 that day, and it is
not known whether that was before or after.

### The gates

| Gate | Result | What happened |
| --- | --- | --- |
| **1. Same answers as Hugging Face**, Llama 3.2 1B, FP32 | ✅ **pass** | 200 of 200 prompts identical: 18,964 tokens, not one differs |
| **1. Same answers as Hugging Face**, Llama 3.2 3B, BF16 | ❌ **fail, 193 / 200** | 182 identical, 11 near-ties; 7 split where the top two choices were one or two BF16 rounding steps apart (see below) |
| **2. 64 users for 30 minutes** | ❌ **fail, as written** | 52,728 requests at 2,752 tokens/s: 0 errors, 0 out-of-memory, 0 preemptions, all KV memory returned. Fails only on its strictest check: 160 prompts got a different greedy answer on a repeat (see below) |
| **3. 70% of vLLM's throughput** | ✅ **pass, 113%** | 1,982 vs 1,756 tokens/s at 64 users against vLLM v0.32.0 on the same GPU; it started at 52% (see below) |
| Unit and property tests (CPU) | ✅ pass | 100 passed in 25 s |
| GPU tests (Triton kernels, CUDA graphs, INT4, FP8, varlen) | ✅ pass | 36 passed in 13 s |

**About the 7 failures on 3B.** At six of them, Hugging Face's own top two candidate
tokens were exactly 0.125 apart, at the seventh 0.25. That is one (or two) BF16 rounding
steps for logits of this size, the smallest difference BF16 can represent there, so the
two implementations each rounded a coin flip differently. All seven are in the long
prompt set (280 to 1,415 tokens), where rounding has the most steps to add up. The
gate's rule asked for a gap under 0.01, which BF16 cannot express at this scale, so it
is reported as a failure, as written. The 1B run in FP32, where rounding is 65,000
times finer, matches every token.

### One user: how fast does a reply stream?

Llama 3.2 1B, one request, 256 generated tokens. Each row adds one technique.

| Engine | Tokens/s | ms per token | GPU memory |
| --- | ---: | ---: | ---: |
| Phase 1 loop (contiguous cache, plain PyTorch) | 58.5 | 17.1 | 2.33 GiB |
| Paged cache, PyTorch attention | 42.2 | 23.7 | 2.37 GiB |
| + Triton paged attention kernel | 63.2 | 15.8 | 2.37 GiB |
| + CUDA graphs | **91.4** | 10.9 | 2.50 GiB |
| + FP8 weights | 118.0 | 8.5 | 1.53 GiB |
| + INT4 weights | **194.2** | 5.1 | 1.18 GiB |

![single-user decode speed, 1B](docs/results/decode-Llama-3.2-1B-Instruct.svg)

Llama 3.2 3B: 29.0 tokens/s in the Phase 1 loop, **36.5** with CUDA graphs, 52.0 with
FP8 and **86.7 with INT4** (2.49 GiB instead of 6.34 GiB).

The paged cache on its own is *slower* than the simple loop, because the reference path
copies each request's keys and values out of the blocks before attention. The Triton
kernel removes that copy, and CUDA graphs remove the Python and launch overhead, which
is much of the cost for a model this small.

These are higher than on 6 November, when the same table read 16.6 tokens/s for the
Phase 1 loop and 81.5 with CUDA graphs. Most of that is the machine, not the engine:
with the fused kernels turned off the Phase 1 loop now does 48.5 and CUDA graphs 85.5.
The fused kernels add the rest, 21% and 7%.

### Many users at once

The real server (`engine serve`, 1B, BF16) with N clients that each stream chat
completions back to back for 60 seconds.

| Users | Tokens/s | Requests/s | First token, median | First token, p99 | Per token, median | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 82 | 1.0 | 28 ms | 204 ms | 11 ms | 0 |
| 4 | 290 | 3.1 | 38 ms | 343 ms | 12 ms | 0 |
| 16 | 888 | 9.2 | 53 ms | 319 ms | 14 ms | 0 |
| 32 | 1,419 | 14.6 | 85 ms | 417 ms | 18 ms | 0 |
| 64 | **1,982** | 20.9 | 184 ms | 1,779 ms | 24 ms | 0 |
| 64, for 30 minutes (Gate 2) | **2,752** | 29.3 | 65 ms | 459 ms | 21 ms | 0 |

![throughput](docs/results/load-throughput.svg)

Throughput grows 24x from 1 to 64 users while each user's stream slows only from 11 to
24 ms per token, still about 42 tokens per second each, faster than anyone reads. Zero
errors and zero preemptions at every level, and all KV memory returned at the end of
every run. A 60-second run is noisy at 64 users (an earlier sweep that morning, with a
first RMSNorm kernel, gave 2,247); the 30-minute soak, at 2,752 tokens/s, is the
steadier number.

### Against vLLM (Gate 3)

The same load test against **vLLM v0.32.0**, the most widely used open source LLM server,
in Docker on the same laptop GPU: same model (Llama 3.2 1B, BF16), same 64-sequence
limit, vLLM's defaults otherwise (FlashAttention, `torch.compile`, CUDA graphs, prefix
caching, chunked prefill).

| Users | engine tok/s | vLLM tok/s | engine / vLLM | First token p50, engine / vLLM | Per token p50, engine / vLLM | Per token p99, engine / vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 82 | 86 | 95% | 28 / 31 ms | 11.4 / 10.7 ms | |
| 4 | 290 | 319 | 91% | 38 / 50 ms | 12.2 / 11.3 ms | |
| 16 | 888 | 990 | 90% | 53 / 88 ms | 14.3 / 12.2 ms | 69 / 43 ms |
| 32 | **1,419** | 1,360 | **104%** | 85 / 168 ms | 17.6 / 13.9 ms | 69 / 71 ms |
| 64 | **1,982** | 1,756 | **113%** | **184 / 414 ms** | 23.8 / 19.8 ms | 134 / 138 ms |

![engine vs vLLM](docs/results/vs-vllm.svg)

The gate asks for 70% at 64 users. The engine reaches **113%**: it serves more tokens
per second than vLLM from 32 users up, and its first token arrives sooner at every
load. It took three rounds of profiling to get here, starting at 52%.

vLLM still has the faster *typical* step (12.2 against 14.3 ms per token at 16 users)
and a steadier tail at 16 users. Where the engine wins is getting new requests started:
at 64 users its first token comes in 184 ms against vLLM's 414, and a whole request
takes 3.01 s on average against 3.44 s.

vLLM was measured at 02:00 on 7 November, possibly before the machine change described
above. vLLM replays CUDA graphs for every step, so cheaper kernel launches should help
it less than they helped the engine's slower paths, but a rerun on the same evening is
still owed (Docker Desktop would not start).

**Round 1: 52% to 77%.** With 64 users someone finishes almost every other step and the
next request arrives. Reading even a few prompt tokens forced the whole step off the CUDA
graph onto the slow path, about 85 ms however small it was, against 20 ms for a graph
step. *Prompt coalescing* made newcomers wait a few steps so they could be read together,
and *one FlashAttention varlen call* per layer replaced one attention call per prompt.

**Round 2: why was the slow path slow?** A profile of one such step: 14 ms of GPU work,
77 ms of CPU time, almost all of it issuing 772 kernel launches at 44 µs each (Windows'
display driver model makes launches expensive). So those steps now run as graphs too,
cut at attention, whose shape changes every step: one graph per layer for the norm,
projections, rotary embedding, cache write and MLP, and attention launched directly in
between. 17 replays instead of 772 launches: **80 ms down to 25 ms**. Coalescing then
only delayed prompts, so it is off by default now.

**Round 3: the GPU and the web server.** With the slow steps gone, a decode step itself
was the cost. About 35 tiny element-wise kernels per layer (norm, rotary, activation)
became a few fused Triton kernels, and decode attention now reads each key/value block
once for the 4 query heads that share it instead of 4 times: **7% off every decode
step**, bit for bit the same output. Last, driving the engine with no HTTP gave 1,094
tokens/s at 16 users, but through the server only 532. The web server's per-token Python and the engine's per-step
Python were taking turns on one interpreter lock. Handing a step's tokens over in one
wake-up instead of one per token, then moving the engine into its own process, closed
that gap.

| Change | 16 users | 64 users | vs vLLM at 64 |
| --- | ---: | ---: | ---: |
| First measurement | 535 | 911 | 52% |
| + prompt coalescing, one varlen call (round 1) | 534 | 1,360 | 77% |
| + mixed-step graphs (round 2) | 532 | 1,300 | 74% |
| + fused kernels, GQA attention, engine process (round 3) | **888** | **1,982** | **113%** |

Through HTTP, round 2 only shows up once round 3 removes the server bottleneck in front
of it: driving the engine directly, round 2 alone took 16 users from 811 to 1,094
tokens/s. Each change is tested for identical output against the slower path it
replaces (below), and the FP32 batching test still gives 128 of 128 identical answers.
The first round 3 sweep, that morning, reached 2,247 (128%), but with an RMSNorm kernel
that changed answers (see *What did not work*); this is the sweep with the exact one.

### Batching changes rounding, not math

Under load, a greedy prompt sent twice can come back with a different answer. That would
be a serious bug if the engine mixed up requests, so it gets its own test
(`bench/batch_invariance.py`): 64 prompts, each run alone, then all of them twice over
in batches of up to 64 with prompts arriving mid-flight and the prefix cache on.

| Precision | Loaded requests | Same tokens as running alone | Different | Largest top-2 gap where they split |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 128 | **128** | 0 | |
| BF16 | 128 | 71 | 57 | 0.25 |

In FP32 every token matches, so batching, paging and caching compute exactly the right
thing. In BF16 more than a third of the answers drift, and every drift starts at a token
where the model's top two choices were within 0.25 of each other (one or two BF16
rounding steps): a different batch size changes the order numbers are added in, and that rounds a coin
flip the other way. vLLM does exactly the same: in the Gate 3 runs it returned a
different greedy answer on repeat for 117 prompts at 64 users. Gate 2's
strict "same answer every time" rule cannot pass in BF16 for that reason, so it is
reported as failed rather than relaxed after the fact.

### Quality: what do smaller weights cost?

Perplexity on WikiText-2 (lower is better).

| Model | BF16 | FP8 | INT4 (groups of 128) |
| --- | ---: | ---: | ---: |
| Llama 3.2 1B | 13.17, 2.62 GiB | 13.33 (+0.16), 1.71 GiB | 15.84 (+2.67), 1.29 GiB |
| Llama 3.2 3B | 11.05, 6.31 GiB | 11.14 (+0.09), 3.68 GiB | 12.01 (+0.96), 2.45 GiB |

FP8 is close to free. INT4 costs more, and costs the small model the most; INT4 3B is
both smaller and better than BF16 1B (12.01 vs 13.17, 2.45 vs 2.62 GiB).

### What did not work (yet)

- **Speculative decoding is correct but slower here.** It passes the exactness and
  chi-square tests, and a 1B draft gets 56% of its guesses accepted on 3B. But the
  verify pass runs without CUDA graphs, so on 3B it reaches 25.0 tokens/s against 36.5
  for plain decoding with graphs. Capturing the verify step in a graph is the next step.
- **Repeat answers under load** differ in BF16 (above). Making BF16 batch-invariant
  needs kernels that add in a fixed order whatever the batch size, at some cost in
  speed.
- **Two gates fail as written**: Gate 1 on 3B and Gate 2, both only on BF16 rounding
  ties. Their rules should measure gaps in BF16 rounding steps rather than a fixed 0.01.
- **vLLM still has the faster typical step**, 6 to 27% per token at every load, and is
  ahead at 1 to 16 users (90 to 95%). The rest is GPU work: vLLM's `torch.compile`
  fuses more (residual adds into the norms, the q/k/v and gate/up projections into one
  multiply each). Merging projections was measured here and saved 5% of the multiply
  time at 16 users while changing BF16 rounding, so it is not in yet.
- **A test oracle went stale, and for a few hours it hid a real bug.** Gate 1 compares
  against Hugging Face outputs cached on disk. On 7 November the same code that matched
  all 200 the day before "failed" 41: Hugging Face itself, rerun, no longer produced its
  own cached answers (on 3B in BF16, 82 of 200 changed), while the engine matched the
  fresh run exactly. Something below PyTorch (same driver, same libraries) now rounds
  differently. Against the fresh 3B reference, though, the first fused RMSNorm kernel
  failed 27 prompts: it added up its squares in a different order than PyTorch. On
  random data that had looked harmless (99.9% of outputs equal), but over 128 greedy
  tokens it was enough. PyTorch now computes that sum and the kernel does the rest, bit
  for bit, and 3B is back to 7 rounding ties. Lessons: test fused kernels for exact
  equality, not "almost always", and give the cache a fingerprint of the GPU's own
  output so it notices drift by itself.

## Quickstart

You need Python 3.11+, an NVIDIA GPU (compute capability 8.0 or newer) and Linux, WSL2
or Windows.

```bash
git clone https://github.com/AashutoshR/inference-engine && cd inference-engine
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,serve,eval]"
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu132
pip install triton-windows                             # Windows only
```

Download a model (`--source unsloth` uses ungated copies of Meta's weights, so no
Hugging Face login is needed):

```bash
python scripts/download_models.py --source unsloth             # Llama 3.2 1B
python scripts/download_models.py --source unsloth --also 3b   # and 3B
```

Serve it and talk to it with the OpenAI SDK:

```bash
engine serve --model models/Llama-3.2-1B-Instruct
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "What is a KV cache?"}],
    stream=True,
)
for chunk in reply:
    print(chunk.choices[0].delta.content or "", end="")
```

Or run it in Docker:

```bash
docker build -t engine .
docker run --gpus all -p 8000:8000 -v "$PWD/models:/models" engine --model /models/Llama-3.2-1B-Instruct
```

### Useful options

| Option | Default | What it does |
| --- | --- | --- |
| `--quantization int4` / `fp8` | none | Smaller, faster weights (FP8 needs an RTX 40xx or newer) |
| `--max-num-seqs` | 64 | Requests that can run in the same step |
| `--max-num-batched-tokens` | 4096 | Token budget per step; long prompts are chunked to fit |
| `--max-waiting` | 256 | Queue length before new requests get HTTP 429 |
| `--timeout` | 600 | Seconds before a request is cancelled |
| `--gpu-memory-utilization` | 0.9 | Share of GPU memory the KV cache may use |
| `--no-prefix-caching`, `--no-chunked-prefill`, `--no-cuda-graphs`, `--no-mixed-graphs` | | Turn a feature off to see what it was worth |
| `--in-process-engine` | off | Run the engine in a thread of the web server's process |
| `ENGINE_FUSED_KERNELS=0` | | Environment variable: plain PyTorch instead of the fused kernels |

API: `POST /v1/chat/completions` and `POST /v1/completions` (streaming or not),
`GET /v1/models`, `GET /dashboard`, `GET /debug/state`, `GET /metrics`, `GET /health`.
Beyond OpenAI's fields it accepts `top_k`, `min_p` and `repetition_penalty`.

Other tools:

```bash
engine info --model models/Llama-3.1-8B-Instruct --gpu-mem 8   # will it fit, and how
engine generate --model models/Llama-3.2-1B-Instruct --prompt "Hello"
python bench/bench_decode.py --model M --mode graphs           # single-request speed
python bench/load_test.py --concurrency 64 --duration 60       # many users at once
python bench/perplexity.py --model M --quant int4              # does quantization hurt quality?
```

## What is inside

```
engine/
  model/        the Llama network in plain PyTorch, plus a loader that can quantize on the fly
  runtime/      the engine: scheduler, block manager, paged cache, CUDA graphs, sampler,
                speculative decoding, and the original one-request loop kept as a test oracle
  kernels/      Triton kernels: paged attention, fused RMSNorm, RoPE and SiLU
  quant/        INT4 and FP8 linear layers
  server/       the OpenAI-compatible API, the engine process, the dashboard
tests/          unit, property, GPU and golden tests
bench/          speed, load, soak and perplexity benchmarks; results/ holds the numbers
docs/           architecture notes
scripts/        model download, test-prompt builder
```

About 3,100 lines of Python and Triton, plus a 400-line dashboard. No vLLM, no FlashInfer, no custom C++: PyTorch,
Triton and the standard library.

## Status

Built in five phases, each closed by a measurable gate:

1. **Correct baseline.** Own forward pass and sampler. Gate: identical to Hugging Face on 200 prompts. *Passed for 1B in FP32 (200/200); 3B in BF16 at 193/200, the 7 being one- or two-step rounding ties.*
2. **Serving core.** Paged cache, continuous batching, Triton kernel, OpenAI API. Gate: 64 concurrent streams for 30 minutes with no errors or leaks. *Ran: 0 errors, 0 leaks, 52,728 requests at 2,752 tokens/s; fails only the strict repeat-answer check, which BF16 rounding breaks.*
3. **Throughput and latency.** Chunked prefill, prefix caching, preemption, CUDA graphs, mixed-step graphs, fused kernels, a separate engine process. Gate: 70% of vLLM's throughput on the same GPU. *Passed: 113% of vLLM v0.32.0 at 64 users (52% at the first measurement).*
4. **Quantization and speculation.** INT4, FP8, speculative decoding.
5. **Hardening.** Load and soak tests, metrics, dashboard, Docker, docs.

Model weights are under Meta's Llama license.
