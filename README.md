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
proven correct: the engine must produce exactly the same words as Hugging Face's
reference implementation, token for token, on 200 test prompts.

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
| Weights too big or too slow to read | **INT4 and FP8 weights**: 4-bit or 8-bit weights unpacked inside the matmul | 4x less weight memory, 8B models fit on 8 GB |
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
             AsyncEngine (engine/server/async_engine.py)
               one background thread owns the GPU; HTTP handlers never touch it
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
token's keys. The Triton kernel in `engine/kernels/paged_attention.py` runs one GPU
program per request and attention head, walks the request's block table, and reads keys
and values directly from wherever the blocks are, computing the softmax as it goes.

**Fewer launches.** A generation step for a small model is mostly overhead: hundreds of
tiny GPU kernel launches driven from Python. The engine records each step once per batch
size as a CUDA graph and replays it, so a whole forward pass becomes one launch.

**Smaller weights.** With `--quantization int4`, each weight matrix is stored as 4-bit
numbers with a scale and offset for every group of 128. The GPU kernel unpacks them
inside the matrix multiply, so a quarter of the bytes cross the memory bus. This is what
lets Llama 3.1 8B (15 GB in 16-bit) run on an 8 GB card.

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
| End to end, real Llama 3.2 1B | Hugging Face `generate` on 200 prompts, 128 tokens each, FP32: every token identical |
| Real Llama 3.2 3B in BF16 | Same 200 prompts; any difference must be a near-tie (top two choices within 0.01) |
| Paged cache, batching, chunking, preemption, prefix caching | The simple one-request loop, token for token, plus ~200 randomly generated workloads with arrivals, cancellations and tiny memory |
| Block allocator | Leak and double-free checks after every operation and every engine step |
| Triton kernel | A plain PyTorch implementation, 12 shapes and dtypes |
| CUDA graphs | The same engine without graphs |
| Speculative decoding | Plain greedy (exact), and a chi-square test on 10,000 samples for sampling |
| HTTP API | The engine called directly; streamed text equals non-streamed text |

```bash
pytest                       # 98 CPU tests, about 40 seconds
pytest -m gpu                # 21 GPU tests: Triton kernel, CUDA graphs, INT4, FP8
ENGINE_MODEL_DIR=models/Llama-3.2-1B-Instruct pytest -m slow     # the 200-prompt gate
```

Building it this way caught real bugs: a batch losing track of rows when one request
finished mid-step, a deadlock where a waiting request held memory needed by the one in
front of it, and a CUDA graph reading memory that had been freed after capture.

## Results

Measured on an RTX 4060 Laptop GPU (8 GB), Windows 11, PyTorch 2.15, CUDA 13.2.
Raw numbers are in [`bench/results/`](bench/results).

*Benchmarks are being collected; this table fills in as they finish.*

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
| `--no-prefix-caching`, `--no-chunked-prefill`, `--no-cuda-graphs` | | Turn a feature off to see what it was worth |

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
  kernels/      the Triton paged-attention kernel
  quant/        INT4 and FP8 linear layers
  server/       the OpenAI-compatible API, the background engine thread, the dashboard
tests/          unit, property, GPU and golden tests
bench/          speed, load, soak and perplexity benchmarks; results/ holds the numbers
docs/           architecture notes
scripts/        model download, test-prompt builder
```

About 3,100 lines of Python and Triton, plus a 400-line dashboard. No vLLM, no FlashInfer, no custom C++: PyTorch,
Triton and the standard library.

## Status

Built in five phases, each closed by a measurable gate:

1. **Correct baseline.** Own forward pass and sampler. Gate: identical to Hugging Face on 200 prompts. *Passed for 1B in FP32.*
2. **Serving core.** Paged cache, continuous batching, Triton kernel, OpenAI API. Gate: 64 concurrent streams for 30 minutes with no errors or leaks.
3. **Throughput and latency.** Chunked prefill, prefix caching, preemption, CUDA graphs. Gate: 70% of vLLM's throughput on the same GPU.
4. **Quantization and speculation.** INT4, FP8, speculative decoding.
5. **Hardening.** Load and soak tests, metrics, dashboard, Docker, docs.

Model weights are under Meta's Llama license.
