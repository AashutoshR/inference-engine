# engine

An LLM inference server built from scratch: own forward pass, paged KV cache,
continuous batching, custom kernels and an OpenAI-compatible API. The target is
70% of vLLM's throughput on the same GPU.

**Status: Phase 1 (correct baseline).** Model and runtime modules are stubs with
exact signatures; the tests define when each one is done.

## Setup

Use Python 3.12 (the widest PyTorch/Triton wheel support), on Linux or WSL2 if you can.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
pre-commit install
```

Check the GPU and the model download:

```bash
python -c "import torch; print(torch.cuda.get_device_name())"
huggingface-cli login
python scripts/download_models.py
engine info --model models/Llama-3.2-1B-Instruct --gpu-mem 8
```

## Hardware note

The development machine has an RTX 4060 Laptop GPU with 8 GB of memory:

| Model | bf16 weights | Fits in 8 GB? |
| --- | --- | --- |
| Llama 3.2 1B | 2.3 GiB | Yes, with room for a large KV cache |
| Llama 3.2 3B | 6.0 GiB | Barely; small KV cache |
| Llama 3.1 8B | 15.0 GiB | No. Needs INT4 (Phase 4) or a rented 24 GB GPU |

Run `engine info --model <dir>` for exact numbers.

## Layout

```
engine/
  config.py          ModelConfig, stop IDs, memory estimates      (done)
  cli.py             engine info | generate | serve               (info works now)
  model/
    layers.py        RMSNorm, RoPE, attention, MLP, decoder layer  Week 1
    llama.py         LlamaForCausalLM, HF-compatible names         Week 1
    loader.py        safetensors -> model                          Week 1
  runtime/
    params.py        SamplingParams                                (done)
    kv_cache.py      contiguous KV cache                           Week 2
    sampler.py       greedy, temperature, top-k/p, min-p           Week 2
    detokenizer.py   UTF-8-safe streaming                          Week 2
    generate.py      static-batch generation loop                  Week 2
  kernels/           Triton/CUDA kernels                           Phase 2
tests/               unit tests + golden test against HF           Week 3
bench/               baseline benchmark, results committed         Week 3
scripts/             model download
```

## Working through Phase 1

Work test by test. Each stub raises `NotImplementedError` until you write it:

```bash
pytest tests/test_layers.py -k rmsnorm          # Week 1, one piece at a time
pytest tests/test_layers.py                     # Week 1 done when all pass
pytest tests/test_sampler.py                    # Week 2
ENGINE_MODEL_DIR=models/Llama-3.2-1B-Instruct pytest -m slow    # Gate 1
python bench/bench_baseline.py --model models/Llama-3.2-1B-Instruct
```

`test_logits_match_hf_with_same_weights` is the most useful test. It loads Hugging
Face's own Llama weights into your model and compares the logits. Once it passes,
your forward pass is correct.

## Continuous integration

GitHub Actions runs lint and the CPU tests on every push. GPU and model-weight tests
(`-m gpu`, `-m slow`) run locally before merging. CI stays red until the Week 1 tests
pass. That's intended: the tests define the work.
