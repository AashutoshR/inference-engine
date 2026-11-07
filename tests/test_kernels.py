"""Phase 2: the Triton paged-attention kernel against a PyTorch reference (GPU only)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.kernels import paged_attention as pa  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not pa.available(torch.device("cuda")), reason="needs CUDA and Triton"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("nh", "nkv", "hd"), [(32, 8, 64), (24, 8, 128), (4, 2, 16)])
@pytest.mark.parametrize("block_size", [4, 16])
def test_paged_decode_matches_reference(dtype, nh, nkv, hd, block_size):
    torch.manual_seed(0)
    dev = "cuda"
    ctx = torch.tensor([1, 17, 200, 64, 333], dtype=torch.int32, device=dev)
    b = len(ctx)
    max_blocks = -(-int(ctx.max()) // block_size)
    num_blocks = b * max_blocks + 3
    # Scatter each sequence's blocks randomly over the cache, as a real allocator would.
    perm = torch.randperm(num_blocks, device=dev)[: b * max_blocks].int()
    tables = perm.view(b, max_blocks).contiguous()
    k = torch.randn(num_blocks * block_size, nkv, hd, device=dev, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(b, nh, hd, device=dev, dtype=dtype)

    got = pa.paged_decode_attention(q, k, v, tables, ctx, block_size)
    want = pa.reference_decode_attention(q, k, v, tables, ctx, block_size)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4  # plan: max abs error under 1e-2 in BF16
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)


def test_engine_with_triton_matches_static_loop():
    """The whole decode path on the GPU: same tokens as the Phase 1 loop."""
    from engine.config import ModelConfig
    from engine.model.llama import LlamaForCausalLM
    from engine.runtime.engine import LLMEngine
    from engine.runtime.generate import generate
    from engine.runtime.params import SamplingParams
    from tests.conftest import TINY_RAW
    from tests.test_engine import PROMPTS
    from tests.test_generate import ByteTokenizer

    torch.manual_seed(0)
    model = LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).cuda().eval()
    params = SamplingParams(temperature=0, max_tokens=20)
    want = {
        rid: [o.token_id for o in generate(model, ByteTokenizer(), [p], params, ())]
        for rid, p in PROMPTS.items()
    }
    eng = LLMEngine(model, ByteTokenizer(), (), num_blocks=64, block_size=4, use_triton=True)
    assert eng.generate(PROMPTS, params) == want


@pytest.mark.parametrize("graphs", [False, True])
def test_cuda_graphs_match_eager(graphs):
    """Three sequences pad up to the 4-bucket; outputs must not change."""
    from engine.config import ModelConfig
    from engine.model.llama import LlamaForCausalLM
    from engine.runtime.engine import LLMEngine
    from engine.runtime.generate import generate
    from engine.runtime.params import SamplingParams
    from tests.conftest import TINY_RAW
    from tests.test_engine import PROMPTS
    from tests.test_generate import ByteTokenizer

    torch.manual_seed(0)
    model = LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).cuda().eval()
    prompts = dict(list(PROMPTS.items())[:3])
    params = SamplingParams(temperature=0, max_tokens=30)
    want = {
        rid: [o.token_id for o in generate(model, ByteTokenizer(), [p], params, ())]
        for rid, p in prompts.items()
    }
    eng = LLMEngine(
        model, ByteTokenizer(), (), num_blocks=64, block_size=4, max_num_seqs=8,
        use_triton=True, enable_cuda_graphs=graphs, debug_checks=True,
    )  # fmt: skip
    assert (eng.graphs is not None) == graphs
    assert eng.generate(prompts, params) == want
    assert eng.num_free_blocks == 64


@pytest.mark.parametrize("budget", [4096, 8])
def test_mixed_steps_match_static_loop(budget, monkeypatch):
    """Prompts join while others decode, get chunked (budget 8) and hit the prefix cache:
    every step mixes kernel decode rows with per-chunk SDPA. Tokens must not change."""
    from engine.config import ModelConfig
    from engine.model.llama import LlamaForCausalLM
    from engine.runtime.engine import LLMEngine
    from engine.runtime.generate import generate
    from engine.runtime.paged_cache import PagedBatch
    from engine.runtime.params import SamplingParams
    from tests.conftest import TINY_RAW
    from tests.test_generate import ByteTokenizer

    torch.manual_seed(0)
    model = LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).cuda().eval()
    shared = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    prompts = {"a": [5, 6, 7], "b": shared + [1, 2], "c": [99], "d": shared + [3],
               "e": list(range(20, 41)), "f": shared}  # fmt: skip
    params = SamplingParams(temperature=0, max_tokens=12)
    want = {
        rid: [o.token_id for o in generate(model, ByteTokenizer(), [p], params, ())]
        for rid, p in prompts.items()
    }
    calls = []
    real = PagedBatch._mixed.__func__
    monkeypatch.setattr(PagedBatch, "_mixed",
                        classmethod(lambda cls, *a: calls.append(1) or real(cls, *a)))  # fmt: skip
    eng = LLMEngine(
        model, ByteTokenizer(), (), num_blocks=64, block_size=4, max_num_seqs=8,
        max_num_batched_tokens=budget, use_triton=True, enable_cuda_graphs=False,
        debug_checks=True,
    )  # fmt: skip
    got = {rid: [] for rid in prompts}
    pending = list(prompts.items())
    while pending or eng.has_unfinished():
        if pending:
            rid, p = pending.pop(0)
            eng.add_request(rid, p, params)
        for out in eng.step():
            got[out.request_id].append(out.token_id)
    assert got == want
    assert calls, "no mixed step happened"
    assert eng.cached_prompt_tokens_total > 0
    assert eng.num_free_blocks == 64


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("budget", [4096, 8])
def test_mixed_step_graphs_match_eager_steps(dtype, budget):
    """Mixed steps replayed as piecewise CUDA graphs (padded to a token bucket, attention
    eager in between) give the same tokens as the same engine running them eagerly:
    prompts join while others decode, get chunked (budget 8) and hit the prefix cache."""
    from engine.config import ModelConfig
    from engine.model.llama import LlamaForCausalLM
    from engine.runtime.engine import LLMEngine
    from engine.runtime.params import SamplingParams
    from tests.conftest import TINY_RAW
    from tests.test_generate import ByteTokenizer

    torch.manual_seed(0)
    model = LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).cuda().to(dtype).eval()
    shared = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    prompts = {"a": [5, 6, 7], "b": shared + [1, 2], "c": [99], "d": shared + [3],
               "e": list(range(20, 41)), "f": shared, "g": list(range(50, 90))}  # fmt: skip
    params = SamplingParams(temperature=0, max_tokens=12)

    def run(mixed_graphs):
        eng = LLMEngine(
            model, ByteTokenizer(), (), num_blocks=64, block_size=4, max_num_seqs=8,
            max_num_batched_tokens=budget, use_triton=True, enable_cuda_graphs=True,
            enable_mixed_graphs=mixed_graphs, coalesce_prefill=False, debug_checks=True,
        )  # fmt: skip
        got = {rid: [] for rid in prompts}
        pending = list(prompts.items())
        while pending or eng.has_unfinished():
            if pending:
                rid, p = pending.pop(0)
                eng.add_request(rid, p, params)
            for out in eng.step():
                got[out.request_id].append(out.token_id)
        assert eng.num_free_blocks == 64
        return got, eng

    want, eager = run(False)
    got, eng = run(True)
    assert got == want
    assert not any(s["mixed_graph"] for s in eager.step_log)
    assert sum(s["mixed_graph"] for s in eng.step_log) >= 3
    assert eng.cached_prompt_tokens_total > 0


def test_varlen_prompt_chunks_match_per_chunk_sdpa(monkeypatch):
    """BF16 mixed step: FlashAttention varlen over all prompt chunks gives the same
    attention as one SDPA per chunk (the FP32 path), to BF16 rounding."""
    from engine.config import ModelConfig
    from engine.model.layers import Attention
    from engine.runtime import paged_cache
    from engine.runtime.paged_cache import PagedBatch, PagedKVCache
    from tests.conftest import TINY_RAW

    if not paged_cache.varlen_available(torch.zeros(1, device="cuda", dtype=torch.bfloat16)):
        pytest.skip("this torch has no FlashAttention varlen")
    torch.manual_seed(0)
    cfg = ModelConfig.from_dict(TINY_RAW)
    cache = PagedKVCache(cfg, 64, 4, "cuda", torch.bfloat16)
    cache.k.normal_()
    cache.v.normal_()
    attn = Attention(cfg, 0).cuda()
    # (new tokens, context): decode rows, a fresh prompt, chunks after a prefix hit
    spec = [(1, 9), (6, 6), (1, 3), (5, 13), (3, 20), (1, 1)]
    tables, nxt = [], 0
    for _, ctx in spec:
        n = -(-ctx // 4)
        tables.append(list(range(nxt, nxt + n)))
        nxt += n
    qlens, ctxs = [q for q, _ in spec], [c for _, c in spec]
    q = torch.randn(cfg.num_attention_heads, sum(qlens), cfg.head_dim, device="cuda",
                    dtype=torch.bfloat16)  # fmt: skip
    fast = PagedBatch.build(cache, tables, ctxs, qlens, "cuda", use_kernel=True)
    assert fast.mixed and fast.varlen is not None and not fast.chunks
    monkeypatch.setattr(paged_cache, "varlen_available", lambda t: False)
    slow = PagedBatch.build(cache, tables, ctxs, qlens, "cuda", use_kernel=True)
    assert slow.varlen is None and len(slow.chunks) == 3
    got, want = attn._mixed(q, fast).float(), attn._mixed(q, slow).float()
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2), (got - want).abs().max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_rope_and_silu_are_bitwise_identical(dtype):
    """The fused kernels round exactly where the PyTorch ops do."""
    import torch.nn.functional as F

    from engine.kernels import fused
    from engine.model.layers import apply_rotary

    torch.manual_seed(0)
    n, nh, hd = 77, 8, 64
    q = torch.randn(1, n, nh, hd, device="cuda").to(dtype)
    pos = torch.arange(n, device="cuda")[None] * 37
    inv = 1.0 / (500000 ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    freqs = pos[..., None].float() * inv
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos(), emb.sin()
    want, _ = apply_rotary(q.transpose(1, 2), q.transpose(1, 2), cos, sin)
    got = q[0].clone()
    fused.rope_(got, cos[0], sin[0])
    assert torch.equal(want.transpose(1, 2)[0], got)

    gate = (torch.randn(64, 1000, device="cuda") * 4).to(dtype)
    up = torch.randn(64, 1000, device="cuda").to(dtype)
    assert torch.equal(fused.silu_mul(gate, up), F.silu(gate) * up)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [2048, 3072])
def test_fused_rms_norm_is_bitwise_identical(dtype, dim):
    from engine.kernels import fused

    torch.manual_seed(0)
    x = (torch.randn(300, dim, device="cuda") * 3).to(dtype)
    w = torch.randn(dim, device="cuda").to(dtype)
    xf = x.float()
    want = w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)
    assert torch.equal(fused.rms_norm(x, w, 1e-5), want)
