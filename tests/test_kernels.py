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
