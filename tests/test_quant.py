"""Phase 4: INT4 weight-only quantization (GPU: the kernel is CUDA-only)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from engine.quant.int4 import Int4Linear, dequantize, quantize_rtn  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_rtn_roundtrip_error_is_within_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    codes, sz = quantize_rtn(w, 64)
    assert codes.min() >= 0 and codes.max() <= 15
    step = sz[..., 0].t().float().repeat_interleave(64, dim=1)  # [N, K]
    err = (dequantize(codes, sz, 64) - w).abs()
    assert (err <= step * 0.5 + 1e-2).all()


@cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("n", "k", "g"), [(256, 512, 128), (512, 2048, 128), (64, 64, 32)])
def test_int4_linear_matches_dequantized_matmul(n, k, g):
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    layer = Int4Linear(w, g)
    x = torch.randn(3, 5, k, device="cuda", dtype=torch.bfloat16)
    codes, sz = quantize_rtn(w, g)
    want = x.float() @ dequantize(codes, sz, g).t()
    got = layer(x).float()
    rel = (got - want).abs().max() / want.abs().mean()
    assert rel < 2e-2, rel


@cuda
@pytest.mark.gpu
def test_int4_model_loads_and_tracks_bf16(tmp_path):
    transformers = pytest.importorskip("transformers")
    from engine.model.loader import load_model
    from tests.conftest import TINY_RAW
    from tests.test_layers import _hf_kwargs

    torch.manual_seed(0)
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**_hf_kwargs())).save_pretrained(
        tmp_path
    )
    (tmp_path / "config.json").write_text(json.dumps(TINY_RAW))
    ref = load_model(tmp_path, device="cuda")
    q = load_model(tmp_path, device="cuda", quantization="int4", group_size=32)
    assert isinstance(q.model.layers[0].self_attn.q_proj, Int4Linear)
    assert not isinstance(q.lm_head, Int4Linear)

    ids = torch.randint(0, TINY_RAW["vocab_size"], (1, 16), device="cuda")
    pos = torch.arange(16, device="cuda")[None]
    a, b = ref(ids, pos), q(ids, pos)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    assert cos > 0.98, cos


fp8_ok = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs an FP8 GPU (sm_89+)",
)


@fp8_ok
@pytest.mark.gpu
def test_fp8_linear_tracks_bf16():
    from engine.quant.fp8 import Fp8Linear

    torch.manual_seed(0)
    w = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16)
    want = x.float() @ w.float().t()
    got = Fp8Linear(w)(x).float()
    rel = (got - want).norm() / want.norm()
    assert rel < 0.05, rel


@fp8_ok
@pytest.mark.gpu
def test_fp8_model_loads(tmp_path):
    transformers = pytest.importorskip("transformers")
    from engine.model.loader import load_model
    from engine.quant.fp8 import Fp8Linear
    from tests.conftest import TINY_RAW
    from tests.test_layers import _hf_kwargs

    torch.manual_seed(0)
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**_hf_kwargs())).save_pretrained(
        tmp_path
    )
    (tmp_path / "config.json").write_text(json.dumps(TINY_RAW))
    ref = load_model(tmp_path, device="cuda")
    q = load_model(tmp_path, device="cuda", quantization="fp8")
    assert isinstance(q.model.layers[1].mlp.down_proj, Fp8Linear)
    ids = torch.randint(0, TINY_RAW["vocab_size"], (1, 16), device="cuda")
    pos = torch.arange(16, device="cuda")[None]
    cos = torch.nn.functional.cosine_similarity(ref(ids, pos).flatten(), q(ids, pos).flatten(), 0)
    assert cos > 0.99, cos
