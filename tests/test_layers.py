"""Week 1 definition of done. These fail with NotImplementedError until each piece exists.

Run one at a time while you work, e.g. ``pytest tests/test_layers.py -k rmsnorm``.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from engine.model.layers import MLP, RMSNorm, RotaryEmbedding, apply_rotary  # noqa: E402
from engine.model.llama import LlamaForCausalLM  # noqa: E402
from engine.runtime.kv_cache import ContiguousKVCache  # noqa: E402
from tests.conftest import TINY_RAW  # noqa: E402

# --- Test oracles: straight transcriptions of the HF Llama math -------------------------


def ref_inv_freq(cfg) -> torch.Tensor:
    dim, base = cfg.head_dim, cfg.rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    s = cfg.rope_scaling
    if s is None:
        return inv_freq
    low_wavelen = s.original_max_position_embeddings / s.low_freq_factor
    high_wavelen = s.original_max_position_embeddings / s.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / s.factor, inv_freq)
    smooth = (s.original_max_position_embeddings / wavelen - s.low_freq_factor) / (
        s.high_freq_factor - s.low_freq_factor
    )
    smoothed = (1 - smooth) * scaled / s.factor + smooth * scaled
    is_medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# --- Layers -----------------------------------------------------------------------------


def test_rmsnorm_matches_formula():
    torch.manual_seed(0)
    norm = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    xf = x.float()
    expected = norm.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)
    torch.testing.assert_close(norm(x), expected)


def test_rotary_cos_sin_match_llama3_scaling(tiny_cfg):
    rope = RotaryEmbedding(tiny_cfg)
    positions = torch.arange(10).unsqueeze(0)
    cos, sin = rope(positions)
    freqs = positions[..., None].float() * ref_inv_freq(tiny_cfg)
    emb = torch.cat((freqs, freqs), dim=-1)
    torch.testing.assert_close(cos, emb.cos())
    torch.testing.assert_close(sin, emb.sin())


def test_apply_rotary_uses_rotate_half_layout():
    torch.manual_seed(0)
    q, k = torch.randn(1, 4, 3, 16), torch.randn(1, 2, 3, 16)
    cos, sin = torch.randn(1, 3, 16), torch.randn(1, 3, 16)
    q_out, k_out = apply_rotary(q, k, cos, sin)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    torch.testing.assert_close(q_out, q * c + rotate_half(q) * s)
    torch.testing.assert_close(k_out, k * c + rotate_half(k) * s)


def test_mlp_is_swiglu(tiny_cfg):
    torch.manual_seed(0)
    mlp = MLP(tiny_cfg)
    x = torch.randn(2, 3, tiny_cfg.hidden_size)
    expected = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    torch.testing.assert_close(mlp(x), expected)


def test_kv_cache_update_returns_filled_prefix(tiny_cfg):
    cache = ContiguousKVCache(tiny_cfg, batch_size=1, max_len=8, device="cpu", dtype=torch.float32)
    k1, v1 = torch.randn(1, 2, 3, 16), torch.randn(1, 2, 3, 16)
    k_all, v_all = cache.update(0, k1, v1, start_pos=0)
    assert k_all.shape == (1, 2, 3, 16)
    k2, v2 = torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16)
    k_all, v_all = cache.update(0, k2, v2, start_pos=3)
    torch.testing.assert_close(k_all, torch.cat([k1, k2], dim=2))
    torch.testing.assert_close(v_all, torch.cat([v1, v2], dim=2))


# --- Whole model ------------------------------------------------------------------------


def test_state_dict_names_match_hf(tiny_cfg):
    """Passes today: guards the module names the loader depends on."""
    transformers = pytest.importorskip("transformers")
    hf = transformers.LlamaForCausalLM(transformers.LlamaConfig(**_hf_kwargs()))
    ours = LlamaForCausalLM(tiny_cfg)
    assert set(ours.state_dict()) == set(hf.state_dict())


def test_logits_match_hf_with_same_weights(tiny_cfg):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    hf = transformers.LlamaForCausalLM(transformers.LlamaConfig(**_hf_kwargs())).eval()
    ours = LlamaForCausalLM(tiny_cfg).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)

    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    positions = torch.arange(12).expand(2, -1)
    with torch.no_grad():
        expected = hf(input_ids=ids).logits.float()
    torch.testing.assert_close(ours(ids, positions), expected, atol=1e-4, rtol=1e-4)


def test_prefill_then_decode_equals_full_forward(tiny_cfg):
    torch.manual_seed(0)
    model = LlamaForCausalLM(tiny_cfg).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 10))
    full = model(ids, torch.arange(10).unsqueeze(0))

    cache = ContiguousKVCache(tiny_cfg, 1, max_len=16, device="cpu", dtype=torch.float32)
    step_logits = [model(ids[:, :6], torch.arange(6).unsqueeze(0), cache, start_pos=0)[:, -1]]
    for pos in range(6, 10):
        out = model(ids[:, pos : pos + 1], torch.tensor([[pos]]), cache, start_pos=pos)
        step_logits.append(out[:, -1])
    torch.testing.assert_close(torch.stack(step_logits, dim=1), full[:, 5:], atol=1e-4, rtol=1e-4)


def _hf_kwargs() -> dict:
    keep = [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "tie_word_embeddings",
        "bos_token_id",
        "eos_token_id",
    ]
    return {k: TINY_RAW[k] for k in keep} | {"attention_bias": False, "mlp_bias": False}
