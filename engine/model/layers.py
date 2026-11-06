"""Transformer building blocks (Phase 1, Week 1).

Module and parameter names deliberately mirror Hugging Face's Llama
(``self_attn.q_proj``, ``mlp.gate_proj``, ``input_layernorm`` ...), so a HF
state dict loads with no renaming.

Shapes: B = batch, T = tokens in this step, H = hidden size,
nh = query heads, nkv = KV heads, hd = head dim.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from engine.config import ModelConfig
from engine.runtime.kv_cache import ContiguousKVCache
from engine.runtime.paged_cache import PagedBatch


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim]. Normalize in float32, cast back to x.dtype, then scale by weight.

        Match HF exactly: weight * (x_fp32 * rsqrt(mean(x_fp32^2) + eps)).to(x.dtype)
        """
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normed.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE inverse frequencies (with Llama 3 scaling) and returns cos/sin."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # persistent=False keeps it out of the state dict, so HF checkpoints load strictly.
        self.register_buffer("inv_freq", self.compute_inv_freq(cfg), persistent=False)

    @staticmethod
    def compute_inv_freq(cfg: ModelConfig) -> torch.Tensor:
        dim, base = cfg.head_dim, cfg.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        s = cfg.rope_scaling
        if s is None:
            return inv_freq
        # Llama 3: long wavelengths are slowed down by `factor`, short ones are kept,
        # and the band between is interpolated smoothly.
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

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [B, T] int64 -> (cos, sin), each [B, T, hd], in float32."""
        freqs = positions[..., None].float() * self.inv_freq.float()  # [B, T, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: [B, nh, T, hd], k: [B, nkv, T, hd]. Use HF's rotate_half layout, not interleaved pairs."""
    cos = cos.unsqueeze(1).to(q.dtype)  # [B, 1, T, hd], broadcasts over heads
    sin = sin.unsqueeze(1).to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_attention_mask(
    q_len: int,
    start_pos: int,
    attention_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Boolean mask [B or 1, 1, q_len, start_pos + q_len]; True = may attend. None = no mask.

    attention_mask: [B, >= start_pos + q_len], 1 for real tokens and 0 for left padding.
    Every query may always see itself, so a padded query row is never fully masked: a fully
    masked row gives NaN, and NaN in a pad's K/V would leak into real rows (0 * NaN = NaN).
    """
    kv_len = start_pos + q_len
    if attention_mask is None and q_len == 1:
        return None  # a single decode token sees the whole prefix
    q_pos = torch.arange(start_pos, kv_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    mask = (k_pos <= q_pos)[None, None]  # causal, [1, 1, q, kv]
    if attention_mask is not None:
        keys_ok = attention_mask[:, None, None, :kv_len].bool()
        mask = (mask & keys_ok) | (k_pos == q_pos)[None, None]
    return mask


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        h, hd = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, cfg.num_attention_heads * hd, bias=False)
        self.k_proj = nn.Linear(h, cfg.num_key_value_heads * hd, bias=False)
        self.v_proj = nn.Linear(h, cfg.num_key_value_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * hd, h, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: ContiguousKVCache | PagedBatch | None = None,
        start_pos: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, T, H] -> [B, T, H].

        1. Project to q, k, v and reshape to [B, heads, T, hd].
        2. Apply RoPE to q and k.
        3. With a cache: write k, v at start_pos and attend over the whole filled prefix.
        4. Repeat KV heads num_kv_groups times (or pass enable_gqa=True to SDPA).
        5. torch.nn.functional.scaled_dot_product_attention with a causal mask
           (combined with attention_mask for padded batches), then o_proj.
        """
        b, t, _ = x.shape
        hd = self.cfg.head_dim
        q = self.q_proj(x).view(b, t, -1, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, -1, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, hd).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)

        if isinstance(cache, PagedBatch):
            return self.o_proj(self._paged(q[0], k[0], v[0], cache))[None]
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v, start_pos)
        else:
            start_pos = 0  # without a cache the keys are exactly this step's tokens

        groups = self.cfg.num_kv_groups
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        if attention_mask is None and start_pos == 0 and t > 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = build_attention_mask(t, start_pos, attention_mask, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))

    def _paged(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch: PagedBatch
    ) -> torch.Tensor:
        """Reference paged attention over flattened tokens.

        q: [nh, N, hd], k, v: [nkv, N, hd] for the step's N new tokens (all sequences
        concatenated). Writes k, v into the paged cache, gathers each sequence's context
        through its block table, and returns [N, nh * hd].
        """
        batch.cache.write(self.layer_idx, k.transpose(0, 1), v.transpose(0, 1), batch.slot_mapping)
        if batch.use_kernel:  # decode step: Triton kernel reads K/V through block tables
            from engine.kernels.paged_attention import paged_decode_attention

            out = paged_decode_attention(
                q.transpose(0, 1),
                batch.cache.k[self.layer_idx],
                batch.cache.v[self.layer_idx],
                batch.block_tables,
                batch.context_lens,
                batch.cache.block_size,
            )
            return out.flatten(1)
        k_ctx, v_ctx = batch.cache.gather(self.layer_idx, batch.kv_slots)  # [B, ctx, nkv, hd]
        k_ctx, v_ctx = k_ctx.transpose(1, 2).to(q.dtype), v_ctx.transpose(1, 2).to(q.dtype)
        groups = self.cfg.num_kv_groups
        if groups > 1:
            k_ctx = k_ctx.repeat_interleave(groups, dim=1)
            v_ctx = v_ctx.repeat_interleave(groups, dim=1)

        q_pad = q.transpose(0, 1)[batch.q_index].transpose(1, 2)  # [B, nh, max_q, hd]
        out = F.scaled_dot_product_attention(q_pad, k_ctx, v_ctx, attn_mask=batch.attn_mask)
        out = out.transpose(1, 2).flatten(2)  # [B, max_q, nh * hd]
        return out[batch.q_valid]  # back to flattened order: sequence by sequence


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: ContiguousKVCache | PagedBatch | None = None,
        start_pos: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-norm residual block: x + attn(norm(x)), then + mlp(norm(.))."""
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, start_pos, attention_mask)
        return h + self.mlp(self.post_attention_layernorm(h))
