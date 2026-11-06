"""Triton paged attention for decode steps (Phase 2, FR-6).

One program per (sequence, query head). It walks the sequence's context through its
block table in tiles of BLOCK_N tokens, loading K and V straight from the paged cache,
and keeps an online softmax (running max, running sum, running weighted V). Nothing is
gathered into a padded copy first, which is what the PyTorch reference path does.

Only decode steps (one new token per sequence) use it; prefill and mixed steps use the
reference path in ``engine.model.layers.Attention._paged``.
"""

from __future__ import annotations

import os

import torch

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only installs, CI
    HAS_TRITON = False


def available(device: torch.device) -> bool:
    return HAS_TRITON and device.type == "cuda"


if HAS_TRITON:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr, tables_ptr, ctx_ptr,
        stride_qb, stride_qh,
        stride_ks, stride_kh,
        stride_tb,
        scale,
        GROUPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):  # fmt: skip
        b = tl.program_id(0)
        h = tl.program_id(1)
        kvh = h // GROUPS
        ctx = tl.load(ctx_ptr + b)

        d = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d).to(tl.float32) * scale

        m = tl.full([1], float("-inf"), tl.float32)
        l_sum = tl.zeros([1], tl.float32)
        acc = tl.zeros([HEAD_DIM], tl.float32)

        for start in range(0, ctx, BLOCK_N):
            pos = start + tl.arange(0, BLOCK_N)
            valid = pos < ctx
            block = tl.load(tables_ptr + b * stride_tb + pos // BLOCK_SIZE, mask=valid, other=0)
            slot = block * BLOCK_SIZE + pos % BLOCK_SIZE
            kv_off = slot[:, None] * stride_ks + kvh * stride_kh + d[None, :]
            k = tl.load(k_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1)
            s = tl.where(valid, s, float("-inf"))

            m_new = tl.maximum(m, tl.max(s, axis=0))
            p = tl.exp(s - m_new)
            alpha = tl.exp(m - m_new)
            v = tl.load(v_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_sum = l_sum * alpha + tl.sum(p, axis=0)
            m = m_new

        out = acc / l_sum
        tl.store(out_ptr + b * stride_qb + h * stride_qh + d, out.to(out_ptr.dtype.element_ty))


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    block_n: int = 64,
) -> torch.Tensor:
    """q: [B, nh, hd]; k_cache, v_cache: [num_slots, nkv, hd] (one layer);
    block_tables: [B, max_blocks] int32; context_lens: [B] int32 -> [B, nh, hd]."""
    b, nh, hd = q.shape
    nkv = k_cache.shape[1]
    q = q.contiguous()
    out = torch.empty_like(q)
    _paged_decode_kernel[(b, nh)](
        q, k_cache, v_cache, out, block_tables, context_lens,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        block_tables.stride(0),
        hd**-0.5,
        GROUPS=nh // nkv,
        BLOCK_SIZE=block_size,
        HEAD_DIM=hd,
        BLOCK_N=block_n,
    )  # fmt: skip
    return out


def reference_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Plain PyTorch, one sequence at a time: the oracle for tests."""
    outs = []
    groups = q.shape[1] // k_cache.shape[1]
    for i in range(q.shape[0]):
        n = int(context_lens[i])
        pos = torch.arange(n, device=q.device)
        slots = block_tables[i, pos // block_size].long() * block_size + pos % block_size
        k = k_cache[slots].float().repeat_interleave(groups, dim=1)  # [n, nh, hd]
        v = v_cache[slots].float().repeat_interleave(groups, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[i].float(), k) * q.shape[-1] ** -0.5
        outs.append(torch.einsum("hn,nhd->hd", scores.softmax(-1), v))
    return torch.stack(outs).to(q.dtype)
