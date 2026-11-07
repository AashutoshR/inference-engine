"""Triton paged attention for decode rows (Phase 2, FR-6; GQA kernel in Phase 3).

Each program walks one sequence's context through its block table in tiles of BLOCK_N
tokens, loading K and V straight from the paged cache, and keeps an online softmax
(running max, running sum, running weighted V). Nothing is gathered into a padded copy
first, which is what the PyTorch reference path does.

Two kernels, picked by dtype:

- BF16/FP16: one program per (sequence, KV head). Llama shares each KV head between
  several query heads (4 in the 1B and 8B models); those become the rows of one small
  matrix, so each K/V tile is read once instead of once per query head, and both
  products run on the matrix units (tl.dot). The softmax weights are rounded to BF16
  before multiplying V, as in FlashAttention (and the engine's own prompt path).
  At 16 sequences x 500 tokens a layer takes 0.022 ms instead of 0.058; at 64 x 1,500,
  0.80 ms instead of 1.10.
- FP32: one program per (sequence, query head), all in float32. Slower, but it keeps
  the FP32 reference mode as exact as it was.
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

    @triton.jit
    def _paged_decode_gqa_kernel(
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
        """One program per (sequence, KV head). The GROUPS query heads sharing that KV
        head are the rows of one matrix (padded to 16, the smallest tl.dot), so each
        K/V tile is read once and both products run on the matrix units."""
        b = tl.program_id(0)
        kvh = tl.program_id(1)
        ctx = tl.load(ctx_ptr + b)

        g = tl.arange(0, 16)
        g_ok = g < GROUPS
        d = tl.arange(0, HEAD_DIM)
        q_off = b * stride_qb + (kvh * GROUPS + g)[:, None] * stride_qh + d[None, :]
        q = tl.load(q_ptr + q_off, mask=g_ok[:, None], other=0.0)

        m = tl.full([16], float("-inf"), tl.float32)
        l_sum = tl.zeros([16], tl.float32)
        acc = tl.zeros([16, HEAD_DIM], tl.float32)

        for start in range(0, ctx, BLOCK_N):
            pos = start + tl.arange(0, BLOCK_N)
            valid = pos < ctx
            block = tl.load(tables_ptr + b * stride_tb + pos // BLOCK_SIZE, mask=valid, other=0)
            slot = block * BLOCK_SIZE + pos % BLOCK_SIZE
            kv_off = slot[:, None] * stride_ks + kvh * stride_kh + d[None, :]
            k = tl.load(k_ptr + kv_off, mask=valid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale  # [16, BLOCK_N], float32 accumulation
            s = tl.where(valid[None, :], s, float("-inf"))

            m_new = tl.maximum(m, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m - m_new)
            v = tl.load(v_ptr + kv_off, mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_sum = l_sum * alpha + tl.sum(p, axis=1)
            m = m_new

        out = acc / l_sum[:, None]
        tl.store(out_ptr + q_off, out.to(out_ptr.dtype.element_ty), mask=g_ok[:, None])


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
    if use_gqa_kernel(q.dtype, nh // nkv, hd):
        _paged_decode_gqa_kernel[(b, nkv)](
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


def use_gqa_kernel(dtype: torch.dtype, groups: int, head_dim: int) -> bool:
    """Half precision, at most 16 query heads per KV head, head_dim a power of two >= 16."""
    return (
        dtype in (torch.bfloat16, torch.float16)
        and groups <= 16
        and head_dim >= 16
        and head_dim & (head_dim - 1) == 0
    )


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
