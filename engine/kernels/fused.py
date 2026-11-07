"""Fused element-wise kernels: RMSNorm, RoPE, SiLU-and-multiply (Triton, GPU only).

A decode step of a 1B model reads 2.5 GB of weights in about 10 ms; everything else is
small element-wise work that PyTorch runs as separate kernels: RMSNorm is 7 of them
(cast, square, mean, add, rsqrt, multiply, scale), RoPE is 12 (casts, slices, negate,
concatenate, multiply, add, for q and for k), SiLU-and-multiply is 2. That is ~35 tiny
kernels per layer, ~560 per step at ~2 us each even inside a CUDA graph. Here each is
one kernel per call.

Each kernel rounds where PyTorch rounds, so outputs match the unfused ops:

- RoPE: every product and the sum are rounded to the model dtype, as in HF's
  ``q * cos + rotate_half(q) * sin``. Bitwise identical.
- SiLU-and-multiply: SiLU in float32 with IEEE exp and division, rounded, then the
  product rounded. Bitwise identical.
- RMSNorm: PyTorch computes the mean of squares, because a fused sum adds in another
  order, and in BF16 that alone changed 27 of 200 greedy answers on Llama 3.2 3B. One
  kernel does the rest (add, rsqrt, multiply, cast, scale). Bitwise identical.

Floating-point contraction (fusing a * b + c into one FMA) is turned off for these
kernels, since PyTorch's separate kernels round between the multiply and the add.
"""

from __future__ import annotations

import os

import torch

from engine.kernels.paged_attention import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

# ENGINE_FUSED_KERNELS=0 runs the plain PyTorch ops everywhere (for comparisons).
ENABLED = os.environ.get("ENGINE_FUSED_KERNELS", "1") != "0"


def available(t: torch.Tensor) -> bool:
    return ENABLED and HAS_TRITON and t.is_cuda


if HAS_TRITON:

    @triton.jit
    def _rms_norm_kernel(x_ptr, ms_ptr, w_ptr, out_ptr, stride_x, stride_o, n_cols, eps,
                         BLOCK: tl.constexpr):  # fmt: skip
        """Everything after the mean of squares, which PyTorch computes (see rms_norm)."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        ms = tl.load(ms_ptr + row)
        normed = (x * libdevice.rsqrt(ms + eps)).to(out_ptr.dtype.element_ty)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = w * normed.to(tl.float32)
        tl.store(out_ptr + row * stride_o + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rope_kernel(x_ptr, cos_ptr, sin_ptr, stride_xn, stride_xh, stride_cn,
                     HALF: tl.constexpr):  # fmt: skip
        n = tl.program_id(0)  # token
        h = tl.program_id(1)  # head
        d = tl.arange(0, HALF)
        base = x_ptr + n * stride_xn + h * stride_xh
        dt = x_ptr.dtype.element_ty
        x1 = tl.load(base + d).to(tl.float32)
        x2 = tl.load(base + HALF + d).to(tl.float32)
        # cos/sin come in float32; HF casts them to the model dtype first.
        c1 = tl.load(cos_ptr + n * stride_cn + d).to(dt).to(tl.float32)
        c2 = tl.load(cos_ptr + n * stride_cn + HALF + d).to(dt).to(tl.float32)
        s1 = tl.load(sin_ptr + n * stride_cn + d).to(dt).to(tl.float32)
        s2 = tl.load(sin_ptr + n * stride_cn + HALF + d).to(dt).to(tl.float32)
        # out = x * cos + rotate_half(x) * sin, rotate_half(x) = cat(-x2, x1)
        o1 = (x1 * c1).to(dt).to(tl.float32) + (-x2 * s1).to(dt).to(tl.float32)
        o2 = (x2 * c2).to(dt).to(tl.float32) + (x1 * s2).to(dt).to(tl.float32)
        tl.store(base + d, o1.to(dt))
        tl.store(base + HALF + d, o2.to(dt))

    @triton.jit
    def _silu_mul_kernel(g_ptr, u_ptr, out_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        dt = out_ptr.dtype.element_ty
        g = tl.load(g_ptr + i, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + i, mask=mask, other=0.0).to(tl.float32)
        silu = libdevice.div_rn(g, 1.0 + libdevice.exp(-g)).to(dt).to(tl.float32)
        tl.store(out_ptr + i, (silu * u).to(dt), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """weight * (x_fp32 * rsqrt(mean(x_fp32^2) + eps)).to(x.dtype), over the last dim.

    The mean of squares is left to PyTorch: a fused sum adds in another order, and that
    changed greedy answers (27 of 200 prompts on Llama 3.2 3B in BF16). The rest (add,
    rsqrt, multiply, cast, scale) is one kernel, bitwise identical in every dtype."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    ms = x2.float().pow(2).mean(-1)
    out = torch.empty(x2.shape, dtype=torch.promote_types(weight.dtype, x.dtype), device=x.device)
    n_cols = shape[-1]
    _rms_norm_kernel[(x2.shape[0],)](
        x2, ms, weight, out, x2.stride(0), out.stride(0), n_cols, eps,
        BLOCK=triton.next_power_of_2(n_cols), enable_fp_fusion=False,
    )  # fmt: skip
    return out.view(shape)


def rope_(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """In place: x [N, heads, hd] (rows contiguous in hd), cos/sin [N, hd] float32."""
    n, heads, hd = x.shape
    assert x.stride(-1) == 1 and cos.stride(-1) == 1 and cos.stride() == sin.stride()
    _rope_kernel[(n, heads)](
        x, cos, sin, x.stride(0), x.stride(1), cos.stride(0), HALF=hd // 2,
        enable_fp_fusion=False,
    )  # fmt: skip


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """F.silu(gate) * up, both rounded to the input dtype as PyTorch does."""
    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    block = 1024
    _silu_mul_kernel[(triton.cdiv(n, block),)](gate, up, out, n, BLOCK=block,
                                               enable_fp_fusion=False)  # fmt: skip
    return out
