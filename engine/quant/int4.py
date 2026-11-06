"""INT4 weight-only quantization with a fused dequant matmul (Phase 4, FR-15).

Each linear weight [N, K] is split into groups of ``group_size`` input channels. A group
stores 4-bit codes q in [0, 15] plus a BF16 scale and zero point, and dequantizes as
``w = (q - 8) * scale + zero``. That is the layout of PyTorch's tinygemm kernel
(``aten._weight_int4pack_mm``, the one torchao uses): it unpacks the codes inside the
matmul, so weights are read from memory at 4 bits instead of 16. Decode is bound by
weight reads, which is where the speedup comes from.

Quantization is round-to-nearest (RTN) per group with an asymmetric range. Embeddings
and the LM head stay in BF16. ``quantize_rtn`` and ``dequantize`` are plain PyTorch and
double as the test oracle.
"""

from __future__ import annotations

import torch
from torch import nn

GROUP_SIZES = (32, 64, 128, 256)


def quantize_rtn(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [N, K] -> (codes int32 [N, K] in [0, 15], scales_and_zeros [K/g, N, 2] bf16)."""
    n, k = w.shape
    if k % group_size:
        raise ValueError(f"in_features {k} not divisible by group size {group_size}")
    wg = w.float().view(n, k // group_size, group_size)
    lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    scale = ((hi - lo) / 15).clamp(min=1e-8)
    # Round scale and zero to BF16 first, so codes are computed against the values the
    # kernel will actually use.
    scale = scale.to(torch.bfloat16).float()
    zero = (lo + 8 * scale).to(torch.bfloat16).float()
    codes = ((wg - zero) / scale + 8).round().clamp(0, 15).to(torch.int32).view(n, k)
    sz = torch.stack([scale.squeeze(-1), zero.squeeze(-1)], -1)  # [N, K/g, 2]
    return codes, sz.transpose(0, 1).contiguous().to(torch.bfloat16)


def dequantize(codes: torch.Tensor, sz: torch.Tensor, group_size: int) -> torch.Tensor:
    n, k = codes.shape
    scale = sz[..., 0].t().float().unsqueeze(-1)  # [N, K/g, 1]
    zero = sz[..., 1].t().float().unsqueeze(-1)
    return ((codes.view(n, k // group_size, group_size).float() - 8) * scale + zero).view(n, k)


def _inner_k_tiles(k: int) -> int:
    for tiles in (8, 4, 2):
        if k % (tiles * 16) == 0:
            return tiles
    raise ValueError(f"in_features {k} must be a multiple of 32 for the int4 kernel")


class Int4Linear(nn.Module):
    """Drop-in for a bias-free nn.Linear, BF16 activations, INT4 weights."""

    def __init__(self, weight: torch.Tensor, group_size: int = 128) -> None:
        super().__init__()
        if group_size not in GROUP_SIZES:
            raise ValueError(f"group_size must be one of {GROUP_SIZES}")
        n, k = weight.shape
        if n % 8:
            raise ValueError(f"out_features {n} must be a multiple of 8")
        self.in_features, self.out_features, self.group_size = k, n, group_size
        codes, sz = quantize_rtn(weight, group_size)
        packed = (codes[:, ::2] << 4 | codes[:, 1::2]).to(torch.uint8)
        self.register_buffer(
            "packed", torch.ops.aten._convert_weight_to_int4pack(packed, _inner_k_tiles(k))
        )
        self.register_buffer("scales_and_zeros", sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features).to(torch.bfloat16)
        y = torch.ops.aten._weight_int4pack_mm(
            x2, self.packed, self.group_size, self.scales_and_zeros
        )
        return y.view(*shape[:-1], self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group_size={self.group_size}"


LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def is_quantizable(name: str) -> bool:
    """State-dict key of a decoder-layer linear weight (not embeddings or the LM head)."""
    return name.endswith(".weight") and name.split(".")[-2] in LINEAR_NAMES


def install(model: nn.Module, name: str, module: nn.Module) -> None:
    """Replace the linear that owns state-dict key ``name`` with ``module``."""
    path = name.removesuffix(".weight").split(".")
    parent = model.get_submodule(".".join(path[:-1]))
    setattr(parent, path[-1], module)


def make_quantized(kind: str, weight: torch.Tensor, group_size: int = 128) -> nn.Module:
    if kind == "int4":
        return Int4Linear(weight, group_size)
    if kind == "fp8":
        from engine.quant.fp8 import Fp8Linear

        return Fp8Linear(weight)
    raise ValueError(f"unknown quantization {kind!r}; supported: int4, fp8")


def quantize_model(model: nn.Module, group_size: int = 128) -> nn.Module:
    """Quantize every decoder-layer linear of an already-loaded BF16 model in place."""
    for name, param in list(model.named_parameters()):
        if is_quantizable(name):
            install(model, name, Int4Linear(param.detach(), group_size))
    return model
