"""FP8 (e4m3) weights and activations on GPUs with FP8 tensor cores (Phase 4, FR-15).

Ada (sm_89, e.g. RTX 40xx) and Hopper run FP8 matmuls natively. Weights are quantized
once with one scale for the whole tensor; activations are quantized on the fly, also with
one scale per call. ``torch._scaled_mm`` multiplies in FP8 and rescales into BF16. (Per-row
scales would be more accurate, but PyTorch supports them only on Hopper, not on Ada.)
Halves weight memory versus BF16 with little quality loss; INT4 halves it again.
"""

from __future__ import annotations

import torch
from torch import nn

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max  # 448


def supported(device: torch.device | str = "cuda") -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) >= (8, 9)


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor absmax scaling: returns (fp8 values, float32 scalar scale)."""
    amax = x.abs().amax().float().clamp(min=1e-12)
    scale = amax / FP8_MAX
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


class Fp8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        n, k = weight.shape
        if n % 16 or k % 16:
            raise ValueError("FP8 matmul needs both dimensions to be multiples of 16")
        self.in_features, self.out_features = k, n
        w8, w_scale = quantize_fp8(weight)  # [N, K] row-major, scalar scale
        self.register_buffer("weight_fp8", w8.contiguous())
        self.register_buffer("weight_scale", w_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        x8, x_scale = quantize_fp8(x2)  # [M, K], scalar scale
        # _scaled_mm wants its second operand column-major: W.t() of a row-major W is.
        y = torch._scaled_mm(
            x8, self.weight_fp8.t(), scale_a=x_scale, scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
        )  # fmt: skip
        return y.view(*shape[:-1], self.out_features).to(x.dtype)
