"""Contiguous KV cache (Week 2). Phase 2 replaces it with a paged cache and block tables."""

from __future__ import annotations

import torch

from engine.config import ModelConfig


class ContiguousKVCache:
    """One preallocated buffer per layer for a static batch.

    k, v: [layers, B, nkv, max_len, hd]. All sequences in the batch share one length
    (prompts are left-padded), so a single ``start_pos`` addresses every row.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        max_len: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        shape = (cfg.num_hidden_layers, batch_size, cfg.num_key_value_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len

    def update(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k_new, v_new ([B, nkv, T, hd]) at start_pos; return the filled prefix
        [B, nkv, start_pos + T, hd] of this layer for attention. Raise if it overflows max_len.
        """
        end = start_pos + k_new.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} tokens > max_len {self.max_len}")
        self.k[layer, :, :, start_pos:end] = k_new
        self.v[layer, :, :, start_pos:end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]
