"""Turn logits into next tokens (Week 2)."""

from __future__ import annotations

import torch

from engine.runtime.params import SamplingParams


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
    prev_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """logits: [B, vocab] float32 -> token ids [B] int64.

    Order: repetition penalty (over prev_tokens) -> greedy shortcut when params.greedy
    -> temperature -> top-k -> top-p -> min-p -> torch.multinomial(probs, 1, generator=...).
    Masked-out logits become -inf; never renormalize by hand after masking, softmax does it.
    """
    logits = logits.float()
    if params.repetition_penalty != 1.0 and prev_tokens is not None:
        logits = apply_repetition_penalty(logits, prev_tokens, params.repetition_penalty)
    if params.greedy:
        return logits.argmax(dim=-1)

    logits = logits / params.temperature
    if params.top_k > 0 and params.top_k < logits.shape[-1]:
        kth = logits.topk(params.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if params.top_p < 1.0:
        logits = _top_p_mask(logits, params.top_p)
    if params.min_p > 0.0:
        probs = logits.softmax(dim=-1)
        floor = params.min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < floor, float("-inf"))

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style penalty, as in HF: positive logits / penalty, negative logits * penalty.

    prev_tokens: [B, L] ids already in each sequence (prompt and output). Repeats are fine.
    """
    seen = logits.gather(-1, prev_tokens)
    seen = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.scatter(-1, prev_tokens, seen)


def _top_p_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability adds up to at least top_p."""
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs  # probability of all higher-ranked tokens
    # The 1e-6 slack stops float rounding from adding one extra token at an exact boundary.
    drop_sorted = mass_before >= top_p - 1e-6
    drop = drop_sorted.scatter(-1, order, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))
