"""The Phase 1 generation loop: static batch, contiguous cache (Week 2)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from engine.runtime.kv_cache import ContiguousKVCache
from engine.runtime.params import SamplingParams
from engine.runtime.sampler import sample
from engine.runtime.stop import OutputProcessor


@dataclass
class StepOutput:
    seq_index: int  # which prompt in the batch
    token_id: int
    text: str  # detokenized delta, possibly ""
    finished: bool
    finish_reason: str | None = None  # "stop" | "length"


def generate(
    model: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    params: SamplingParams,
    stop_token_ids: tuple[int, ...],
) -> Iterator[StepOutput]:
    """Yield one StepOutput per sequence per step until every sequence finishes.

    1. Left-pad prompts to one length; build attention_mask and positions (pads get 0).
    2. Allocate ContiguousKVCache(batch, max_prompt_len + params.max_tokens).
    3. Prefill: model(input_ids, positions, cache, start_pos=0) -> sample from logits[:, -1].
    4. Decode loop: feed the last token with start_pos += 1 until stop id, stop string,
       or max_tokens. Finished rows keep running (static batch) but emit nothing.
    5. Seed a torch.Generator from params.seed for reproducibility.

    A stop token is yielded as the sequence's last token (as HF generate returns it), with
    no text. A stop string ends the sequence, and neither it nor anything after it is emitted.
    """
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("every prompt needs at least one token")
    weight = next(model.parameters())
    device, dtype = weight.device, weight.dtype
    batch, prompt_len = len(prompts), max(len(p) for p in prompts)
    total_len = prompt_len + params.max_tokens

    # Left padding. A pad slot holds a copy of the row's first real token: it is masked out
    # of attention anyway, and the repetition penalty then never counts a fake token.
    input_ids = torch.tensor(
        [[p[0]] * (prompt_len - len(p)) + p for p in prompts], device=device, dtype=torch.long
    )
    padded = any(len(p) < prompt_len for p in prompts)
    attention_mask = None
    if padded:
        attention_mask = torch.ones(batch, total_len, device=device, dtype=torch.long)
        for row, p in enumerate(prompts):
            attention_mask[row, : prompt_len - len(p)] = 0
        positions = (attention_mask[:, :prompt_len].cumsum(-1) - 1).clamp(min=0)
    else:
        positions = torch.arange(prompt_len, device=device).expand(batch, -1)

    generator = None
    if params.seed is not None:
        generator = torch.Generator(device=device).manual_seed(params.seed)

    cache = ContiguousKVCache(model.cfg, batch, total_len, device=device, dtype=dtype)
    processors = [OutputProcessor(tokenizer, p, params, stop_token_ids) for p in prompts]
    finished = [False] * batch
    history = input_ids  # [B, L]: everything each row has seen, for the repetition penalty

    logits = model(input_ids, positions, cache, 0, attention_mask, last_only=True)[:, -1]
    for step in range(params.max_tokens):
        tokens = sample(logits, params, generator, prev_tokens=history)
        for row, tok in enumerate(tokens.tolist()):
            if finished[row]:
                continue
            text, finished[row], reason = processors[row].process(tok)
            yield StepOutput(row, tok, text, finished[row], reason)
        if all(finished) or step == params.max_tokens - 1:
            return

        next_ids = tokens[:, None]
        history = torch.cat([history, next_ids], dim=1)
        positions = positions[:, -1:] + 1
        logits = model(next_ids, positions, cache, prompt_len + step, attention_mask)[:, -1]
