"""The Phase 1 generation loop: static batch, contiguous cache (Week 2)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from engine.runtime.detokenizer import IncrementalDetokenizer
from engine.runtime.kv_cache import ContiguousKVCache
from engine.runtime.params import SamplingParams
from engine.runtime.sampler import sample


@dataclass
class StepOutput:
    seq_index: int  # which prompt in the batch
    token_id: int
    text: str  # detokenized delta, possibly ""
    finished: bool
    finish_reason: str | None = None  # "stop" | "length"


class _StopStringFilter:
    """Holds back text that could be the start of a stop string, so none of it leaks out."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self.stops = stops
        self.holdback = max((len(s) for s in stops), default=1) - 1
        self.text = ""
        self.emitted = 0

    def push(self, delta: str) -> tuple[str, bool]:
        """Add a delta; return (text safe to emit now, whether a stop string was hit)."""
        self.text += delta
        # Everything before `emitted` was already checked; a new match starts in the held-back
        # tail or the new delta.
        hits = [i for s in self.stops if (i := self.text.find(s, self.emitted)) != -1]
        if hits:
            return self._emit_until(min(hits)), True
        return self._emit_until(len(self.text) - self.holdback), False

    def flush(self) -> str:
        return self._emit_until(len(self.text))

    def _emit_until(self, end: int) -> str:
        if end <= self.emitted:
            return ""
        out, self.emitted = self.text[self.emitted : end], end
        return out


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
    detoks = [IncrementalDetokenizer(tokenizer, p) for p in prompts]
    filters = [_StopStringFilter(params.stop) for _ in prompts]
    stop_set = set(stop_token_ids)
    finished = [False] * batch
    history = input_ids  # [B, L]: everything each row has seen, for the repetition penalty

    logits = model(input_ids, positions, cache, 0, attention_mask)[:, -1]
    for step in range(params.max_tokens):
        tokens = sample(logits, params, generator, prev_tokens=history)
        for row, tok in enumerate(tokens.tolist()):
            if finished[row]:
                continue
            out = _advance(row, tok, step, params, stop_set, detoks[row], filters[row])
            finished[row] = out.finished
            yield out
        if all(finished) or step == params.max_tokens - 1:
            return

        next_ids = tokens[:, None]
        history = torch.cat([history, next_ids], dim=1)
        positions = positions[:, -1:] + 1
        logits = model(next_ids, positions, cache, prompt_len + step, attention_mask)[:, -1]


def _advance(
    row: int,
    tok: int,
    step: int,
    params: SamplingParams,
    stop_set: set[int],
    detok: IncrementalDetokenizer,
    stop_filter: _StopStringFilter,
) -> StepOutput:
    if tok in stop_set:
        return StepOutput(row, tok, stop_filter.flush(), True, "stop")
    text, hit_stop = stop_filter.push(detok.add(tok))
    if hit_stop:
        return StepOutput(row, tok, text, True, "stop")
    if step == params.max_tokens - 1:
        return StepOutput(row, tok, text + stop_filter.flush(), True, "length")
    return StepOutput(row, tok, text, False)
