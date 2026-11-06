"""Speculative decoding for one sequence (Phase 4, FR-16).

Each round a proposer guesses k tokens; the target model scores all of them in a single
forward pass; a prefix of the guesses is accepted, plus one token from the target.
Decode is bound by reading the target's weights, so checking k+1 positions costs about
the same as generating one, and every accepted guess is a step saved.

Proposers:
  * a draft model (a smaller model with the same tokenizer), or
  * n-gram lookup (no model): find the last earlier occurrence of the context's final
    n-gram and propose what followed it. Strong on summaries, code edits, quoting.

Acceptance (Leviathan et al. 2023; Chen et al. 2023) keeps the target's distribution
exactly: accept guess x with probability min(1, p(x)/q(x)); on rejection, sample from
norm(max(0, p - q)). A deterministic proposer has q = one-hot, so p(x) is the
acceptance probability. Greedy decoding accepts while each guess equals the target's
argmax, so its output is identical to plain greedy decoding.

Caches are contiguous (one sequence). Rolling back rejected guesses is free: their K/V
sit past the accepted length and get overwritten next round.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from engine.runtime.generate import StepOutput
from engine.runtime.kv_cache import ContiguousKVCache
from engine.runtime.params import SamplingParams
from engine.runtime.sampler import distribution
from engine.runtime.stop import OutputProcessor


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    history: list[int] = field(default_factory=list)  # accepted guesses per round

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0


def accept_or_resample(
    p: torch.Tensor, q: torch.Tensor, x: int, generator: torch.Generator | None
) -> tuple[bool, int]:
    """One speculative-sampling decision. p, q: [vocab] target and proposal distributions;
    x: the proposed token. Returns (accepted, token): x itself, or a residual sample."""
    px, qx = float(p[x]), float(q[x])
    u = torch.rand((), generator=generator, device=p.device).item()
    if qx > 0 and u < min(1.0, px / qx):
        return True, x
    residual = (p - q).clamp(min=0)
    total = residual.sum()
    if total <= 0:  # p == q up to rounding: any sample from p is right
        residual, total = p, p.sum()
    return False, int(torch.multinomial(residual / total, 1, generator=generator))


def ngram_propose(tokens: list[int], k: int, max_n: int = 3, min_n: int = 1) -> list[int]:
    """Prompt-lookup proposal: continue the latest earlier match of the final n-gram."""
    for n in range(max_n, min_n - 1, -1):
        if len(tokens) <= n:
            continue
        tail = tokens[-n:]
        for start in range(len(tokens) - n - 1, -1, -1):
            if tokens[start : start + n] == tail:
                follow = tokens[start + n : start + n + k]
                if follow:
                    return follow
    return []


class SpeculativeDecoder:
    def __init__(
        self,
        target: Any,
        draft: Any | None = None,
        num_speculative_tokens: int = 4,
        ngram_max: int = 3,
    ) -> None:
        self.target, self.draft = target, draft
        self.k = num_speculative_tokens
        self.ngram_max = ngram_max
        self.stats = SpecStats()

    def _forward(self, model, cache, tokens: list[int], start: int) -> torch.Tensor:
        dev = next(model.parameters()).device
        ids = torch.tensor([tokens], device=dev)
        pos = torch.arange(start, start + len(tokens), device=dev)[None]
        return model(ids, pos, cache, start)[0]  # [T, vocab] float32

    def generate(
        self,
        tokenizer: Any,
        prompt: list[int],
        params: SamplingParams,
        stop_token_ids: tuple[int, ...],
    ) -> Iterator[StepOutput]:
        if params.repetition_penalty != 1.0:
            raise ValueError("speculative decoding does not support repetition_penalty")
        cfg, dev = self.target.cfg, next(self.target.parameters()).device
        dtype = next(self.target.parameters()).dtype
        length = len(prompt) + params.max_tokens + self.k + 2
        t_cache = ContiguousKVCache(cfg, 1, length, dev, dtype)
        d_cache = None
        if self.draft is not None:
            dd = next(self.draft.parameters())
            d_cache = ContiguousKVCache(self.draft.cfg, 1, length, dd.device, dd.dtype)
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=dev).manual_seed(params.seed)
        proc = OutputProcessor(tokenizer, prompt, params, stop_token_ids)

        tokens = list(prompt)
        # Invariant: caches hold K/V for tokens[:t_len] / tokens[:d_len]; the target has not
        # yet seen tokens[t_len:], which always includes the newest token.
        t_len = d_len = 0
        # Prefill all but the last prompt token; the first round feeds the rest.
        if len(tokens) > 1:
            self._forward(self.target, t_cache, tokens[:-1], 0)
            t_len = len(tokens) - 1
            if self.draft is not None:
                self._forward(self.draft, d_cache, tokens[:-1], 0)
                d_len = len(tokens) - 1

        while True:
            budget = params.max_tokens - proc.num_generated
            k = min(self.k, budget - 1)
            guesses, q_dists = self._propose(tokens, d_cache, d_len, k, params, gen)
            if self.draft is not None and guesses:
                # The draft fed tokens[d_len:] and all guesses but the last.
                d_len = len(tokens) + len(guesses) - 1

            fed = tokens[t_len:] + guesses
            logits = self._forward(self.target, t_cache, fed, t_len)
            base = len(tokens) - t_len - 1  # row of logits predicting the first new token
            self.stats.rounds += 1
            self.stats.proposed += len(guesses)

            new, accepted = self._verify(logits, base, guesses, q_dists, params, gen)
            self.stats.accepted += accepted
            self.stats.history.append(accepted)
            t_len = len(tokens) + accepted  # target K/V valid up to the last accepted guess
            if self.draft is not None:
                d_len = min(d_len, len(tokens) + accepted)

            for tok in new:
                tokens.append(tok)
                self.stats.emitted += 1
                text, finished, reason = proc.process(tok)
                yield StepOutput(0, tok, text, finished, reason)
                if finished:
                    return

    def _propose(self, tokens, d_cache, d_len, k, params, gen):
        if k <= 0:
            return [], []
        if self.draft is None:
            guesses = ngram_propose(tokens, k, self.ngram_max)
            return guesses, [None] * len(guesses)  # None = deterministic (one-hot) proposal
        guesses, q_dists = [], []
        feed, start = tokens[d_len:], d_len
        for _ in range(k):
            row = self._forward(self.draft, d_cache, feed, start)[-1]
            start += len(feed)
            if params.greedy:
                tok, q = int(row.argmax()), None
            else:
                q = distribution(row[None], params)[0]
                tok = int(torch.multinomial(q, 1, generator=gen))
            guesses.append(tok)
            q_dists.append(q)
            feed = [tok]
        return guesses, q_dists

    def _verify(self, logits, base, guesses, q_dists, params, gen) -> tuple[list[int], int]:
        out: list[int] = []
        for i, x in enumerate(guesses):
            row = logits[base + i]
            if params.greedy:
                best = int(row.argmax())
                if best != x:
                    return out + [best], i
                out.append(x)
                continue
            p = distribution(row[None], params)[0]
            q = q_dists[i]
            if q is None:
                q = torch.zeros_like(p)
                q[x] = 1.0
            ok, tok = accept_or_resample(p, q, x, gen)
            out.append(tok)
            if not ok:
                return out, i
        # Every guess accepted: one more token from the target for free.
        row = logits[base + len(guesses)]
        if params.greedy:
            bonus = int(row.argmax())
        else:
            bonus = int(torch.multinomial(distribution(row[None], params)[0], 1, generator=gen))
        return out + [bonus], len(guesses)
