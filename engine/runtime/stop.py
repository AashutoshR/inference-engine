"""Per-sequence output handling shared by the static loop and the serving engine:
detokenize, apply stop tokens, stop strings and max_tokens."""

from __future__ import annotations

from typing import Any

from engine.runtime.detokenizer import IncrementalDetokenizer
from engine.runtime.params import SamplingParams


class StopStringFilter:
    """Holds back text that could be the start of a stop string, so none of it leaks out."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self.stops = stops
        self.holdback = max((len(s) for s in stops), default=1) - 1
        self.text = ""
        self.emitted = 0

    def push(self, delta: str) -> tuple[str, bool]:
        """Add a delta; return (text safe to emit now, whether a stop string was hit)."""
        self.text += delta
        # Everything before `emitted` was already checked; a new match starts in the
        # held-back tail or the new delta.
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


class OutputProcessor:
    """Turns one sequence's sampled tokens into text deltas and a finish decision.

    A stop token finishes with reason "stop" and no text of its own (it is still the
    sequence's last token, as HF generate returns it). A stop string finishes with
    reason "stop", and neither it nor anything after it is emitted.
    """

    def __init__(
        self,
        tokenizer: Any,
        prompt_ids: list[int],
        params: SamplingParams,
        stop_token_ids: tuple[int, ...] | set[int],
    ) -> None:
        self.params = params
        self.stop_ids = set(stop_token_ids)
        self.detok = IncrementalDetokenizer(tokenizer, prompt_ids)
        self.filter = StopStringFilter(params.stop)
        self.num_generated = 0

    def process(self, token_id: int) -> tuple[str, bool, str | None]:
        """Returns (text delta, finished, finish_reason)."""
        self.num_generated += 1
        if token_id in self.stop_ids:
            return self.filter.flush(), True, "stop"
        text, hit_stop = self.filter.push(self.detok.add(token_id))
        if hit_stop:
            return text, True, "stop"
        if self.num_generated >= self.params.max_tokens:
            return text + self.filter.flush(), True, "length"
        return text, False, None
