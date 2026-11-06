"""Per-request sampling parameters. No torch dependency: the API layer uses this too."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0  # 0 means greedy
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    min_p: float = 0.0  # 0.0 disables
    repetition_penalty: float = 1.0
    max_tokens: int = 256
    seed: int | None = None
    stop: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError("min_p must be in [0, 1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0
