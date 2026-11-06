"""Helpers for comparing the engine against the Hugging Face reference."""

from __future__ import annotations

from collections.abc import Sequence


def first_divergence(ours: Sequence[int], ref: Sequence[int]) -> int | None:
    """Index of the first differing token, or None if identical.

    A length difference counts as a divergence at the shorter length.
    """
    for i, (a, b) in enumerate(zip(ours, ref, strict=False)):
        if a != b:
            return i
    if len(ours) != len(ref):
        return min(len(ours), len(ref))
    return None


def is_near_tie(top2_logit_gap: float, threshold: float = 1e-2) -> bool:
    """Gate 1 rule: a BF16 mismatch is acceptable when the reference's top-2 gap is tiny."""
    return top2_logit_gap < threshold
