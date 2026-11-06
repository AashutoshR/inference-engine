"""Week 2 definition of done for the sampler."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.runtime.params import SamplingParams  # noqa: E402
from engine.runtime.sampler import sample  # noqa: E402


def test_greedy_is_argmax():
    logits = torch.randn(4, 100)
    out = sample(logits, SamplingParams(temperature=0))
    torch.testing.assert_close(out, logits.argmax(-1))


def test_top_k_never_picks_outside_k():
    torch.manual_seed(0)
    logits = torch.randn(1, 100)
    allowed = set(logits.topk(5).indices[0].tolist())
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        tok = sample(logits, SamplingParams(temperature=1.0, top_k=5), generator=gen)
        assert tok.item() in allowed


def test_top_p_keeps_smallest_set_over_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    gen = torch.Generator().manual_seed(0)
    seen = {sample(logits, SamplingParams(top_p=0.8), generator=gen).item() for _ in range(300)}
    assert seen == {0, 1}


def test_same_seed_same_tokens():
    logits = torch.randn(2, 1000)
    a = sample(logits, SamplingParams(), generator=torch.Generator().manual_seed(42))
    b = sample(logits, SamplingParams(), generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(a, b)
