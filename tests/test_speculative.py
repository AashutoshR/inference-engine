"""Phase 4: speculative decoding. Greedy output must equal plain greedy exactly, and
sampled output must keep the target's distribution (chi-square on 10,000 samples)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from engine.config import ModelConfig  # noqa: E402
from engine.model.llama import LlamaForCausalLM  # noqa: E402
from engine.runtime.generate import generate  # noqa: E402
from engine.runtime.params import SamplingParams  # noqa: E402
from engine.runtime.speculative import (  # noqa: E402
    SpeculativeDecoder,
    accept_or_resample,
    ngram_propose,
)
from tests.conftest import TINY_RAW  # noqa: E402
from tests.test_generate import ByteTokenizer  # noqa: E402


def tiny(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    return LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).eval()


@pytest.fixture(scope="module")
def target():
    return tiny(0)


@pytest.fixture(scope="module")
def draft():
    return tiny(1)


def plain(model, prompt, params):
    return [o.token_id for o in generate(model, ByteTokenizer(), [prompt], params, ())]


def spec(target, draft, prompt, params, k=3):
    dec = SpeculativeDecoder(target, draft, num_speculative_tokens=k)
    return [o.token_id for o in dec.generate(ByteTokenizer(), prompt, params, ())], dec.stats


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("prompt", [[5], [3, 1, 4, 1, 5, 9, 2, 6], list(range(10, 40))])
def test_greedy_with_draft_model_equals_plain_greedy(target, draft, prompt, k):
    params = SamplingParams(temperature=0, max_tokens=25)
    got, stats = spec(target, draft, prompt, params, k)
    assert got == plain(target, prompt, params)
    assert stats.emitted == 25


def test_greedy_with_itself_as_draft_accepts_everything(target):
    params = SamplingParams(temperature=0, max_tokens=24)
    got, stats = spec(target, target, [7, 8, 9], params, k=4)
    assert got == plain(target, [7, 8, 9], params)
    assert stats.acceptance_rate == 1.0
    assert stats.rounds < 24  # several tokens per target pass


def test_greedy_with_ngram_lookup_equals_plain_greedy(target):
    prompt = [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7]  # repetitive: lookup finds matches
    params = SamplingParams(temperature=0, max_tokens=30)
    got, stats = spec(target, None, prompt, params, k=4)
    assert got == plain(target, prompt, params)
    assert stats.proposed > 0


def test_ngram_propose_continues_the_latest_match():
    assert ngram_propose([1, 2, 3, 9, 1, 2, 3, 7, 1, 2, 3], k=2) == [7, 1]
    assert ngram_propose([1, 2, 3], k=2) == []


def test_stop_token_and_max_tokens(target, draft):
    params = SamplingParams(temperature=0, max_tokens=20)
    free = plain(target, [4, 5], params)
    stop = free[5]
    dec = SpeculativeDecoder(target, draft, num_speculative_tokens=4)
    outs = list(dec.generate(ByteTokenizer(), [4, 5], params, (stop,)))
    assert [o.token_id for o in outs] == free[: free.index(stop) + 1]
    assert outs[-1].finish_reason == "stop"


def _chi_square_p_value(stat: float, df: int) -> float:
    """Upper tail of chi-square via the Wilson-Hilferty normal approximation."""
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def test_acceptance_rule_preserves_target_distribution():
    """10,000 draws of accept-or-resample must be distributed as p, whatever q is."""
    gen = torch.Generator().manual_seed(0)
    vocab = 12
    p = torch.softmax(torch.randn(vocab, generator=gen) * 1.5, 0)
    q = torch.softmax(torch.randn(vocab, generator=gen) * 1.5, 0)
    n = 10_000
    counts = torch.zeros(vocab)
    for _ in range(n):
        x = int(torch.multinomial(q, 1, generator=gen))
        _, tok = accept_or_resample(p, q, x, gen)
        counts[tok] += 1
    expected = p * n
    stat = float(((counts - expected) ** 2 / expected).sum())
    assert _chi_square_p_value(stat, vocab - 1) > 1e-3, (stat, counts, expected)


@pytest.mark.slow
def test_sampled_first_token_matches_target_end_to_end(target, draft):
    """The whole decoder, temperature 1: first-token frequencies vs the target's softmax."""
    prompt = [3, 9, 27, 81]
    probs = target(torch.tensor([prompt]), torch.arange(len(prompt))[None])[0, -1].softmax(-1)
    n, counts = 10_000, torch.zeros_like(probs)
    dec = SpeculativeDecoder(target, draft, num_speculative_tokens=1)
    for i in range(n):
        p = SamplingParams(temperature=1.0, max_tokens=2, seed=i)
        tok = next(iter(dec.generate(ByteTokenizer(), prompt, p, ()))).token_id
        counts[tok] += 1
    keep = probs * n >= 5
    expected = probs[keep] * n
    observed = counts[keep]
    observed_rest = n - observed.sum()
    stat = float(((observed - expected) ** 2 / expected).sum())
    rest = n - expected.sum()
    stat += float((observed_rest - rest) ** 2 / rest) if rest > 0 else 0.0
    assert _chi_square_p_value(stat, int(keep.sum())) > 1e-3, stat
