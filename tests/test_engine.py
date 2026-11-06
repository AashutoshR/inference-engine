"""Phase 2: the paged, continuously batched engine must produce exactly what the Phase 1
static loop produces, however requests arrive, get preempted or get cancelled."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from engine.model.llama import LlamaForCausalLM  # noqa: E402
from engine.runtime.engine import LLMEngine  # noqa: E402
from engine.runtime.generate import generate  # noqa: E402
from engine.runtime.params import SamplingParams  # noqa: E402
from tests.test_generate import ByteTokenizer  # noqa: E402

GREEDY = SamplingParams(temperature=0, max_tokens=12)
PROMPTS = {
    "a": [5, 6, 7],
    "b": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    "c": [99],
    "d": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
}


@pytest.fixture(scope="module")
def tiny_model():
    from engine.config import ModelConfig
    from tests.conftest import TINY_RAW

    torch.manual_seed(0)
    return LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).eval()


@pytest.fixture(scope="module")
def expected(tiny_model):
    """Phase 1 static loop, one prompt at a time: the reference."""
    return {
        rid: [o.token_id for o in generate(tiny_model, ByteTokenizer(), [p], GREEDY, ())]
        for rid, p in PROMPTS.items()
    }


def make_engine(model, num_blocks=64, block_size=4, **kw) -> LLMEngine:
    kw.setdefault("max_num_batched_tokens", 4096)
    return LLMEngine(
        model, ByteTokenizer(), (), num_blocks=num_blocks, block_size=block_size,
        debug_checks=True, **kw,
    )  # fmt: skip


def test_paged_forward_matches_full_forward(tiny_model):
    """One prefill step through the paged path gives the same last-token logits."""
    from engine.runtime.paged_cache import PagedBatch

    eng = make_engine(tiny_model)
    seqs = list(PROMPTS.values())
    tables = [eng.block_manager.grow(i, len(p)) for i, p in enumerate(seqs)]
    batch = PagedBatch.build(
        eng.cache, tables, [len(p) for p in seqs], [len(p) for p in seqs], "cpu"
    )
    flat = torch.tensor([[t for p in seqs for t in p]])
    pos = torch.tensor([[i for p in seqs for i in range(len(p))]])
    got = tiny_model(flat, pos, batch)[:, 0]
    for row, p in enumerate(seqs):
        want = tiny_model(torch.tensor([p]), torch.arange(len(p))[None])[0, -1]
        torch.testing.assert_close(got[row], want, atol=1e-5, rtol=1e-5)


def test_all_at_once_matches_static_loop(tiny_model, expected):
    eng = make_engine(tiny_model)
    assert eng.generate(PROMPTS, GREEDY) == expected
    assert eng.num_free_blocks == 64


def test_staggered_arrivals_match(tiny_model, expected):
    """Requests join mid-flight: prefill and decode steps interleave."""
    eng = make_engine(tiny_model)
    got: dict[str, list[int]] = {rid: [] for rid in PROMPTS}
    pending = list(PROMPTS.items())
    while pending or eng.has_unfinished():
        if pending:
            rid, prompt = pending.pop(0)
            eng.add_request(rid, prompt, GREEDY)
        for out in eng.step():
            got[out.request_id].append(out.token_id)
    assert got == expected
    assert eng.num_free_blocks == 64


def test_preemption_recomputes_and_still_matches(tiny_model, expected):
    """A cache too small for everyone forces preemption; output must not change."""
    eng = make_engine(tiny_model, num_blocks=10, block_size=4)  # 40 slots for ~90 tokens
    assert eng.generate(PROMPTS, GREEDY) == expected
    assert eng.scheduler.num_preemptions > 0
    assert eng.num_free_blocks == 10


def test_abort_frees_blocks_at_once(tiny_model):
    eng = make_engine(tiny_model)
    eng.add_request("x", PROMPTS["b"], SamplingParams(temperature=0, max_tokens=50))
    eng.add_request("y", PROMPTS["a"], SamplingParams(temperature=0, max_tokens=50))
    eng.step()
    eng.step()
    assert eng.num_free_blocks < 64
    assert eng.abort("x")
    assert not eng.abort("x")
    while eng.has_unfinished():
        assert all(o.request_id == "y" for o in eng.step())
    assert eng.num_free_blocks == 64


def test_max_num_seqs_limits_the_batch(tiny_model, expected):
    eng = make_engine(tiny_model, max_num_seqs=2)
    eng_out = {rid: [] for rid in PROMPTS}
    for rid, p in PROMPTS.items():
        eng.add_request(rid, p, GREEDY)
    while eng.has_unfinished():
        assert len(eng.scheduler.running) <= 2
        for out in eng.step():
            eng_out[out.request_id].append(out.token_id)
    assert eng_out == expected


def test_request_too_long_for_cache_is_rejected(tiny_model):
    eng = make_engine(tiny_model, num_blocks=2, block_size=4)
    with pytest.raises(ValueError, match="exceeds (max_model_len|the KV cache)"):
        eng.add_request("big", [1] * 5, SamplingParams(max_tokens=10))


def test_seeded_sampling_is_independent_of_batching(tiny_model):
    params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=10, seed=7)
    alone = make_engine(tiny_model).generate({"s": [8, 9]}, params)["s"]
    crowd = make_engine(tiny_model).generate({"s": [8, 9], "t": [1, 2, 3], "u": [4]}, params)
    assert crowd["s"] == alone


def test_sequences_finishing_at_different_steps(tiny_model):
    """Regression: a sequence finishing mid-step must not shift the rows of the others."""
    lengths = {"a": 3, "b": 12, "c": 7, "d": 1}
    want = {
        rid: [
            o.token_id
            for o in generate(
                tiny_model,
                ByteTokenizer(),
                [PROMPTS[rid]],
                SamplingParams(temperature=0, max_tokens=n),
                (),
            )  # fmt: skip
        ]
        for rid, n in lengths.items()
    }
    eng = make_engine(tiny_model)
    for rid, n in lengths.items():
        eng.add_request(rid, PROMPTS[rid], SamplingParams(temperature=0, max_tokens=n))
    got = {rid: [] for rid in lengths}
    while eng.has_unfinished():
        for out in eng.step():
            got[out.request_id].append(out.token_id)
    assert got == want


# --- Phase 3: chunked prefill and prefix caching ------------------------------------------


def test_chunked_prefill_matches(tiny_model, expected):
    """A 5-token budget splits the 18-token prompt over several steps."""
    eng = make_engine(tiny_model, max_num_seqs=4, max_num_batched_tokens=5)
    steps = 0
    got = {rid: [] for rid in PROMPTS}
    for rid, p in PROMPTS.items():
        eng.add_request(rid, p, GREEDY)
    while eng.has_unfinished():
        for out in eng.step():
            got[out.request_id].append(out.token_id)
        steps += 1
    assert got == expected
    assert steps > GREEDY.max_tokens  # the long prompt took extra steps


def test_prefix_cache_reuses_blocks_and_output_is_unchanged(tiny_model):
    shared = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # three 4-token blocks
    prompts = {"first": shared + [1, 2], "second": shared + [3], "third": shared}
    want = {
        rid: [o.token_id for o in generate(tiny_model, ByteTokenizer(), [p], GREEDY, ())]
        for rid, p in prompts.items()
    }
    eng = make_engine(tiny_model)
    got = {}
    for rid, p in prompts.items():  # one after another, so later ones find the cache warm
        got.update(eng.generate({rid: p}, GREEDY))
    assert got == want
    # "second" reuses all three shared blocks; "third" (exactly the shared part) must still
    # compute its last token, so it reuses only two.
    assert eng.cached_prompt_tokens_total == 12 + 8
    assert eng.block_manager.num_cached_blocks > 0


def test_prefix_cache_off_gives_same_output(tiny_model, expected):
    eng = make_engine(tiny_model, enable_prefix_caching=False)
    assert eng.generate(PROMPTS, GREEDY) == expected
    assert eng.cached_prompt_tokens_total == 0


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_random_workloads_match_static_loop(tiny_model, data):
    """Random cache sizes, budgets, shared prefixes, arrivals and cancellations: every
    surviving request must produce exactly the static loop's tokens, with no leaks."""
    block_size = data.draw(st.sampled_from([1, 2, 4, 8]))
    max_seqs = data.draw(st.integers(1, 4))
    budget = data.draw(st.integers(max_seqs, 24))
    stems = [[3, 4, 5, 6, 7, 8, 9, 10], [20, 21, 22, 23, 24, 25]]
    reqs = {}
    for i in range(data.draw(st.integers(1, 6))):
        stem = data.draw(st.sampled_from(stems))[: data.draw(st.integers(0, 8))]
        tail = data.draw(st.lists(st.integers(30, 120), min_size=1, max_size=6))
        reqs[f"r{i}"] = (stem + tail, data.draw(st.integers(1, 8)))
    longest = max(len(p) + n for p, n in reqs.values())
    num_blocks = data.draw(st.integers(-(-longest // block_size), 40))
    eng = make_engine(
        tiny_model, num_blocks=num_blocks, block_size=block_size, max_num_seqs=max_seqs,
        max_num_batched_tokens=budget,
    )  # fmt: skip
    cancel = {rid for rid in reqs if data.draw(st.integers(0, 4)) == 0}

    got = {rid: [] for rid in reqs}
    pending = list(reqs)
    step = 0
    while pending or eng.has_unfinished():
        if pending and (not eng.has_unfinished() or data.draw(st.booleans())):
            rid = pending.pop(0)
            prompt, n = reqs[rid]
            eng.add_request(rid, prompt, SamplingParams(temperature=0, max_tokens=n))
        for out in eng.step():
            got[out.request_id].append(out.token_id)
        step += 1
        for rid in list(cancel):
            if got[rid]:
                eng.abort(rid)
                cancel.discard(rid)
                got.pop(rid)
        assert step < 2000, "no progress"
    assert eng.num_free_blocks == num_blocks
    for rid, tokens in got.items():
        prompt, n = reqs[rid]
        params = SamplingParams(temperature=0, max_tokens=n)
        want = [o.token_id for o in generate(tiny_model, ByteTokenizer(), [prompt], params, ())]
        assert tokens == want, rid
