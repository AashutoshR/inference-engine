"""Phase 2: the OpenAI-compatible API, end to end on the tiny model (CPU)."""

from __future__ import annotations

import asyncio
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from engine.config import ModelConfig  # noqa: E402
from engine.model.llama import LlamaForCausalLM  # noqa: E402
from engine.runtime.engine import LLMEngine  # noqa: E402
from engine.runtime.params import SamplingParams  # noqa: E402
from engine.server.api import create_app  # noqa: E402
from engine.server.async_engine import AsyncEngine  # noqa: E402
from tests.conftest import TINY_RAW  # noqa: E402
from tests.test_generate import ByteTokenizer  # noqa: E402


class ChatByteTokenizer(ByteTokenizer):
    """ASCII bytes as tokens (the tiny vocab is 128), with a minimal chat template."""

    def apply_chat_template(self, messages, add_generation_prompt=True, **_):
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        return self.encode(text + ("<assistant>" if add_generation_prompt else ""))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).eval()


def make_engine(model, num_blocks=128):
    return LLMEngine(model, ChatByteTokenizer(), (), num_blocks=num_blocks, block_size=4)


@pytest.fixture
def client(model):
    app = create_app(make_engine(model), ChatByteTokenizer(), "tiny", request_timeout=30)
    with TestClient(app) as c:
        yield c


def chat_body(**kw):
    return {"model": "tiny", "messages": [{"role": "user", "content": "Hi there"}],
            "temperature": 0, "max_tokens": 6, **kw}  # fmt: skip


def sse_events(response) -> list:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[len("data: ") :]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_chat_completion_matches_engine(client, model):
    r = client.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 6

    tok = ChatByteTokenizer()
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hi there"}])
    ids = make_engine(model).generate({"x": prompt}, SamplingParams(temperature=0, max_tokens=6))
    assert choice["message"]["content"] == tok.decode(ids["x"])
    assert body["usage"]["prompt_tokens"] == len(prompt)


def test_streaming_equals_non_streaming(client):
    full = client.post("/v1/chat/completions", json=chat_body()).json()
    with client.stream(
        "POST", "/v1/chat/completions",
        json=chat_body(stream=True, stream_options={"include_usage": True}),
    ) as r:  # fmt: skip
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == full["choices"][0]["message"]["content"]
    finals = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert finals[-1] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 6


def test_completions_accepts_token_ids(client):
    r = client.post("/v1/completions", json={"prompt": [5, 6, 7], "max_tokens": 4,
                                             "temperature": 0})  # fmt: skip
    assert r.status_code == 200, r.text
    assert r.json()["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_stop_string_ends_generation(client):
    full = client.post("/v1/chat/completions", json=chat_body(max_tokens=20)).json()
    text = full["choices"][0]["message"]["content"]
    stop = text[4:6]
    r = client.post("/v1/chat/completions", json=chat_body(max_tokens=20, stop=[stop])).json()
    assert r["choices"][0]["message"]["content"] == text[: text.find(stop)]
    assert r["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "patch",
    [{"n": 2}, {"temperature": -1}, {"messages": []}, {"top_p": 0}, {"max_tokens": 0}],
)
def test_bad_requests_get_400(client, patch):
    r = client.post("/v1/chat/completions", json=chat_body(**patch))
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_prompt_longer_than_cache_is_400(model):
    app = create_app(make_engine(model, num_blocks=2), ChatByteTokenizer(), "tiny")
    with TestClient(app) as c:
        r = c.post("/v1/completions", json={"prompt": [1] * 20, "max_tokens": 1})
    assert r.status_code == 400


def test_full_queue_gets_429(model):
    app = create_app(make_engine(model), ChatByteTokenizer(), "tiny", max_waiting=0)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"


def test_models_health_and_metrics(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"
    assert client.get("/health").status_code == 200
    client.post("/v1/chat/completions", json=chat_body())
    metrics = client.get("/metrics").text
    assert 'engine_requests_total{status="ok"} 1.0' in metrics
    assert "engine_time_to_first_token_seconds_count 1.0" in metrics
    assert "engine_kv_blocks_used_ratio" in metrics


def test_leaving_a_stream_early_frees_kv_blocks(model):
    """FR-10: a client that disconnects stops using the GPU within one step."""
    engine = make_engine(model)
    runner = AsyncEngine(engine)

    async def scenario():
        runner.start()
        try:
            stream = runner.generate("r", [5, 6, 7], SamplingParams(temperature=0, max_tokens=200))
            got = [await anext(stream) for _ in range(3)]
            assert engine.num_free_blocks < 128
            await stream.aclose()  # what Starlette does when the client goes away
            for _ in range(200):
                if not engine.has_unfinished():
                    break
                await asyncio.sleep(0.01)
            return got
        finally:
            runner.shutdown()

    got = asyncio.run(scenario())
    assert len(got) == 3
    assert not engine.has_unfinished()
    assert engine.num_free_blocks == 128


def test_timeout_aborts_request(model):
    engine = make_engine(model)
    app = create_app(engine, ChatByteTokenizer(), "tiny", request_timeout=1e-6)
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json=chat_body(max_tokens=50))
        assert r.status_code == 504
        for _ in range(200):
            if not engine.has_unfinished():
                break
            asyncio.run(asyncio.sleep(0.01))
    assert engine.num_free_blocks == 128


def test_dashboard_and_state(client):
    page = client.get("/dashboard")
    assert page.status_code == 200 and "<canvas" in page.text
    assert client.get("/", follow_redirects=False).status_code in (302, 307)
    client.post("/v1/chat/completions", json=chat_body(max_tokens=3))
    for _ in range(100):  # the engine thread refreshes the snapshot every second when idle
        r = client.get("/debug/state")
        if r.status_code == 200 and r.json()["totals"]["finished_requests"] >= 1:
            break
        asyncio.run(asyncio.sleep(0.05))
    state = r.json()
    assert len(state["blocks"]) == state["config"]["num_blocks"] == len(state["owners"])
    assert set(state["blocks"]) <= set(".cos")
    assert state["totals"]["generated_tokens"] >= 3
    assert state["steps"] and {"ms", "prefill_tokens", "decode_tokens"} <= set(state["steps"][-1])


def test_chat_logprobs_match_the_model(client, model):
    r = client.post("/v1/chat/completions",
                    json=chat_body(max_tokens=4, logprobs=True, top_logprobs=3)).json()  # fmt: skip
    entries = r["choices"][0]["logprobs"]["content"]
    assert len(entries) == r["usage"]["completion_tokens"] == 4
    tok = ChatByteTokenizer()
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hi there"}])
    want = model(torch.tensor([prompt]), torch.arange(len(prompt))[None])[0, -1].log_softmax(-1)
    first = entries[0]
    assert abs(first["logprob"] - float(want.max())) < 1e-4  # greedy picks the top token
    assert [t["logprob"] for t in first["top_logprobs"]] == sorted(
        (t["logprob"] for t in first["top_logprobs"]), reverse=True
    )
    assert len(first["top_logprobs"]) == 3 and first["top_logprobs"][0]["token"] == first["token"]


def test_streamed_logprobs_one_per_token(client):
    with client.stream("POST", "/v1/chat/completions",
                       json=chat_body(max_tokens=5, logprobs=True, stream=True)) as r:  # fmt: skip
        events = [e for e in sse_events(r) if e != "[DONE]"]
    per_token = [e["choices"][0]["logprobs"]["content"] for e in events
                 if e["choices"] and "logprobs" in e["choices"][0]]  # fmt: skip
    assert sum(len(c) for c in per_token) == 5


def test_legacy_completion_logprobs(client):
    r = client.post("/v1/completions", json={"prompt": [5, 6, 7], "max_tokens": 3,
                                             "temperature": 0, "logprobs": 2}).json()  # fmt: skip
    lp = r["choices"][0]["logprobs"]
    assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == 3
    assert all(len(alts) <= 2 for alts in lp["top_logprobs"])


@pytest.mark.parametrize("patch", [{"logprobs": True, "top_logprobs": 21}, {"top_logprobs": 2}])
def test_bad_logprob_requests_get_400(client, patch):
    assert client.post("/v1/chat/completions", json=chat_body(**patch)).status_code == 400


def tiny_engine() -> LLMEngine:
    """Module level, so a spawned engine process can import and call it."""
    torch.manual_seed(0)
    return make_engine(LlamaForCausalLM(ModelConfig.from_dict(TINY_RAW)).eval())


def test_engine_in_its_own_process_serves_the_same_answers(model):
    """`engine serve` runs the engine in a separate process: same API, same tokens, and
    the engine process exits on shutdown."""
    from engine.server.process_engine import ProcessEngine

    runner = ProcessEngine(tiny_engine)
    proc = runner._proc
    try:
        app = create_app(runner, ChatByteTokenizer(), "tiny", request_timeout=30)
        with TestClient(app) as c:
            assert c.get("/health").status_code == 200
            got = c.post("/v1/chat/completions", json=chat_body()).json()
            with c.stream("POST", "/v1/chat/completions", json=chat_body(stream=True)) as r:
                events = sse_events(r)
            too_long = c.post("/v1/completions",
                              json={"prompt": [1] * 600, "max_tokens": 1})  # fmt: skip
            for _ in range(100):
                state = c.get("/debug/state")
                if state.status_code == 200 and state.json()["totals"]["finished_requests"] >= 2:
                    break
                asyncio.run(asyncio.sleep(0.05))
            metrics = c.get("/metrics").text
    finally:
        runner.shutdown()
    assert not proc.is_alive()
    with TestClient(create_app(make_engine(model), ChatByteTokenizer(), "tiny")) as c:
        want = c.post("/v1/chat/completions", json=chat_body()).json()
    assert got["choices"] == want["choices"] and got["usage"] == want["usage"]
    streamed = "".join(e["choices"][0]["delta"].get("content", "")
                       for e in events[:-1] if e["choices"])  # fmt: skip
    assert streamed == want["choices"][0]["message"]["content"]
    assert too_long.status_code == 400
    assert state.json()["totals"]["finished_requests"] >= 2
    assert "engine_generation_tokens_total 12.0" in metrics
