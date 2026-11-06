"""OpenAI-compatible endpoints (Phase 2, FR-9 and FR-10).

    POST /v1/chat/completions   chat, streamed (SSE) or not
    POST /v1/completions        raw prompt (text or token ids), streamed or not
    GET  /v1/models             the one served model
    GET  /health                200 while the engine thread is alive
    GET  /metrics               Prometheus
    GET  /dashboard             live view of the engine: KV blocks, scheduler, requests
    GET  /debug/state           the data behind the dashboard, as JSON

Point the OpenAI SDK at it with ``base_url="http://host:port/v1"`` and any API key.
Extensions beyond OpenAI's schema: ``top_k``, ``min_p``, ``repetition_penalty``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from engine.runtime.chat import chat_prompt_ids
from engine.runtime.engine import LLMEngine, RequestOutput
from engine.runtime.params import SamplingParams
from engine.server.async_engine import AsyncEngine, QueueFull, RequestTimeout

DEFAULT_MAX_TOKENS = 512

# --- request schemas ----------------------------------------------------------------------


class _SamplingFields(BaseModel):
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0, lt=1)
    repetition_penalty: float = Field(default=1.0, gt=0)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    n: int = 1
    user: str | None = None
    # Chat: logprobs is a bool and top_logprobs a count. Completions (legacy): logprobs is
    # the count of alternatives.
    logprobs: bool | int | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)

    def logprob_settings(self) -> tuple[bool, int]:
        if isinstance(self.logprobs, bool) or self.logprobs is None:
            return bool(self.logprobs), self.top_logprobs or 0
        return True, int(self.logprobs)  # legacy completions: logprobs=N

    def sampling_params(self, max_tokens: int | None = None) -> SamplingParams:
        stop = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop or ())
        want, top = self.logprob_settings()
        return SamplingParams(
            logprobs=want,
            top_logprobs=top,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=max_tokens or self.max_tokens or DEFAULT_MAX_TOKENS,
            seed=self.seed,
            stop=stop,
        )


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str | list[dict[str, Any]]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(p.get("text", "") for p in self.content if p.get("type") == "text")


class ChatRequest(_SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]


# --- metrics ------------------------------------------------------------------------------


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter("engine_requests_total", "Requests by outcome", ["status"],
                                registry=r)  # fmt: skip
        self.prompt_tokens = Counter("engine_prompt_tokens_total", "Prompt tokens", registry=r)
        self.gen_tokens = Counter("engine_generation_tokens_total", "Generated tokens", registry=r)
        self.ttft = Histogram("engine_time_to_first_token_seconds", "TTFT", registry=r,
                              buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30))  # fmt: skip
        self.itl = Histogram("engine_inter_token_latency_seconds", "ITL", registry=r,
                             buckets=(.005, .01, .02, .04, .08, .16, .32, .64, 1.28))  # fmt: skip
        self.step_time = Histogram("engine_step_seconds", "Engine step time", registry=r)
        self.running = Gauge("engine_running_sequences", "Running sequences", registry=r)
        self.waiting = Gauge("engine_waiting_sequences", "Queued sequences", registry=r)
        self.kv_used = Gauge("engine_kv_blocks_used_ratio", "KV blocks in use", registry=r)
        self.preemptions = Gauge("engine_preemptions_total", "Preemptions so far", registry=r)
        self.prefix_hit = Gauge(
            "engine_prefix_cache_hit_ratio",
            "Prompt tokens served from the prefix cache",
            registry=r,
        )

    def on_step(self, engine: LLMEngine, outputs: list[RequestOutput], seconds: float) -> None:
        self.step_time.observe(seconds)
        self.gen_tokens.inc(len(outputs))
        sched, bm = engine.scheduler, engine.block_manager
        self.running.set(len(sched.running))
        self.waiting.set(len(sched.waiting))
        self.kv_used.set(1 - bm.num_free_blocks / bm.allocator.num_blocks)
        self.preemptions.set(sched.num_preemptions)
        if engine.prompt_tokens_total:
            self.prefix_hit.set(engine.cached_prompt_tokens_total / engine.prompt_tokens_total)


# --- app ----------------------------------------------------------------------------------


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status)


def create_app(
    engine: LLMEngine,
    tokenizer: Any,
    model_name: str,
    *,
    max_waiting: int = 256,
    request_timeout: float | None = 600.0,
) -> FastAPI:
    metrics = Metrics()
    runner = AsyncEngine(engine, max_waiting, request_timeout, on_step=metrics.on_step)
    capacity = engine.scheduler.capacity_tokens

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runner.start()
        yield
        runner.shutdown()

    app = FastAPI(title="engine", lifespan=lifespan)
    app.state.runner = runner
    app.state.metrics = metrics

    @app.exception_handler(RequestValidationError)
    async def _bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(
            400, f"{where}: {first.get('msg', 'invalid request')}", "invalid_request_error"
        )

    @app.get("/health")
    async def health() -> Response:
        alive = runner._thread is not None and runner._thread.is_alive()
        return Response(status_code=200 if alive else 503)

    dashboard_html = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/", include_in_schema=False)
    async def root() -> Response:
        return RedirectResponse("/dashboard")

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(dashboard_html)

    @app.get("/debug/state")
    async def debug_state() -> Response:
        snap = runner.snapshot
        if snap is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        return JSONResponse(snap)

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model",
                                            "owned_by": "engine", "created": 0}]}  # fmt: skip

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request) -> Response:
        messages = [{"role": m.role, "content": m.text()} for m in req.messages]
        try:
            prompt_ids = chat_prompt_ids(tokenizer, messages)
        except Exception as exc:  # template errors, e.g. bad role order
            return _error(400, f"chat template: {exc}", "invalid_request_error")
        max_tokens = req.max_completion_tokens or req.max_tokens
        return await _complete(req, request, prompt_ids, max_tokens, chat=True)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request) -> Response:
        prompt_ids = tokenizer.encode(req.prompt) if isinstance(req.prompt, str) else req.prompt
        return await _complete(req, request, list(prompt_ids), req.max_tokens, chat=False)

    async def _complete(
        req: _SamplingFields, request: Request, prompt_ids: list[int], max_tokens: int | None,
        *, chat: bool,
    ) -> Response:  # fmt: skip
        if req.n != 1:
            return _error(400, "only n=1 is supported", "invalid_request_error")
        if not prompt_ids:
            return _error(400, "prompt is empty", "invalid_request_error")
        room = capacity - len(prompt_ids)
        if room < 1:
            return _error(400, f"prompt has {len(prompt_ids)} tokens; the KV cache holds "
                               f"{capacity}", "invalid_request_error")  # fmt: skip
        if max_tokens is None:
            max_tokens = min(DEFAULT_MAX_TOKENS, room)
        try:
            params = req.sampling_params(max_tokens)
        except ValueError as exc:
            return _error(400, str(exc), "invalid_request_error")

        prefix = "chatcmpl" if chat else "cmpl"
        request_id = f"{prefix}-{uuid.uuid4().hex}"
        created = int(time.time())
        model = req.model or model_name
        metrics.prompt_tokens.inc(len(prompt_ids))
        arrival = time.perf_counter()
        try:
            stream = runner.generate(request_id, prompt_ids, params)
            first = await anext(stream)  # surfaces QueueFull / bad request before headers go out
        except QueueFull as exc:
            metrics.requests.labels("rejected").inc()
            return _error(429, f"server busy: {exc}", "rate_limit_error")
        except ValueError as exc:
            metrics.requests.labels("invalid").inc()
            return _error(400, str(exc), "invalid_request_error")
        except RequestTimeout as exc:
            metrics.requests.labels("timeout").inc()
            return _error(504, str(exc), "timeout")
        metrics.ttft.observe(time.perf_counter() - arrival)

        if req.stream:
            include_usage = bool((req.stream_options or {}).get("include_usage"))
            body = _sse(stream, first, request_id, created, model, chat, include_usage,
                        params.logprobs)  # fmt: skip
            return StreamingResponse(body, media_type="text/event-stream")

        text, last, outs = first.text, first, [first]
        try:
            async for out in stream:
                text, last = text + out.text, out
                outs.append(out)
                if await request.is_disconnected():
                    await stream.aclose()  # aborts the request
                    metrics.requests.labels("cancelled").inc()
                    return Response(status_code=499)
        except RequestTimeout as exc:
            metrics.requests.labels("timeout").inc()
            return _error(504, str(exc), "timeout")
        except RuntimeError as exc:
            metrics.requests.labels("error").inc()
            return _error(500, str(exc), "server_error")
        metrics.requests.labels("ok").inc()
        usage = _usage(last)
        if chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": text},
                      "finish_reason": last.finish_reason,
                      "logprobs": chat_logprobs(outs) if params.logprobs else None}  # fmt: skip
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "finish_reason": last.finish_reason,
                      "logprobs": legacy_logprobs(outs) if params.logprobs else None}  # fmt: skip
            obj = "text_completion"
        return JSONResponse({"id": request_id, "object": obj, "created": created,
                             "model": model, "choices": [choice], "usage": usage})  # fmt: skip

    def token_text(token_id: int) -> str:
        return tokenizer.decode([token_id])

    def lp_entry(token_id: int, logprob: float) -> dict:
        t = token_text(token_id)
        return {"token": t, "logprob": logprob, "bytes": list(t.encode("utf-8"))}

    def chat_logprobs(outs: list[RequestOutput]) -> dict:
        return {"content": [
            {**lp_entry(o.token_id, o.logprob),
             "top_logprobs": [lp_entry(i, v) for i, v in o.top_logprobs or ()]}
            for o in outs if o.logprob is not None
        ]}  # fmt: skip

    def legacy_logprobs(outs: list[RequestOutput]) -> dict:
        outs = [o for o in outs if o.logprob is not None]
        return {
            "tokens": [token_text(o.token_id) for o in outs],
            "token_logprobs": [o.logprob for o in outs],
            "top_logprobs": [{token_text(i): v for i, v in o.top_logprobs or ()} for o in outs],
        }

    async def _sse(
        stream: AsyncIterator[RequestOutput], first: RequestOutput,
        request_id: str, created: int, model: str, chat: bool, include_usage: bool,
        with_logprobs: bool,
    ) -> AsyncIterator[bytes]:  # fmt: skip
        obj = "chat.completion.chunk" if chat else "text_completion"

        def chunk(out: RequestOutput | None, role: bool = False) -> bytes:
            if chat:
                delta: dict[str, str] = {"role": "assistant"} if role else {}
                if out is not None and out.text:
                    delta["content"] = out.text
                choice = {"index": 0, "delta": delta,
                          "finish_reason": out.finish_reason if out else None}  # fmt: skip
                if with_logprobs and out is not None:
                    choice["logprobs"] = chat_logprobs([out])
            else:
                choice = {"index": 0, "text": out.text if out else "",
                          "finish_reason": out.finish_reason if out else None}  # fmt: skip
                if with_logprobs and out is not None:
                    choice["logprobs"] = legacy_logprobs([out])
            payload = {"id": request_id, "object": obj, "created": created, "model": model,
                       "choices": [choice]}  # fmt: skip
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

        last, prev_t = first, time.perf_counter()
        try:
            if chat:
                yield chunk(None, role=True)
            yield chunk(first)
            if not first.finished:
                async for out in stream:
                    now = time.perf_counter()
                    metrics.itl.observe(now - prev_t)
                    prev_t, last = now, out
                    if out.text or out.finished or with_logprobs:
                        yield chunk(out)
            if include_usage:
                payload = {"id": request_id, "object": obj, "created": created, "model": model,
                           "choices": [], "usage": _usage(last)}  # fmt: skip
                yield f"data: {json.dumps(payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            metrics.requests.labels("ok").inc()
        except (RequestTimeout, RuntimeError) as exc:
            metrics.requests.labels("error").inc()
            err = {"error": {"message": str(exc), "type": "server_error"}}
            yield f"data: {json.dumps(err)}\n\n".encode()
        finally:
            await stream.aclose()  # client gone mid-stream -> abort and free KV blocks

    return app


def _usage(out: RequestOutput) -> dict[str, int]:
    return {"prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens,
            "total_tokens": out.prompt_tokens + out.completion_tokens}  # fmt: skip
