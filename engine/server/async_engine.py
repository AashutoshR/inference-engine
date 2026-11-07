"""Run the LLMEngine in one background thread and stream its outputs to asyncio callers.

One thread owns the engine (and the GPU): it admits new requests, applies aborts, calls
``step()`` and hands each output to the right request's asyncio queue. HTTP handlers
never touch the engine directly, so the engine needs no locks of its own.

``engine serve`` uses ``ProcessEngine`` (engine/server/process_engine.py) instead, which
keeps this request and stream handling but runs the engine in a process of its own.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from engine.runtime.engine import LLMEngine, RequestOutput
from engine.runtime.params import SamplingParams

log = logging.getLogger(__name__)


class QueueFull(RuntimeError):
    """Too many requests waiting; the API answers HTTP 429."""


class RequestTimeout(RuntimeError):
    """The request ran past its deadline and was aborted."""


@dataclass
class _Stream:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue


class AsyncEngine:
    def __init__(
        self,
        engine: LLMEngine,
        max_waiting: int = 256,
        request_timeout: float | None = 600.0,
        on_step: Callable[[dict, int], None] | None = None,
    ) -> None:
        self.engine = engine
        self.max_waiting = max_waiting
        self.request_timeout = request_timeout
        self.on_step = on_step  # metrics hook: (engine_stats(...), outputs this step)
        self._lock = threading.Lock()
        self._new: deque[tuple[str, list[int], SamplingParams]] = deque()
        self._aborts: deque[str] = deque()
        self._streams: dict[str, _Stream] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: BaseException | None = None
        # Latest engine.snapshot(), refreshed by the engine thread (the only thread that
        # may read engine state). Readers just take the reference.
        self.snapshot: dict | None = None
        self.snapshot_interval = 0.25
        self._snapshot_at = 0.0

    # --- lifecycle --------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="engine-loop", daemon=True)
            self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def capacity_tokens(self) -> int:
        return self.engine.scheduler.capacity_tokens

    @property
    def num_waiting(self) -> int:
        return len(self._new) + len(self.engine.scheduler.waiting)

    @property
    def num_running(self) -> int:
        return len(self.engine.scheduler.running)

    # --- requests ---------------------------------------------------------------------------

    async def generate(
        self, request_id: str, prompt_ids: list[int], params: SamplingParams
    ) -> AsyncIterator[RequestOutput]:
        """Yield this request's outputs as the engine produces them.

        Leaving early (client disconnect, exception, ``break``) aborts the request, which
        frees its KV blocks within one engine step.
        """
        if self.num_waiting >= self.max_waiting:
            raise QueueFull(f"{self.num_waiting} requests already waiting")
        stream = _Stream(asyncio.get_running_loop(), asyncio.Queue())
        with self._lock:
            if request_id in self._streams:
                raise ValueError(f"duplicate request id {request_id!r}")
            self._streams[request_id] = stream
        self._submit(request_id, list(prompt_ids), params)

        deadline = None if self.request_timeout is None else time.monotonic() + self.request_timeout
        finished = False
        try:
            while True:
                timeout = None if deadline is None else max(deadline - time.monotonic(), 0)
                try:
                    item = await asyncio.wait_for(stream.queue.get(), timeout)
                except TimeoutError:
                    raise RequestTimeout(f"request {request_id} timed out") from None
                if isinstance(item, BaseException):
                    finished = True  # the engine already dropped it
                    raise item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            with self._lock:
                self._streams.pop(request_id, None)
            if not finished:
                self.abort(request_id)

    def _submit(self, request_id: str, prompt_ids: list[int], params: SamplingParams) -> None:
        with self._lock:
            self._new.append((request_id, prompt_ids, params))
        self._wake.set()

    def abort(self, request_id: str) -> None:
        with self._lock:
            self._aborts.append(request_id)
        self._wake.set()

    # --- the engine thread ------------------------------------------------------------------

    def _refresh_snapshot(self, every: float) -> None:
        now = time.monotonic()
        if now - self._snapshot_at >= every:
            self._snapshot_at = now
            try:
                self.snapshot = self.engine.snapshot()
            except Exception:  # introspection must never take the server down
                log.exception("engine snapshot failed")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._admit_and_abort()
            if not self.engine.has_unfinished():
                self._refresh_snapshot(1.0)
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            start = time.perf_counter()
            try:
                outputs = self.engine.step()
            except Exception as exc:  # e.g. a CUDA error: fail the requests, keep serving
                log.exception("engine step failed; failing all in-flight requests")
                self.last_error = exc
                self._fail_all(exc)
                continue
            if self.on_step is not None:
                self.on_step(engine_stats(self.engine, time.perf_counter() - start), len(outputs))
            self._deliver_step(outputs)
            self._refresh_snapshot(self.snapshot_interval)

    def _admit_and_abort(self) -> None:
        with self._lock:
            new, self._new = self._new, deque()
            aborts, self._aborts = self._aborts, deque()
        for request_id, prompt, params in new:
            try:
                self.engine.add_request(request_id, prompt, params)
            except ValueError as exc:
                self._deliver(request_id, exc)
        for request_id in aborts:
            self.engine.abort(request_id)

    def _fail_all(self, exc: BaseException) -> None:
        for seq in list(self.engine.scheduler.running) + list(self.engine.scheduler.waiting):
            self.engine.abort(seq.request_id)
            self._deliver(seq.request_id, RuntimeError(f"engine error: {exc}"))

    def _deliver_step(self, outputs: list[RequestOutput]) -> None:
        """Hand a whole step's outputs over with one wake-up per event loop, not one per
        token: each wake-up is a socket write, and the loop then runs once per step."""
        by_loop: dict[asyncio.AbstractEventLoop, list[tuple[asyncio.Queue, object]]] = {}
        with self._lock:
            for out in outputs:
                stream = self._streams.get(out.request_id)
                if stream is not None:  # None: the caller already left; its abort is queued
                    by_loop.setdefault(stream.loop, []).append((stream.queue, out))
        for loop, items in by_loop.items():
            loop.call_soon_threadsafe(_put_all, items)

    def _deliver(self, request_id: str, item: object) -> None:
        with self._lock:
            stream = self._streams.get(request_id)
        if stream is None:
            return  # the caller already left; its abort is queued
        stream.loop.call_soon_threadsafe(stream.queue.put_nowait, item)


def _put_all(items: list[tuple[asyncio.Queue, object]]) -> None:
    for queue, item in items:
        queue.put_nowait(item)


def engine_stats(engine: LLMEngine, seconds: float) -> dict:
    """What the metrics need after a step, as plain data (it crosses a process boundary)."""
    sched, bm = engine.scheduler, engine.block_manager
    prompt = engine.prompt_tokens_total
    return {
        "seconds": seconds,
        "running": len(sched.running),
        "waiting": len(sched.waiting),
        "kv_used": 1 - bm.num_free_blocks / bm.allocator.num_blocks,
        "preemptions": sched.num_preemptions,
        "prefix_hit": engine.cached_prompt_tokens_total / prompt if prompt else None,
    }
