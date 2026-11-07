"""Run the LLMEngine in a process of its own, so HTTP work never waits on it (or it on HTTP).

In one process the engine thread and the HTTP thread share one GIL. Streaming a token
costs the HTTP side a few hundred microseconds of Python (JSON, Starlette, a socket
write), and with 16 to 64 streams that work keeps taking the GIL in the middle of the
engine's step, which only needs a few milliseconds of Python but needs them at once. At
16 users this held the engine to 26 ms per step against 13.6 ms with no HTTP at all.

So the engine gets its own process and its own GIL, as in vLLM. The HTTP process keeps
everything ``AsyncEngine`` does for callers (streams, timeouts, aborts on disconnect) and
talks to the engine over two pipes:

    requests  HTTP -> engine   ("add", id, prompt ids, params), ("abort", id), ("stop",)
    results   engine -> HTTP   ("ready", info), ("step", outputs, stats), ("ack", n),
                               ("snapshot", dashboard dict), ("error", id, exception),
                               ("fail", [ids], message), ("stopped",)

One "step" message carries the whole step's outputs (token ids and detokenized text), so
each step costs one pickle and one pipe write. If the HTTP process dies, the engine
process sees its pipe close and exits, so no orphan keeps holding the GPU.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from typing import Any

from engine.runtime.engine import LLMEngine
from engine.runtime.params import SamplingParams
from engine.server.async_engine import AsyncEngine, engine_stats

log = logging.getLogger(__name__)


class ProcessEngine(AsyncEngine):
    def __init__(
        self,
        factory: Callable[[], LLMEngine],
        max_waiting: int = 256,
        request_timeout: float | None = 600.0,
        startup_timeout: float = 600.0,
    ) -> None:
        """factory: a picklable callable that builds the engine inside the new process
        (e.g. functools.partial of a module-level function). Returns once it is built."""
        super().__init__(None, max_waiting, request_timeout)  # type: ignore[arg-type]
        ctx = mp.get_context("spawn")  # never fork a process that may hold CUDA state
        self._requests, child_requests = ctx.Pipe(duplex=False)[::-1]
        self._results, child_results = ctx.Pipe(duplex=False)
        self._send_lock = threading.Lock()
        self._proc = ctx.Process(
            target=_engine_main,
            args=(factory, child_requests, child_results, self.snapshot_interval),
            name="engine",
            daemon=True,
        )
        self._proc.start()
        child_requests.close()
        child_results.close()
        if not self._results.poll(startup_timeout):
            self._proc.kill()
            raise RuntimeError("engine process did not start in time")
        try:
            kind, info = self._results.recv()
        except EOFError:
            raise RuntimeError("engine process exited while starting; see its log") from None
        assert kind == "ready", kind
        self.info = info
        self._stats: dict = {"running": 0, "waiting": 0}
        self._pending = 0  # requests sent that the engine has not acknowledged yet

    # --- lifecycle --------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="engine-results", daemon=True)
            self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):  # already gone
            self._send(("stop",))
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    @property
    def alive(self) -> bool:
        return self._proc.is_alive() and super().alive

    @property
    def capacity_tokens(self) -> int:
        return self.info["capacity_tokens"]

    @property
    def num_waiting(self) -> int:
        return self._stats["waiting"] + self._pending

    @property
    def num_running(self) -> int:
        return self._stats["running"]

    # --- requests ---------------------------------------------------------------------------

    def _send(self, msg: tuple) -> None:
        with self._send_lock:
            self._requests.send(msg)

    def _submit(self, request_id: str, prompt_ids: list[int], params: SamplingParams) -> None:
        with self._lock:
            self._pending += 1
        self._send(("add", request_id, prompt_ids, params))

    def abort(self, request_id: str) -> None:
        with contextlib.suppress(OSError):  # the engine is gone, and the request with it
            self._send(("abort", request_id))

    # --- results ----------------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                msg = self._results.recv()
            except (EOFError, OSError):
                if not self._stop.is_set():
                    self.last_error = RuntimeError("engine process exited")
                    log.error("engine process exited; failing all in-flight requests")
                    with self._lock:
                        ids = list(self._streams)
                    for request_id in ids:
                        self._deliver(request_id, RuntimeError("engine process exited"))
                return
            kind = msg[0]
            if kind == "step":
                _, outputs, stats = msg
                with self._lock:
                    self._pending -= stats.pop("admitted")
                self._stats = stats
                if self.on_step is not None:
                    self.on_step(stats, len(outputs))
                self._deliver_step(outputs)
            elif kind == "ack":
                with self._lock:
                    self._pending -= msg[1]
            elif kind == "snapshot":
                self.snapshot = msg[1]
            elif kind == "error":  # add_request refused it (e.g. too long)
                with self._lock:
                    self._pending -= 1
                self._deliver(msg[1], msg[2])
            elif kind == "fail":  # a step raised; the engine dropped these requests
                self.last_error = RuntimeError(msg[2])
                for request_id in msg[1]:
                    self._deliver(request_id, RuntimeError(f"engine error: {msg[2]}"))
            elif kind == "stopped":
                return


def _engine_main(factory: Callable[[], LLMEngine], requests: Any, results: Any,
                 snapshot_interval: float) -> None:  # fmt: skip
    """The engine process: drain requests, step, send the step's outputs; repeat."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    engine = factory()
    results.send(("ready", {"capacity_tokens": engine.scheduler.capacity_tokens}))
    snapshot_at = 0.0
    try:
        while True:
            admitted = 0
            timeout = 0.0 if engine.has_unfinished() else 0.05
            while requests.poll(timeout):
                timeout = 0.0
                msg = requests.recv()
                if msg[0] == "add":
                    _, request_id, prompt_ids, params = msg
                    try:
                        engine.add_request(request_id, prompt_ids, params)
                        admitted += 1
                    except ValueError as exc:
                        results.send(("error", request_id, exc))
                elif msg[0] == "abort":
                    engine.abort(msg[1])
                elif msg[0] == "stop":
                    results.send(("stopped",))
                    return
            now = time.monotonic()
            if now - snapshot_at >= (snapshot_interval if engine.has_unfinished() else 1.0):
                snapshot_at = now
                try:
                    results.send(("snapshot", engine.snapshot()))
                except (TypeError, ValueError, AttributeError):
                    log.exception("engine snapshot failed")
            if not engine.has_unfinished():
                if admitted:  # added and aborted before any step ran
                    results.send(("ack", admitted))
                continue
            start = time.perf_counter()
            try:
                outputs = engine.step()
            except Exception as exc:  # e.g. a CUDA error: fail the requests, keep serving
                log.exception("engine step failed; failing all in-flight requests")
                sched = engine.scheduler
                ids = [s.request_id for s in list(sched.running) + list(sched.waiting)]
                for request_id in ids:
                    engine.abort(request_id)
                results.send(("fail", ids, str(exc)))
                continue
            stats = engine_stats(engine, time.perf_counter() - start)
            stats["admitted"] = admitted
            results.send(("step", outputs, stats))
    except (EOFError, BrokenPipeError, ConnectionError):
        return  # the HTTP process is gone: exit and free the GPU
