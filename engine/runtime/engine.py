"""The serving engine core (Phase 2): requests in, token deltas out, one step at a time.

    engine = LLMEngine(model, tokenizer, stop_token_ids)
    engine.add_request("r1", prompt_ids, SamplingParams(max_tokens=64))
    while engine.has_unfinished():
        for out in engine.step():
            print(out.request_id, out.text)

Not thread-safe: one thread owns the engine and calls step(). The API server
(Phase 2) runs it in a background loop and feeds requests through a queue.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from engine.runtime.block_manager import BlockManager
from engine.runtime.paged_cache import PagedBatch, PagedKVCache
from engine.runtime.params import SamplingParams
from engine.runtime.sampler import sample
from engine.runtime.scheduler import Scheduler, Sequence
from engine.runtime.stop import OutputProcessor


@dataclass
class RequestOutput:
    request_id: str
    token_id: int
    text: str  # detokenized delta, possibly ""
    finished: bool
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    # With SamplingParams.logprobs: log-probability of token_id under the model's own
    # distribution (before temperature or top-k/p), and the top alternatives.
    logprob: float | None = None
    top_logprobs: list[tuple[int, float]] | None = None


class LLMEngine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        stop_token_ids: tuple[int, ...],
        *,
        num_blocks: int | None = None,
        block_size: int = 16,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
        use_triton: bool | None = None,
        enable_cuda_graphs: bool | None = None,
        enable_mixed_graphs: bool | None = None,
        coalesce_prefill: bool | None = None,
        max_model_len: int | None = None,
        debug_checks: bool = False,
    ) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.stop_token_ids = tuple(stop_token_ids)
        weight = next(model.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        if num_blocks is None:
            num_blocks = self._blocks_that_fit(block_size, gpu_memory_utilization)
        # One extra block past the allocator's range: scratch space for CUDA-graph padding.
        self.cache = PagedKVCache(model.cfg, num_blocks + 1, block_size, self.device, self.dtype)
        self.max_model_len = min(
            max_model_len or model.cfg.max_position_embeddings, num_blocks * block_size
        )
        self.block_manager = BlockManager(num_blocks, block_size, enable_prefix_caching)
        self.scheduler = Scheduler(
            self.block_manager, max_num_seqs, max_num_batched_tokens, enable_chunked_prefill
        )
        from engine.kernels.paged_attention import available

        if use_triton is None:
            use_triton = available(self.device)
        elif use_triton and not available(self.device):
            raise ValueError("use_triton needs CUDA and the triton package")
        self.use_triton = use_triton
        if enable_cuda_graphs is None:
            enable_cuda_graphs = use_triton
        self.graphs = None
        if enable_cuda_graphs:
            if not use_triton:
                raise ValueError("CUDA graphs need the Triton decode kernel")
            from engine.runtime.cuda_graphs import DecodeGraphs

            self.graphs = DecodeGraphs(
                model, self.cache, num_blocks, max_num_seqs, -(-self.max_model_len // block_size)
            )
        # Mixed steps as piecewise graphs. Not with FP8: its activation scale is the max
        # over the whole step, so padding rows could change what real rows compute.
        if enable_mixed_graphs is None:
            enable_mixed_graphs = self.graphs is not None and _quantization_of(model) != "fp8"
        self.mixed_graphs = None
        if enable_mixed_graphs:
            if self.graphs is None:
                raise ValueError("mixed-step graphs need decode CUDA graphs")
            from engine.runtime.cuda_graphs import MixedGraphs

            self.mixed_graphs = MixedGraphs(
                model, self.cache, num_blocks, max_num_seqs, max_num_batched_tokens
            )
        # Holding new prompts for a few steps only pays when decode steps are graphs and
        # steps that read prompts are not: with mixed-step graphs those are cheap too.
        if coalesce_prefill is None:
            coalesce_prefill = self.graphs is not None and self.mixed_graphs is None
        self.scheduler.coalesce_prefill = coalesce_prefill
        self.prompt_tokens_total = 0
        self.cached_prompt_tokens_total = 0
        self.debug_checks = debug_checks
        self._seq_ids = itertools.count()
        self._requests: dict[str, Sequence] = {}
        # Live introspection for the dashboard: one record per step, plus running totals.
        self.step_log: deque[dict] = deque(maxlen=240)
        self.steps_total = 0
        self.generated_total = 0
        self.finished_total = 0
        self.started_at = time.time()

    def _blocks_that_fit(self, block_size: int, utilization: float) -> int:
        if self.device.type != "cuda":
            raise ValueError("pass num_blocks explicitly when not running on CUDA")
        free, total = torch.cuda.mem_get_info(self.device)
        # Leave (1 - utilization) of the card plus room for activations and logits.
        budget = free - (1 - utilization) * total - 512 * 2**20
        per_block = PagedKVCache.bytes_per_block(self.model.cfg, block_size, self.dtype)
        blocks = int(budget // per_block)
        if blocks < 1:
            raise RuntimeError("no GPU memory left for the KV cache")
        return blocks

    # --- requests -------------------------------------------------------------------------

    def add_request(self, request_id: str, prompt_ids: list[int], params: SamplingParams) -> None:
        if request_id in self._requests:
            raise ValueError(f"duplicate request id {request_id!r}")
        if not prompt_ids:
            raise ValueError("prompt must have at least one token")
        if len(prompt_ids) + params.max_tokens > self.max_model_len:
            raise ValueError(
                f"prompt + max_tokens = {len(prompt_ids) + params.max_tokens} tokens exceeds "
                f"max_model_len ({self.max_model_len})"
            )
        seq = Sequence(next(self._seq_ids), request_id, list(prompt_ids), params)
        seq.processor = OutputProcessor(self.tokenizer, prompt_ids, params, self.stop_token_ids)
        if params.seed is not None:
            seq.generator = torch.Generator(device=self.device).manual_seed(params.seed)
        self.scheduler.add(seq)  # raises if it can never fit
        self._requests[request_id] = seq

    def abort(self, request_id: str) -> bool:
        """Stop a request and free its KV blocks now. False if it was already done."""
        self._requests.pop(request_id, None)
        return self.scheduler.abort(request_id) is not None

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    @property
    def num_free_blocks(self) -> int:
        return self.block_manager.num_free_blocks

    # --- one step -------------------------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        started = time.perf_counter()
        preempted_before = self.scheduler.num_preemptions
        batch = self.scheduler.schedule()
        if batch is None:
            if self.scheduler.waiting and not self.scheduler.running:
                # Nothing runs and the queue head can't start: a scheduler bug, never
                # expected (add() rejects requests that can't fit). Fail loudly, don't spin.
                raise RuntimeError("scheduler made no progress with an idle engine")
            return []
        seqs = batch.seqs
        ids: list[int] = []
        positions: list[int] = []
        context_lens: list[int] = []
        for seq, n in zip(seqs, batch.num_tokens, strict=True):
            start = seq.num_computed
            if not seq.counted:  # first step: count prompt tokens and prefix-cache hits
                seq.counted = True
                self.prompt_tokens_total += len(seq.prompt_ids)
                self.cached_prompt_tokens_total += seq.num_cached_prompt_tokens
            ids.extend(seq.all_ids[start : start + n])
            positions.extend(range(start, start + n))
            context_lens.append(start + n)
        tables = [self.block_manager.tables[s.seq_id] for s in seqs]
        decode_only = len(ids) == len(seqs)
        decode_tokens = sum(1 for s, n in zip(seqs, batch.num_tokens, strict=True)
                            if n == 1 and s.output_ids)  # fmt: skip
        used_graph = used_mixed_graph = False
        if (
            self.graphs is not None
            and decode_only
            and self.graphs.can_run(len(seqs), max(len(t) for t in tables))
        ):
            slots = [
                self.block_manager.slot(s.seq_id, p) for s, p in zip(seqs, positions, strict=True)
            ]
            logits = self.graphs.run(ids, positions, slots, tables, context_lens)[:, 0]
            used_graph = True
        elif self.mixed_graphs is not None and self.mixed_graphs.can_run(len(ids), len(seqs)):
            paged = PagedBatch.build(
                self.cache, tables, context_lens, batch.num_tokens, self.device,
                use_kernel=True, force_mixed=True,
            )  # fmt: skip
            logits = self.mixed_graphs.run(ids, positions, paged)[:, 0]
            used_mixed_graph = True
        else:
            paged = PagedBatch.build(
                self.cache, tables, context_lens, batch.num_tokens, self.device,
                use_kernel=self.use_triton,
            )  # fmt: skip
            input_ids = torch.tensor([ids], device=self.device)
            pos = torch.tensor([positions], device=self.device)
            logits = self.model(input_ids, pos, paged)[:, 0]  # [B, vocab]
        # Greedy rows in one argmax and one device sync, not one per sequence.
        greedy = logits.argmax(-1).tolist()

        outputs = []
        want_logprobs: list[tuple[int, int, int, int]] = []  # (output idx, row, token, top k)
        for row, (seq, n) in enumerate(zip(seqs, batch.num_tokens, strict=True)):
            seq.num_computed += n
            self.block_manager.register_full_blocks(seq.seq_id, seq.all_ids, seq.num_computed)
            if seq.num_computed < seq.length:
                continue  # a prompt chunk: nothing to sample until the prompt is done
            params = seq.params
            if params.greedy and params.repetition_penalty == 1.0:
                token = greedy[row]
            else:
                prev = None
                if params.repetition_penalty != 1.0:
                    prev = torch.tensor([seq.all_ids], device=self.device)
                token = int(sample(logits[row : row + 1], params, seq.generator, prev)[0])
            seq.output_ids.append(token)
            text, finished, reason = seq.processor.process(token)
            if finished:
                self.scheduler.finish(seq, reason)
                self._requests.pop(seq.request_id, None)
            if params.logprobs:
                want_logprobs.append((len(outputs), row, token, params.top_logprobs))
            outputs.append(
                RequestOutput(
                    seq.request_id,
                    token,
                    text,
                    finished,
                    reason,
                    len(seq.prompt_ids),
                    len(seq.output_ids),
                )  # fmt: skip
            )
        if want_logprobs:
            self._attach_logprobs(logits, outputs, want_logprobs)
        if self.debug_checks:
            self.block_manager.check_no_leaks()
        self.steps_total += 1
        self.generated_total += len(outputs)
        self.finished_total += sum(o.finished for o in outputs)
        self.step_log.append({
            "step": self.steps_total,
            "ms": round((time.perf_counter() - started) * 1000, 2),
            "seqs": len(seqs),
            "prefill_tokens": len(ids) - decode_tokens,
            "decode_tokens": decode_tokens,
            "graph": used_graph,
            "mixed_graph": used_mixed_graph,
            "preempted": self.scheduler.num_preemptions - preempted_before,
        })  # fmt: skip
        return outputs

    @staticmethod
    def _attach_logprobs(logits, outputs, wanted) -> None:
        """One batched log-softmax for every row that asked for logprobs."""
        rows = torch.tensor([w[1] for w in wanted], device=logits.device)
        lp = logits[rows].float().log_softmax(-1)
        chosen = lp.gather(1, torch.tensor([[w[2]] for w in wanted], device=lp.device))
        k = max(w[3] for w in wanted)
        top = lp.topk(k, dim=-1) if k else None
        chosen = chosen[:, 0].tolist()
        top_v = top.values.tolist() if top is not None else None
        top_i = top.indices.tolist() if top is not None else None
        for j, (out_idx, _, _, kk) in enumerate(wanted):
            out = outputs[out_idx]
            out.logprob = chosen[j]
            if kk:
                out.top_logprobs = list(zip(top_i[j][:kk], top_v[j][:kk], strict=True))

    def snapshot(self) -> dict:
        """Everything the dashboard shows, as plain data. Call it from the engine's thread.

        ``blocks`` has one character per KV block: ``.`` free, ``c`` cached (evictable,
        still holds a reusable prefix), ``o`` owned by one sequence, ``s`` shared by
        several. ``owners`` gives the owning sequence id per block (-1 if none).
        """
        bm, sched = self.block_manager, self.scheduler
        alloc = bm.allocator
        codes = ["."] * alloc.num_blocks
        owners = [-1] * alloc.num_blocks
        for b in alloc._evictable:
            codes[b] = "c"
        for seq_id, table in bm.tables.items():
            for b in table:
                codes[b] = "s" if alloc.refcount(b) > 1 else "o"
                if owners[b] == -1:
                    owners[b] = seq_id

        def describe(seq: Sequence) -> dict:
            in_prompt = seq.num_computed < len(seq.prompt_ids) and not seq.output_ids
            return {
                "id": seq.seq_id,
                "request": seq.request_id[-8:],
                "phase": "prefill" if in_prompt else "decode",
                "prompt": len(seq.prompt_ids),
                "output": len(seq.output_ids),
                "max_tokens": seq.params.max_tokens,
                "computed": seq.num_computed,
                "cached": seq.num_cached_prompt_tokens,
                "blocks": len(bm.tables.get(seq.seq_id, ())),
                "preemptions": seq.num_preemptions,
            }

        cfg = self.model.cfg
        return {
            "time": time.time(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "config": {
                "layers": cfg.num_hidden_layers,
                "hidden": cfg.hidden_size,
                "heads": cfg.num_attention_heads,
                "kv_heads": cfg.num_key_value_heads,
                "dtype": str(self.dtype).removeprefix("torch."),
                "device": torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else str(self.device),
                "block_size": bm.block_size,
                "num_blocks": alloc.num_blocks,
                "capacity_tokens": sched.capacity_tokens,
                "max_num_seqs": sched.max_num_seqs,
                "max_num_batched_tokens": sched.max_num_batched_tokens,
                "triton": self.use_triton,
                "cuda_graphs": self.graphs is not None,
                "mixed_graphs": self.mixed_graphs is not None,
                "prefix_caching": bm.enable_prefix_caching,
                "chunked_prefill": sched.enable_chunked_prefill,
                "quantization": _quantization_of(self.model),
            },
            "blocks": "".join(codes),
            "owners": owners,
            "running": [describe(s) for s in sched.running],
            "waiting": [describe(s) for s in list(sched.waiting)[:32]],
            "num_waiting": len(sched.waiting),
            "totals": {
                "steps": self.steps_total,
                "generated_tokens": self.generated_total,
                "finished_requests": self.finished_total,
                "prompt_tokens": self.prompt_tokens_total,
                "cached_prompt_tokens": self.cached_prompt_tokens_total,
                "preemptions": sched.num_preemptions,
                "free_blocks": alloc.num_free,
                "cached_blocks": len(alloc._evictable),
            },
            "steps": list(self.step_log)[-120:],
        }

    def generate(
        self, prompts: dict[str, list[int]], params: SamplingParams
    ) -> dict[str, list[int]]:
        """Run requests to completion; returns the generated token ids per request."""
        for request_id, prompt in prompts.items():
            self.add_request(request_id, prompt, params)
        tokens: dict[str, list[int]] = {rid: [] for rid in prompts}
        while self.has_unfinished():
            for out in self.step():
                tokens[out.request_id].append(out.token_id)
        return tokens


def _quantization_of(model: Any) -> str:
    for module in model.modules():
        if hasattr(module, "packed"):
            return f"int4 (group {module.group_size})"
        if hasattr(module, "weight_fp8"):
            return "fp8"
    return "none"
