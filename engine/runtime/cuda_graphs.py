"""CUDA graphs for decode steps (Phase 3, FR-14).

A decode step of a small model is bound by kernel launches and Python, not by the GPU:
16 layers x ~10 kernels each. Capturing the whole forward pass once per batch-size
bucket and replaying it turns hundreds of launches into one.

Graphs need fixed shapes and fixed addresses, so every input lives in a static buffer
sized for the largest bucket. A step of B sequences copies its data into the buffers,
pads up to the next bucket, and replays that bucket's graph. Padding rows read and
write one scratch block that the allocator never hands out. The Triton decode kernel
reads each sequence's context length from GPU memory, so one graph serves any length.
"""

from __future__ import annotations

import torch

from engine.runtime.paged_cache import PagedBatch, PagedKVCache

BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)


class DecodeGraphs:
    def __init__(
        self,
        model: torch.nn.Module,
        cache: PagedKVCache,
        scratch_block: int,
        max_batch: int,
        max_blocks_per_seq: int,
    ) -> None:
        self.model, self.cache = model, cache
        dev = cache.k.device
        self.sizes = [s for s in BUCKETS if s < max_batch] + [max_batch]
        b, w = self.sizes[-1], max_blocks_per_seq
        self.max_width = w
        self.scratch_block = scratch_block
        self.scratch_slot = scratch_block * cache.block_size
        self.ids = torch.zeros(1, b, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, b, dtype=torch.long, device=dev)
        self.slots = torch.full((b,), self.scratch_slot, dtype=torch.long, device=dev)
        self.tables = torch.full((b, w), scratch_block, dtype=torch.int32, device=dev)
        self.ctx = torch.ones(b, dtype=torch.int32, device=dev)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self._batches: dict[int, PagedBatch] = {}
        self._capture()

    def _batch(self, n: int) -> PagedBatch:
        flat = torch.arange(n, device=self.ids.device)
        return PagedBatch(
            cache=self.cache,
            slot_mapping=self.slots[:n],
            kv_slots=flat,
            q_index=flat[:, None],
            q_valid=torch.ones(n, 1, dtype=torch.bool, device=flat.device),
            attn_mask=None,
            last_index=flat,
            use_kernel=True,
            block_tables=self.tables[:n],
            context_lens=self.ctx[:n],
        )

    def _capture(self) -> None:
        pool = None
        for n in reversed(self.sizes):  # largest first, so smaller graphs reuse its pool
            batch = self._batch(n)
            ids, pos = self.ids[:, :n], self.pos[:, :n]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):  # warm up: Triton compiles, cuBLAS picks kernels
                    self.model(ids, pos, batch)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self.model(ids, pos, batch)
            pool = graph.pool()
            # Keep the batch alive: the graph reads its tensors (e.g. last_index) by address,
            # and freed memory would be reused for something else.
            self.graphs[n], self.outputs[n], self._batches[n] = graph, out, batch
        torch.cuda.synchronize()

    def can_run(self, batch_size: int, width: int) -> bool:
        return batch_size <= self.sizes[-1] and width <= self.max_width

    def run(
        self,
        ids: list[int],
        positions: list[int],
        slots: list[int],
        tables: list[list[int]],
        context_lens: list[int],
    ) -> torch.Tensor:
        """One decode step for len(ids) sequences -> logits [B, 1, vocab]."""
        b = len(ids)
        n = next(s for s in self.sizes if s >= b)
        pad = n - b
        width = max(len(t) for t in tables)
        rows = [t + [self.scratch_block] * (width - len(t)) for t in tables]
        rows += [[self.scratch_block] * width] * pad
        self.ids[0, :n].copy_(torch.tensor(ids + [0] * pad), non_blocking=True)
        self.pos[0, :n].copy_(torch.tensor(positions + [0] * pad), non_blocking=True)
        self.slots[:n].copy_(torch.tensor(slots + [self.scratch_slot] * pad), non_blocking=True)
        self.ctx[:n].copy_(torch.tensor(context_lens + [1] * pad, dtype=torch.int32))
        self.tables[:n, :width].copy_(torch.tensor(rows, dtype=torch.int32))
        self.graphs[n].replay()
        return self.outputs[n][:b]
