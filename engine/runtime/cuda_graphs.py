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


# Mixed steps (some sequences reading prompt tokens) by total tokens in the step. Past
# about 1,000 tokens the GPU work outlasts the launch overhead and graphs stop paying.
TOKEN_BUCKETS = (16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048)


class MixedGraphs:
    """Mixed steps as piecewise CUDA graphs.

    A mixed step's attention changes shape every step (which rows decode, how long each
    prompt chunk is), but everything else in the network works on the step's N tokens
    row by row. So each layer is split at attention: one graph runs the previous layer's
    o_proj, residual and MLP plus this layer's norm, q/k/v projections, RoPE and KV-cache
    write; attention then runs eagerly between graphs, reading q and writing its output
    at fixed addresses. That is num_layers + 1 replays and a handful of attention
    launches per layer instead of ~770 launches, which on this machine cost ~44 us each.

    N is padded up to a token bucket. Padding rows embed token 0, write their K/V into
    the scratch block, and are never attended to or sampled, and every op outside
    attention treats rows independently, so real rows compute exactly what they would
    in an eager step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cache: PagedKVCache,
        scratch_block: int,
        max_seqs: int,
        max_tokens: int,
    ) -> None:
        self.model, self.cache = model, cache
        cfg = model.cfg
        dev = cache.k.device
        dtype = model.model.embed_tokens.weight.dtype
        cap = min(max_tokens, TOKEN_BUCKETS[-1])
        self.sizes = [s for s in TOKEN_BUCKETS if s < cap] + [cap]
        t, nh, hd = self.sizes[-1], cfg.num_attention_heads, cfg.head_dim
        self.scratch_slot = scratch_block * cache.block_size
        self.inputs = torch.zeros(2, t, dtype=torch.long, device=dev)  # token ids, positions
        self.slots = torch.full((t,), self.scratch_slot, dtype=torch.long, device=dev)
        self.last = torch.zeros(max_seqs, dtype=torch.long, device=dev)
        self.x = torch.zeros(1, t, cfg.hidden_size, dtype=dtype, device=dev)  # residual
        self.cos = torch.zeros(1, t, hd, device=dev)
        self.sin = torch.zeros(1, t, hd, device=dev)
        self.q = torch.zeros(t, nh, hd, dtype=dtype, device=dev)
        self.attn = torch.zeros(t, nh, hd, dtype=dtype, device=dev)
        self.logits = torch.zeros(max_seqs, 1, cfg.vocab_size, device=dev)
        self.graphs: dict[int, list[torch.cuda.CUDAGraph]] = {}
        self._capture()

    def _piece(self, i: int, n: int) -> None:
        """Graph i of a step with n (padded) tokens: the network from just after layer
        i - 1's attention to just before layer i's (i = 0: from the embedding; i = L:
        through the final norm and lm_head)."""
        layers, n_layers = self.model.model.layers, len(self.model.model.layers)
        x, cos, sin = self.x[:, :n], self.cos[:, :n], self.sin[:, :n]
        if i == 0:
            c, s = self.model.rotary(self.inputs[1:, :n])
            cos.copy_(c)
            sin.copy_(s)
            new = self.model.model.embed_tokens(self.inputs[:1, :n])
        else:
            prev = layers[i - 1]
            h = x + prev.self_attn.o_proj(self.attn[:n].flatten(1))[None]
            new = h + prev.mlp(prev.post_attention_layernorm(h))
        if i == n_layers:
            last = new[0, self.last][:, None]
            self.logits.copy_(self.model.lm_head(self.model.model.norm(last)).float())
            return
        x.copy_(new)
        layer = layers[i]
        q, k, v = layer.self_attn.project(layer.input_layernorm(new), cos, sin)
        self.cache.write(i, k[0].transpose(0, 1), v[0].transpose(0, 1), self.slots[:n])
        self.q[:n].copy_(q[0].transpose(0, 1))

    @torch.inference_mode()
    def _capture(self) -> None:
        pool = None
        pieces = len(self.model.model.layers) + 1
        for n in reversed(self.sizes):  # largest first, so smaller graphs reuse its pool
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):  # warm up: cuBLAS picks its kernels for this shape
                    for i in range(pieces):
                        self._piece(i, n)
            torch.cuda.current_stream().wait_stream(side)
            graphs = []
            for i in range(pieces):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._piece(i, n)
                pool = graph.pool()
                graphs.append(graph)
            self.graphs[n] = graphs
        torch.cuda.synchronize()

    def can_run(self, num_tokens: int, num_seqs: int) -> bool:
        return num_tokens <= self.sizes[-1] and num_seqs <= self.last.numel()

    @torch.inference_mode()
    def run(self, ids: list[int], positions: list[int], batch: PagedBatch) -> torch.Tensor:
        """One mixed step: ids, positions of the step's tokens (flattened), and the step's
        mixed PagedBatch -> logits [B, 1, vocab] for each sequence's last token."""
        t, b = len(ids), batch.last_index.numel()
        n = next(s for s in self.sizes if s >= t)
        pad = [0] * (n - t)
        self.inputs[:, :n].copy_(torch.tensor([ids + pad, positions + pad]), non_blocking=True)
        self.slots[:t].copy_(batch.slot_mapping)
        self.slots[t:n].fill_(self.scratch_slot)
        self.last.zero_()  # stale indices could point past this bucket
        self.last[:b].copy_(batch.last_index)
        graphs, q, out = self.graphs[n], self.q[:t].transpose(0, 1), self.attn[:t]
        for layer, graph in zip(self.model.model.layers, graphs, strict=False):
            graph.replay()
            layer.self_attn._mixed(q, batch, out=out)
        graphs[-1].replay()
        return self.logits[:b]
