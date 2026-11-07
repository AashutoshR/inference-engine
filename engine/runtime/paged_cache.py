"""Paged KV cache and per-step batch metadata (Phase 2, FR-5 and FR-6).

The cache is one flat array of slots per layer: k, v are [layers, num_slots, nkv, hd]
with num_slots = num_blocks * block_size. A step's tokens from every sequence are
flattened into one row, [1, N]; ``slot_mapping`` says where each new token's K/V goes.

Attention here is the PyTorch reference: gather each sequence's context through its
block table into a padded [B, max_ctx] view, then one SDPA call with a mask. It is
correct for prefill, decode, and mixed or chunked steps alike. A Triton paged-attention
kernel replaces the gather later and is tested against this.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.config import ModelConfig


class PagedKVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int = 16,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_blocks, self.block_size = num_blocks, block_size
        shape = (cfg.num_hidden_layers, num_blocks * block_size, cfg.num_key_value_heads,
                 cfg.head_dim)  # fmt: skip
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    @staticmethod
    def bytes_per_block(cfg: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
        per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim
        return per_token * block_size * torch.tensor([], dtype=dtype).element_size()

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
        """k, v: [N, nkv, hd] for the step's new tokens; slots: [N] flat slot ids."""
        self.k[layer].index_copy_(0, slots, k.to(self.k.dtype))
        self.v[layer].index_copy_(0, slots, v.to(self.v.dtype))

    def gather(self, layer: int, kv_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_slots: [B, max_ctx] -> k, v: [B, max_ctx, nkv, hd] (padding reads slot 0)."""
        return self.k[layer][kv_slots], self.v[layer][kv_slots]


@dataclass
class PagedBatch:
    """Everything attention needs for one step, built once and shared by all layers."""

    cache: PagedKVCache
    slot_mapping: torch.Tensor  # [N] where each new token's K/V is written
    kv_slots: torch.Tensor  # [B, max_ctx] cache slot of every context position
    q_index: torch.Tensor  # [B, max_q] flat token index of each padded query (pads -> 0)
    q_valid: torch.Tensor  # [B, max_q] bool
    attn_mask: torch.Tensor | None  # [B, 1, max_q, max_ctx] bool, True = may attend
    last_index: torch.Tensor  # [B] flat index of each sequence's last token this step
    # Decode-only steps with the Triton kernel: block tables and lengths, nothing gathered.
    use_kernel: bool = False
    block_tables: torch.Tensor | None = None  # [B, max_blocks] int32
    context_lens: torch.Tensor | None = None  # [B] int32
    # Mixed steps with the Triton kernel (several sequences, some reading prompt tokens):
    # decode rows go through the kernel, each prompt chunk gets its own unpadded SDPA.
    # Padding everything to the longest chunk would cost B x chunk x context per layer.
    mixed: bool = False
    decode_rows: torch.Tensor | None = None  # [Bd] flat token index of each decode row
    chunks: list[tuple[int, int, torch.Tensor, torch.Tensor | None]] | None = None
    # (first flat token, length, context slots [ctx], mask [len, ctx] or None = causal)
    # In BF16/FP16 all prompt chunks instead go through one FlashAttention varlen call:
    # (query rows [Tq], context slots of every chunk back to back [Tk], cu_q, cu_k,
    # longest chunk, longest context).
    varlen: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None = None

    @classmethod
    def build(
        cls,
        cache: PagedKVCache,
        block_tables: list[list[int]],
        context_lens: list[int],
        query_lens: list[int],
        device: str | torch.device,
        use_kernel: bool = False,
    ) -> PagedBatch:
        """block_tables, context_lens (tokens in cache after this step, new ones included)
        and query_lens (new tokens this step) per sequence, in flattened order."""
        bs = cache.block_size
        b, max_q, max_ctx = len(query_lens), max(query_lens), max(context_lens)
        if use_kernel and max_q == 1:
            return cls._decode(cache, block_tables, context_lens, device)
        if use_kernel and b > 1:
            return cls._mixed(cache, block_tables, context_lens, query_lens, device)

        slot_mapping, kv_slots = [], []
        for table, ctx, q in zip(block_tables, context_lens, query_lens, strict=True):
            slots = [table[p // bs] * bs + p % bs for p in range(ctx)]
            slot_mapping.extend(slots[ctx - q :])
            kv_slots.append(slots + [0] * (max_ctx - ctx))

        starts = torch.tensor([0, *query_lens], device=device).cumsum(0)[:-1]
        qlens = torch.tensor(query_lens, device=device)
        ctxs = torch.tensor(context_lens, device=device)
        i = torch.arange(max_q, device=device)
        q_valid = i[None, :] < qlens[:, None]
        q_index = torch.where(q_valid, starts[:, None] + i[None, :], 0)

        # Query i of sequence s sits at absolute position ctx - q + i; it sees keys up to
        # there. Padded query rows see key 0 only, so no row is fully masked (no NaN).
        q_pos = (ctxs - qlens)[:, None] + i[None, :]  # [B, max_q]
        k_pos = torch.arange(max_ctx, device=device)
        mask = (k_pos[None, None, :] <= q_pos[:, :, None]) & (k_pos < ctxs[:, None])[:, None, :]
        mask = mask | (~q_valid[:, :, None] & (k_pos == 0))
        return cls(
            cache=cache,
            slot_mapping=torch.tensor(slot_mapping, device=device, dtype=torch.long),
            kv_slots=torch.tensor(kv_slots, device=device, dtype=torch.long).view(b, max_ctx),
            q_index=q_index,
            q_valid=q_valid,
            attn_mask=mask[:, None],
            last_index=starts + qlens - 1,
        )

    @classmethod
    def _decode(
        cls,
        cache: PagedKVCache,
        block_tables: list[list[int]],
        context_lens: list[int],
        device: str | torch.device,
    ) -> PagedBatch:
        bs, b = cache.block_size, len(context_lens)
        width = max(len(t) for t in block_tables)
        tables = torch.tensor([t + [0] * (width - len(t)) for t in block_tables], dtype=torch.int32)
        slots = [
            t[(n - 1) // bs] * bs + (n - 1) % bs
            for t, n in zip(block_tables, context_lens, strict=True)
        ]
        flat = torch.arange(b, device=device)
        return cls(
            cache=cache,
            slot_mapping=torch.tensor(slots, device=device, dtype=torch.long),
            kv_slots=flat,  # unused on the kernel path
            q_index=flat[:, None],
            q_valid=torch.ones(b, 1, dtype=torch.bool, device=device),
            attn_mask=None,
            last_index=flat,
            use_kernel=True,
            block_tables=tables.to(device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        )

    @classmethod
    def _mixed(
        cls,
        cache: PagedKVCache,
        block_tables: list[list[int]],
        context_lens: list[int],
        query_lens: list[int],
        device: str | torch.device,
    ) -> PagedBatch:
        bs = cache.block_size
        use_varlen = varlen_available(cache.k)
        slot_mapping, chunks, dec_rows, dec_tables, dec_ctx = [], [], [], [], []
        v_rows, v_slots, v_q, v_k = [], [], [], []
        start = 0
        for table, ctx, q in zip(block_tables, context_lens, query_lens, strict=True):
            slots = [table[p // bs] * bs + p % bs for p in range(ctx)]
            slot_mapping.extend(slots[ctx - q :])
            if q == 1:
                dec_rows.append(start)
                dec_tables.append(table)
                dec_ctx.append(ctx)
            elif use_varlen:  # queries sit at the end of their context: bottom-right causal
                v_rows.extend(range(start, start + q))
                v_slots.extend(slots)
                v_q.append(q)
                v_k.append(ctx)
            else:
                mask = None  # whole prompt in one chunk: plain causal attention
                if ctx > q:  # later chunk, or prefix hit: queries sit at ctx - q + i
                    i = torch.arange(q, device=device)[:, None]
                    mask = torch.arange(ctx, device=device)[None, :] <= (ctx - q) + i
                kv = torch.tensor(slots, device=device, dtype=torch.long)
                chunks.append((start, q, kv, mask))
            start += q
        varlen = None
        if v_q:
            cu = torch.tensor([0, *v_q, 0, *v_k], dtype=torch.int32).view(2, -1)
            cu[:, 1:] = cu[:, 1:].cumsum(1)
            cu = cu.to(device)
            varlen = (torch.tensor(v_rows, dtype=torch.long).to(device),
                      torch.tensor(v_slots, dtype=torch.long).to(device),
                      cu[0], cu[1], max(v_q), max(v_k))  # fmt: skip
        tables = None
        if dec_tables:
            width = max(len(t) for t in dec_tables)
            tables = torch.tensor(
                [t + [0] * (width - len(t)) for t in dec_tables], dtype=torch.int32
            ).to(device)
        qlens = torch.tensor(query_lens, device=device)
        starts = torch.tensor([0, *query_lens], device=device).cumsum(0)[:-1]
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return cls(
            cache=cache,
            slot_mapping=torch.tensor(slot_mapping, device=device, dtype=torch.long),
            kv_slots=empty,  # unused on the mixed path
            q_index=empty,
            q_valid=torch.zeros(0, dtype=torch.bool, device=device),
            attn_mask=None,
            last_index=starts + qlens - 1,
            use_kernel=False,
            block_tables=tables,
            context_lens=(
                torch.tensor(dec_ctx, dtype=torch.int32, device=device) if dec_ctx else None
            ),  # fmt: skip
            mixed=True,
            decode_rows=torch.tensor(dec_rows, device=device, dtype=torch.long),
            chunks=chunks,
            varlen=varlen,
        )


def varlen_available(t: torch.Tensor) -> bool:
    """FlashAttention varlen: CUDA, half precision, and a torch that ships it."""
    if t.device.type != "cuda" or t.dtype not in (torch.bfloat16, torch.float16):
        return False
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
    except ImportError:
        return False
    return True
