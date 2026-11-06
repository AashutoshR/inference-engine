"""KV cache block bookkeeping (Phase 2 FR-5, Phase 3 FR-13 prefix caching). No torch.

The KV cache is split into fixed-size blocks of ``block_size`` token slots. Each
sequence owns a block table: the list of block ids that hold its tokens, in order.
Token ``i`` of a sequence lives in slot ``table[i // block_size] * block_size +
i % block_size`` of the flat cache.

Prefix caching: a full block is identified by a hash of its tokens chained with the
hash of the block before it, so equal hashes mean equal prefixes. Full blocks are
registered under their hash; a new sequence whose prompt starts with the same tokens
reuses those blocks (reference-counted) instead of recomputing them. When the last
owner frees a hashed block it becomes *evictable*: still holding valid K/V and findable
by hash, but reclaimable, least recently used first, when free blocks run out.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Sequence


class OutOfBlocks(RuntimeError):
    """No free KV block. The scheduler avoids this by checking ``can_*`` first."""


class BlockAllocator:
    """Free list, evictable LRU and reference counts over ``num_blocks`` blocks."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._evictable: OrderedDict[int, None] = OrderedDict()  # refcount 0, still hashed
        self._refs = [0] * num_blocks
        self.block_hash: dict[int, int] = {}  # block -> content hash (cached blocks only)
        self.cached: dict[int, int] = {}  # content hash -> block

    @property
    def num_free(self) -> int:
        """Blocks available to allocate: truly free plus evictable."""
        return len(self._free) + len(self._evictable)

    def allocate(self) -> int:
        if self._free:
            block = self._free.popleft()
        elif self._evictable:
            block, _ = self._evictable.popitem(last=False)  # least recently used
            del self.cached[self.block_hash.pop(block)]
        else:
            raise OutOfBlocks("no free KV blocks")
        self._refs[block] = 1
        return block

    def acquire_cached(self, block: int) -> None:
        """Take a reference to a cached block, reviving it if it was evictable."""
        if self._refs[block] == 0:
            self._evictable.pop(block)
        self._refs[block] += 1

    def incref(self, block: int) -> None:
        if self._refs[block] < 1:
            raise ValueError(f"block {block} is free; cannot share it")
        self._refs[block] += 1

    def free(self, block: int) -> None:
        if self._refs[block] < 1:
            raise ValueError(f"double free of block {block}")
        self._refs[block] -= 1
        if self._refs[block] == 0:
            if block in self.block_hash:
                self._evictable[block] = None  # most recently used end
            else:
                self._free.append(block)

    def register(self, block: int, content_hash: int) -> None:
        """Make a full, owned block findable by its content hash."""
        if content_hash in self.cached or block in self.block_hash:
            return  # same content already cached elsewhere, or already registered
        self.cached[content_hash] = block
        self.block_hash[block] = content_hash

    def refcount(self, block: int) -> int:
        return self._refs[block]


def chain_hash(parent: int | None, tokens: Sequence[int]) -> int:
    return hash((parent, tuple(tokens)))


class BlockManager:
    """Block tables per sequence on top of a BlockAllocator."""

    def __init__(
        self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = True
    ) -> None:
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.allocator = BlockAllocator(num_blocks)
        self.tables: dict[int, list[int]] = {}
        # Per sequence: how many leading blocks are registered, and the last block's hash.
        self._registered: dict[int, tuple[int, int | None]] = {}

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    @property
    def num_cached_blocks(self) -> int:
        return len(self.allocator.cached)

    def blocks_needed(self, seq_id: int, num_tokens: int) -> int:
        """Extra blocks ``seq_id`` needs to hold ``num_tokens`` tokens in total."""
        have = len(self.tables.get(seq_id, ()))
        need = -(-num_tokens // self.block_size)  # ceil
        return max(need - have, 0)

    def can_grow(self, seq_id: int, num_tokens: int) -> bool:
        return self.blocks_needed(seq_id, num_tokens) <= self.allocator.num_free

    def grow(self, seq_id: int, num_tokens: int) -> list[int]:
        """Make sure ``seq_id`` has room for ``num_tokens`` tokens; return its block table."""
        needed = self.blocks_needed(seq_id, num_tokens)
        if needed > self.allocator.num_free:
            raise OutOfBlocks(f"sequence {seq_id} needs {needed} blocks")
        table = self.tables.setdefault(seq_id, [])
        table.extend(self.allocator.allocate() for _ in range(needed))
        return table

    def match_prefix(self, seq_id: int, token_ids: Sequence[int]) -> int:
        """Attach cached blocks matching the start of ``token_ids`` to a new sequence.

        Returns the number of tokens whose K/V are now in place. At least the last token
        is always left to compute, because its logits are needed.
        """
        if not self.enable_prefix_caching or seq_id in self.tables:
            return 0
        bs, table, parent = self.block_size, [], None
        for start in range(0, len(token_ids) - 1 - bs + 1, bs):
            h = chain_hash(parent, token_ids[start : start + bs])
            block = self.allocator.cached.get(h)
            if block is None:
                break
            self.allocator.acquire_cached(block)
            table.append(block)
            parent = h
        if table:
            self.tables[seq_id] = table
            self._registered[seq_id] = (len(table), parent)
        return len(table) * bs

    def register_full_blocks(
        self, seq_id: int, token_ids: Sequence[int], num_computed: int
    ) -> None:
        """Register every block of ``seq_id`` that is now full of computed tokens."""
        if not self.enable_prefix_caching:
            return
        done, parent = self._registered.get(seq_id, (0, None))
        table, bs = self.tables[seq_id], self.block_size
        full = min(num_computed // bs, len(table))
        for i in range(done, full):
            parent = chain_hash(parent, token_ids[i * bs : (i + 1) * bs])
            self.allocator.register(table[i], parent)
        self._registered[seq_id] = (max(done, full), parent)

    def free(self, seq_id: int) -> None:
        self._registered.pop(seq_id, None)
        for block in self.tables.pop(seq_id, ()):
            self.allocator.free(block)

    def slot(self, seq_id: int, position: int) -> int:
        """Flat cache slot of token ``position`` in sequence ``seq_id``."""
        table = self.tables[seq_id]
        return table[position // self.block_size] * self.block_size + position % self.block_size

    def check_no_leaks(self) -> None:
        """Debug check of the allocator's invariants against the live block tables."""
        a = self.allocator
        owned: dict[int, int] = {}
        for table in self.tables.values():
            for b in table:
                owned[b] = owned.get(b, 0) + 1
        free, evictable = set(a._free), set(a._evictable)
        assert len(free) == len(a._free), "duplicate block in free list"
        assert not free & evictable, "a block is both free and evictable"
        assert not (free | evictable) & set(owned), "a block is both free and owned"
        assert len(free) + len(evictable) + len(owned) == a.num_blocks, "leaked blocks"
        for b, n in owned.items():
            assert a.refcount(b) == n, f"block {b}: refcount {a.refcount(b)} != {n} owners"
        assert all(b in a.block_hash for b in evictable), "evictable block without a hash"
        assert all(a.cached[h] == b for b, h in a.block_hash.items()), "hash maps disagree"
