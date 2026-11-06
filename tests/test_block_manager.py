"""Phase 2: block allocator properties (no double free, no leak, no unowned sharing)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.runtime.block_manager import BlockAllocator, BlockManager, OutOfBlocks


def test_allocate_until_empty_then_raise():
    alloc = BlockAllocator(3)
    blocks = {alloc.allocate() for _ in range(3)}
    assert blocks == {0, 1, 2}
    with pytest.raises(OutOfBlocks):
        alloc.allocate()


def test_double_free_is_caught():
    alloc = BlockAllocator(2)
    b = alloc.allocate()
    alloc.free(b)
    with pytest.raises(ValueError, match="double free"):
        alloc.free(b)


def test_shared_block_returns_only_after_last_owner():
    alloc = BlockAllocator(1)
    b = alloc.allocate()
    alloc.incref(b)
    alloc.free(b)
    assert alloc.num_free == 0
    alloc.free(b)
    assert alloc.num_free == 1


def test_slots_follow_block_table():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.grow(seq_id=7, num_tokens=6)  # two blocks
    table = bm.tables[7]
    assert len(table) == 2
    assert bm.slot(7, 0) == table[0] * 4
    assert bm.slot(7, 5) == table[1] * 4 + 1
    assert bm.blocks_needed(7, 8) == 0
    assert bm.blocks_needed(7, 9) == 1


@settings(max_examples=200, deadline=None)
@given(
    ops=st.lists(
        st.tuples(st.sampled_from(["grow", "free"]), st.integers(0, 5), st.integers(1, 40)),
        max_size=80,
    )
)
def test_random_grow_and_free_never_leak(ops):
    bm = BlockManager(num_blocks=12, block_size=4)
    lengths: dict[int, int] = {}
    for op, seq, n in ops:
        if op == "grow":
            target = lengths.get(seq, 0) + n
            if bm.can_grow(seq, target):
                bm.grow(seq, target)
                lengths[seq] = target
            else:
                with pytest.raises(OutOfBlocks):
                    bm.grow(seq, target)
        else:
            bm.free(seq)
            lengths.pop(seq, None)
        bm.check_no_leaks()
        for s, length in lengths.items():
            assert len(bm.tables[s]) * 4 >= length
    for s in list(lengths):
        bm.free(s)
    assert bm.num_free_blocks == 12
