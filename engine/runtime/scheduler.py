"""Continuous batching (FR-7) with chunked prefill (FR-12), prefix-cache lookup (FR-13)
and recompute preemption (FR-11).

Every step has a token budget (``max_num_batched_tokens``). Running sequences go first,
oldest first: a decoding sequence needs one token, a sequence part-way through its
prompt gets the next chunk. Whatever budget is left admits waiting sequences, starting
with their first prompt chunk. So decodes never stall behind a long prompt, and a long
prompt is spread over several steps.

Bookkeeping invariant: ``seq.num_computed`` tokens of ``seq.all_ids`` have K/V in the
cache. A step feeds ``all_ids[num_computed : num_computed + n]``; the engine samples a
token only when that reaches the end of the sequence.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from engine.runtime.block_manager import BlockManager
from engine.runtime.params import SamplingParams


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    seq_id: int
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    num_computed: int = 0
    num_cached_prompt_tokens: int = 0  # prompt tokens served from the prefix cache
    status: SeqStatus = SeqStatus.WAITING
    finish_reason: str | None = None
    num_preemptions: int = 0
    counted: bool = False  # included in the engine's prompt-token stats
    processor: Any = None  # OutputProcessor, set by the engine
    generator: Any = None  # torch.Generator when params.seed is set

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_new_tokens(self) -> int:
        return self.length - self.num_computed


@dataclass
class ScheduledBatch:
    seqs: list[Sequence]
    num_tokens: list[int]  # tokens fed for each sequence this step

    @property
    def total_tokens(self) -> int:
        return sum(self.num_tokens)


COALESCE_PER_STEP = 8  # running requests per step a new prompt may be held


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 4096,
        enable_chunked_prefill: bool = True,
    ) -> None:
        if max_num_batched_tokens < max_num_seqs:
            raise ValueError("max_num_batched_tokens must be >= max_num_seqs (one per decode)")
        self.bm = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        # Prompt coalescing (on when the engine replays decode steps as CUDA graphs): a
        # step that reads prompt tokens cannot be a graph and costs several decode steps,
        # whatever its size. So while requests are decoding, new prompts may wait up to
        # one step per COALESCE_PER_STEP running requests and are then read together.
        self.coalesce_prefill = False
        self._held_steps = 0

    @property
    def capacity_tokens(self) -> int:
        return self.bm.allocator.num_blocks * self.bm.block_size

    def add(self, seq: Sequence) -> None:
        # A sequence must fit in the cache on its own, or it could never finish.
        longest = len(seq.prompt_ids) + seq.params.max_tokens
        if longest > self.capacity_tokens:
            raise ValueError(
                f"prompt + max_tokens = {longest} tokens exceeds the KV cache "
                f"({self.capacity_tokens} tokens)"
            )
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, request_id: str) -> Sequence | None:
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.request_id == request_id:
                    queue.remove(seq)
                    self.finish(seq, "abort")
                    return seq
        return None

    def finish(self, seq: Sequence, reason: str | None) -> None:
        seq.status, seq.finish_reason = SeqStatus.FINISHED, reason
        if seq in self.running:
            self.running.remove(seq)
        self.bm.free(seq.seq_id)

    # --- one step ---------------------------------------------------------------------------

    def schedule(self) -> ScheduledBatch | None:
        budget = self.max_num_batched_tokens
        seqs: list[Sequence] = []
        counts: list[int] = []

        # 1. Running sequences, oldest first. Preempt the newest ones if blocks run out.
        queue = list(self.running)
        kept: list[Sequence] = []
        while queue and budget > 0:
            seq = queue.pop(0)
            n = self._chunk(seq, budget)
            if n == 0:
                kept.append(seq)
                continue
            target = seq.num_computed + n
            while not self.bm.can_grow(seq.seq_id, target):
                victim = queue.pop() if queue else seq
                self._preempt(victim)
                if victim is seq:
                    break
            else:
                self.bm.grow(seq.seq_id, target)
                kept.append(seq)
                seqs.append(seq)
                counts.append(n)
                budget -= n
        kept.extend(queue)  # out of budget this step; they run next step
        self.running = kept

        # 2. Admit waiting sequences with what is left, unless coalescing holds them.
        if self.waiting and self._hold_prompts(counts):
            self._held_steps += 1
            return ScheduledBatch(seqs, counts) if seqs else None
        admitted = len(seqs)
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if seq.num_computed == 0:
                cached = self.bm.match_prefix(seq.seq_id, seq.all_ids)
                seq.num_computed = cached
                if not seq.output_ids:
                    seq.num_cached_prompt_tokens = cached
            n = self._chunk(seq, budget)
            if n == 0 or not self.bm.can_grow(seq.seq_id, seq.num_computed + n):
                # Let go of any prefix blocks it matched: a waiting sequence must hold no
                # blocks, or it can starve whoever is ahead of it after a preemption. The
                # blocks stay in the prefix cache (evictable), so matching again is cheap.
                self.bm.free(seq.seq_id)
                seq.num_computed = seq.num_cached_prompt_tokens = 0
                break
            self.bm.grow(seq.seq_id, seq.num_computed + n)
            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            seqs.append(seq)
            counts.append(n)
            budget -= n

        if len(seqs) > admitted:
            self._held_steps = 0
        return ScheduledBatch(seqs, counts) if seqs else None

    def _hold_prompts(self, counts: list[int]) -> bool:
        """Keep waiting prompts out of this step so they can be read together later?"""
        if not self.coalesce_prefill or not counts or any(c > 1 for c in counts):
            return False  # nothing decoding, or this step is not a graph step anyway
        limit = -(-len(self.running) // COALESCE_PER_STEP)  # ceil: 64 running -> 8 steps
        return self._held_steps + 1 < limit

    def _chunk(self, seq: Sequence, budget: int) -> int:
        """Tokens to feed ``seq`` this step within ``budget`` (0 = wait for the next step)."""
        n = seq.num_new_tokens
        if n <= budget:
            return n
        if self.enable_chunked_prefill:
            return budget
        # Without chunking a prompt runs whole; one that exceeds the whole budget runs alone.
        return n if budget == self.max_num_batched_tokens else 0

    def _preempt(self, seq: Sequence) -> None:
        self.bm.free(seq.seq_id)  # full blocks stay in the prefix cache as evictable
        seq.num_computed = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)  # newest victims are pushed first, so oldest end up first
