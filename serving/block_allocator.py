"""
Physical-block allocator for the paged KV cache.

The allocator separates two ideas that a contiguous KV cache conflates:

* A request reserves enough *capacity* to finish without a page fault that
  deadlocks the scheduler.
* It receives physical blocks only when a token actually reaches that block.

For example, a request allowed to generate 128 tokens may reserve 21
16-token blocks, but a request that stops after 20 tokens physically owns only
two blocks. Ref-counting is unused by the first paged-attention path, but makes
the allocator ready for prefix sharing: two block tables can safely point at
the same immutable prefix block.
"""

from __future__ import annotations

import heapq


class BlockAllocator:
    """Allocate, reserve, share, and release fixed-size physical block IDs."""

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self._free_blocks = list(range(num_blocks))
        heapq.heapify(self._free_blocks)
        self._ref_counts = [0] * num_blocks
        # Remaining not-yet-allocated blocks promised to each owner.
        self._reservations: dict[int, int] = {}
        self._owned_blocks: dict[int, set[int]] = {}

    @property
    def free_blocks(self) -> int:
        """Physical blocks that are not currently referenced by any request."""
        return len(self._free_blocks)

    @property
    def reserved_blocks(self) -> int:
        """Unallocated physical blocks promised to live owners."""
        return sum(self._reservations.values())

    @property
    def available_reservation_blocks(self) -> int:
        """Capacity that can still be promised to a newly admitted request."""
        return self.free_blocks - self.reserved_blocks

    def can_reserve(self, request_id: int, block_count: int) -> bool:
        """Whether a new request can reserve `block_count` pages."""
        if request_id in self._reservations:
            raise ValueError(f"request {request_id} already has a reservation")
        return 0 <= block_count <= self.available_reservation_blocks

    def reserve(self, request_id: int, block_count: int) -> None:
        """
        Promise an owner enough additional physical pages to finish.

        This does not remove a physical block from the free list. `allocate()`
        does that lazily and consumes one promised block. Shared blocks added
        through `incref()` need no reservation because they already exist.
        """
        if block_count < 0:
            raise ValueError(f"block_count must be non-negative, got {block_count}")
        if not self.can_reserve(request_id, block_count):
            raise MemoryError(
                f"cannot reserve {block_count} KV blocks for request {request_id}: "
                f"{self.available_reservation_blocks} remain"
            )
        self._reservations[request_id] = block_count
        self._owned_blocks[request_id] = set()

    def allocate(self, request_id: int) -> int:
        """Give a reserved request its next physical block."""
        if request_id not in self._reservations:
            raise KeyError(f"request {request_id} has no block reservation")
        if self._reservations[request_id] <= 0:
            raise MemoryError(
                f"request {request_id} exhausted its reserved KV blocks"
            )
        if not self._free_blocks:
            raise RuntimeError("reserved KV capacity invariant violated: no free physical blocks")

        block_id = heapq.heappop(self._free_blocks)
        self._ref_counts[block_id] = 1
        self._owned_blocks[request_id].add(block_id)
        self._reservations[request_id] -= 1
        return block_id

    def incref(self, request_id: int, block_id: int) -> None:
        """
        Add an existing block to another request's block table.

        This is the primitive RadixAttention will use when a request reuses a
        cached prefix. The caller owns the logical table entry; this allocator
        owns only lifetime accounting.
        """
        self._validate_block_id(block_id)
        if request_id not in self._reservations:
            raise KeyError(f"request {request_id} has no block reservation")
        owned = self._owned_blocks[request_id]
        if block_id in owned:
            raise ValueError(f"request {request_id} already references block {block_id}")
        if self._ref_counts[block_id] == 0:
            raise ValueError(f"cannot share unallocated block {block_id}")
        self._ref_counts[block_id] += 1
        owned.add(block_id)

    def release_request(self, request_id: int) -> None:
        """Drop all references and the capacity reservation for one request."""
        owned = self._owned_blocks.pop(request_id, None)
        if owned is None:
            return
        for block_id in owned:
            self._decref(block_id)
        self._reservations.pop(request_id)

    def ref_count(self, block_id: int) -> int:
        """Return the current number of block-table references."""
        self._validate_block_id(block_id)
        return self._ref_counts[block_id]

    def _decref(self, block_id: int) -> None:
        self._ref_counts[block_id] -= 1
        if self._ref_counts[block_id] < 0:
            raise RuntimeError(f"block {block_id} refcount underflow")
        if self._ref_counts[block_id] == 0:
            heapq.heappush(self._free_blocks, block_id)

    def _validate_block_id(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(f"block_id {block_id} outside [0, {self.num_blocks})")

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(num_blocks={self.num_blocks}, free={self.free_blocks}, "
            f"reserved={self.reserved_blocks})"
        )
