"""Block-aligned radix prefix cache for paged KV memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from serving.paged_kv_cache import PagedKVCache


@dataclass
class PrefixMatch:
    """Longest reusable complete-page prefix for one prompt."""

    token_count: int
    block_ids: list[int]


@dataclass
class RadixNode:
    """One token page and its pinned physical KV block."""

    token_block: tuple[int, ...] | None = None
    block_id: int | None = None
    owner_id: int | None = None
    parent: "RadixNode | None" = None
    children: dict[tuple[int, ...], "RadixNode"] = field(default_factory=dict)
    last_used: int = 0


class RadixCache:
    """
    Radix tree keyed by complete token pages.

    Each node pins one physical KV block across all layers. Requests reuse those
    blocks by incrementing the allocator reference count, so matching performs
    no K/V copies. Leaf-LRU eviction drops only the cache's reference; blocks
    remain alive while any active request still references them.
    """

    def __init__(self, kv_cache: PagedKVCache, max_blocks: int):
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be positive, got {max_blocks}")
        self.kv = kv_cache
        self.block_size = kv_cache.block_size
        self.max_blocks = max_blocks
        self.root = RadixNode()
        self.cached_blocks = 0
        self.hits = 0
        self.misses = 0
        self.matched_tokens = 0
        self.evictions = 0
        self._clock = 0
        self._next_owner_id = -1

    def _touch(self, node: RadixNode) -> None:
        self._clock += 1
        node.last_used = self._clock

    def _new_owner_id(self) -> int:
        owner_id = self._next_owner_id
        self._next_owner_id -= 1
        return owner_id

    def match(self, tokens: list[int]) -> PrefixMatch:
        """
        Return the longest cached complete-page prefix.

        At least one prompt token remains uncached because prefill needs a query
        token to produce the first next-token logits.
        """
        reusable_tokens = ((len(tokens) - 1) // self.block_size) * self.block_size
        node = self.root
        block_ids: list[int] = []
        for start in range(0, reusable_tokens, self.block_size):
            token_block = tuple(tokens[start:start + self.block_size])
            child = node.children.get(token_block)
            if child is None:
                break
            if child.block_id is None:
                raise RuntimeError("radix-cache node has no physical block")
            self._touch(child)
            block_ids.append(child.block_id)
            node = child

        token_count = len(block_ids) * self.block_size
        if token_count:
            self.hits += 1
            self.matched_tokens += token_count
        else:
            self.misses += 1
        return PrefixMatch(token_count=token_count, block_ids=block_ids)

    def insert(
        self,
        tokens: list[int],
        block_ids: list[int],
    ) -> None:
        """Insert every complete prompt page and pin newly-created nodes."""
        complete_blocks = len(tokens) // self.block_size
        if len(block_ids) < complete_blocks:
            raise ValueError(
                f"prompt has {complete_blocks} complete blocks, "
                f"but only {len(block_ids)} physical blocks were supplied"
            )

        node = self.root
        for index in range(complete_blocks):
            start = index * self.block_size
            token_block = tuple(tokens[start:start + self.block_size])
            child = node.children.get(token_block)
            if child is None:
                owner_id = self._new_owner_id()
                block_id = block_ids[index]
                self.kv.allocator.reserve(owner_id, 0)
                try:
                    self.kv.allocator.incref(owner_id, block_id)
                except Exception:
                    self.kv.allocator.release_request(owner_id)
                    raise
                child = RadixNode(
                    token_block=token_block,
                    block_id=block_id,
                    owner_id=owner_id,
                    parent=node,
                )
                node.children[token_block] = child
                self.cached_blocks += 1
            self._touch(child)
            node = child

        while self.cached_blocks > self.max_blocks:
            if not self._evict_one(set()):
                break

    def evict_until_available(
        self,
        required_blocks: int,
        protected_block_ids: set[int] | None = None,
    ) -> bool:
        """Evict leaf nodes until `required_blocks` can be reserved."""
        protected = protected_block_ids or set()
        while self.kv.allocator.available_reservation_blocks < required_blocks:
            if not self._evict_one(protected):
                return False
        return True

    def _evict_one(self, protected_block_ids: set[int]) -> bool:
        leaves = [
            node
            for node in self._nodes()
            if not node.children and node.block_id not in protected_block_ids
        ]
        if not leaves:
            return False
        victim = min(leaves, key=lambda node: node.last_used)
        parent = victim.parent
        if parent is None or victim.token_block is None or victim.owner_id is None:
            raise RuntimeError("invalid radix-cache leaf")
        del parent.children[victim.token_block]
        self.kv.allocator.release_request(victim.owner_id)
        self.cached_blocks -= 1
        self.evictions += 1
        return True

    def _nodes(self) -> list[RadixNode]:
        nodes: list[RadixNode] = []
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def clear(self) -> None:
        """Drop every cached path and release all cache-owned block references."""
        for node in self._nodes():
            if node.owner_id is not None:
                self.kv.allocator.release_request(node.owner_id)
        self.root.children.clear()
        self.cached_blocks = 0
        self.hits = 0
        self.misses = 0
        self.matched_tokens = 0
        self.evictions = 0

    def stats(self) -> dict[str, int]:
        return {
            "cached_blocks": self.cached_blocks,
            "cached_tokens": self.cached_blocks * self.block_size,
            "hits": self.hits,
            "misses": self.misses,
            "matched_tokens": self.matched_tokens,
            "evictions": self.evictions,
        }
