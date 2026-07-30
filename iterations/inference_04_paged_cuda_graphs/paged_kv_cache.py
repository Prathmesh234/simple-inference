"""
Paged KV cache and the correctness-first gathered attention path.

The old cache reserves a rectangle for every request:

    cache[layer, request_slot, kv_head, absolute_position, head_dim]

That gives every short request a full `max_seq_len` row. This cache instead
stores fixed-size physical pages:

    key_pool[layer, physical_block, kv_head, block_offset, head_dim]

Each request owns a small Python block table:

    logical block 0 -> physical block 37
    logical block 1 -> physical block 4
    ...

For a token at absolute position `p`, the write location is:

    logical_block = p // block_size
    block_offset  = p % block_size
    physical_block = block_tables[request_id][logical_block]

`gather()` materializes a request's logically contiguous K/V prefix before
calling PyTorch SDPA. This has an extra copy, intentionally: it makes page
tables and allocation easy to inspect and verify before replacing the gather
with a Triton kernel that reads pages directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .block_allocator import BlockAllocator


class PagedKVCache:
    """A pre-allocated physical KV page pool addressed by per-request tables."""

    def __init__(
        self,
        n_layers: int,
        num_blocks: int,
        block_size: int,
        n_heads_kv: int,
        head_dim: int,
        num_scratch_blocks: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        if min(n_layers, num_blocks, block_size, n_heads_kv, head_dim) <= 0:
            raise ValueError("all paged KV-cache dimensions must be positive")
        if num_scratch_blocks < 0:
            raise ValueError(
                f"num_scratch_blocks must be non-negative, got {num_scratch_blocks}"
            )
        self.n_layers = n_layers
        # `num_blocks` is request-allocatable capacity. Scratch blocks live in
        # the same physical tensor but are never visible to BlockAllocator.
        self.num_blocks = num_blocks
        self.num_scratch_blocks = num_scratch_blocks
        self.total_blocks = num_blocks + num_scratch_blocks
        self.scratch_block_ids = list(range(num_blocks, self.total_blocks))
        self.block_size = block_size
        self.n_heads_kv = n_heads_kv
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.allocator = BlockAllocator(num_blocks)

        shape = (n_layers, self.total_blocks, n_heads_kv, block_size, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.block_tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    def blocks_for_tokens(self, token_count: int) -> int:
        """Number of fixed-size pages needed to hold `token_count` positions."""
        if token_count < 0:
            raise ValueError(f"token_count must be non-negative, got {token_count}")
        return math.ceil(token_count / self.block_size)

    def can_reserve(self, request_id: int, max_tokens: int) -> bool:
        """Whether this request's declared maximum sequence length fits."""
        return self.allocator.can_reserve(request_id, self.blocks_for_tokens(max_tokens))

    def reserve_request(self, request_id: int, max_tokens: int) -> None:
        """
        Reserve eventual capacity, without allocating physical pages yet.

        Reserving at admission prevents a request from reaching a page boundary
        later and deadlocking all active requests because every physical block
        has already been consumed by other growing sequences.
        """
        block_count = self.blocks_for_tokens(max_tokens)
        self.allocator.reserve(request_id, block_count)
        self.block_tables[request_id] = []
        self.lengths[request_id] = 0

    def append(
        self,
        layer_idx: int,
        request_id: int,
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Write contiguous logical positions into their possibly discontiguous pages.

        `k` and `v` are `(n_heads_kv, T, head_dim)`. A prefill may cross several
        page boundaries; the loop splits it into page-local `copy_` operations.
        Decode has `T=1`, so it usually performs one small write.
        """
        self._validate_layer(layer_idx)
        if request_id not in self.block_tables:
            raise KeyError(f"request {request_id} has no paged KV reservation")
        if start_pos < 0:
            raise ValueError(f"start_pos must be non-negative, got {start_pos}")
        self._validate_kv(k, v)

        token_count = k.shape[1]
        end_pos = start_pos + token_count
        self._ensure_capacity(request_id, end_pos)

        source_start = 0
        pos = start_pos
        while pos < end_pos:
            logical_block = pos // self.block_size
            block_offset = pos % self.block_size
            width = min(end_pos - pos, self.block_size - block_offset)
            physical_block = self.block_tables[request_id][logical_block]
            source_end = source_start + width
            self.k_cache[layer_idx, physical_block, :, block_offset:block_offset + width, :].copy_(
                k[:, source_start:source_end, :]
            )
            self.v_cache[layer_idx, physical_block, :, block_offset:block_offset + width, :].copy_(
                v[:, source_start:source_end, :]
            )
            pos += width
            source_start = source_end

        self.lengths[request_id] = max(self.lengths[request_id], end_pos)

    def gather(
        self,
        layer_idx: int,
        request_id: int,
        length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Materialize a logical `(Hkv, T, D)` K/V prefix from physical pages.

        The indexed page read is the key teaching point: the request's block
        table translates logical sequence order back into the physical pool.
        """
        self._validate_layer(layer_idx)
        if request_id not in self.block_tables:
            raise KeyError(f"request {request_id} has no paged KV reservation")
        if length is None:
            length = self.lengths[request_id]
        if not 0 <= length <= self.lengths[request_id]:
            raise ValueError(
                f"requested length {length} outside written prefix "
                f"[0, {self.lengths[request_id]}] for request {request_id}"
            )
        if length == 0:
            empty = torch.empty(
                self.n_heads_kv, 0, self.head_dim, dtype=self.dtype, device=self.device
            )
            return empty, empty.clone()

        table = self.block_tables[request_id][:self.blocks_for_tokens(length)]
        block_ids = torch.tensor(table, dtype=torch.long, device=self.device)
        k_pages = self.k_cache[layer_idx].index_select(0, block_ids)
        v_pages = self.v_cache[layer_idx].index_select(0, block_ids)
        # (pages, Hkv, block_size, D) -> (Hkv, pages * block_size, D).
        k = k_pages.permute(1, 0, 2, 3).reshape(self.n_heads_kv, -1, self.head_dim)
        v = v_pages.permute(1, 0, 2, 3).reshape(self.n_heads_kv, -1, self.head_dim)
        return k[:, :length, :], v[:, :length, :]

    def gather_batch(
        self,
        layer_idx: int,
        request_ids: list[int],
        lengths: list[int],
        pad_to: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather each request then right-pad to one batched SDPA tensor.

        Padding contains zeros and is always excluded by
        `paged_attention_forward`'s per-row length mask.
        """
        if len(request_ids) != len(lengths):
            raise ValueError("request_ids and lengths must have the same length")
        if not request_ids:
            raise ValueError("cannot gather an empty request batch")
        max_length = max(lengths) if pad_to is None else pad_to
        if max_length < max(lengths):
            raise ValueError(f"pad_to {max_length} is shorter than a requested KV prefix")

        batch_size = len(request_ids)
        k_batch = torch.zeros(
            batch_size, self.n_heads_kv, max_length, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        v_batch = torch.zeros_like(k_batch)
        for row, (request_id, length) in enumerate(zip(request_ids, lengths)):
            k, v = self.gather(layer_idx, request_id, length)
            k_batch[row, :, :length, :] = k
            v_batch[row, :, :length, :] = v
        return k_batch, v_batch

    def ensure_capacity(self, request_id: int, end_pos: int) -> None:
        """
        Allocate any page needed to write logical positions below `end_pos`.

        Eager append calls this internally. CUDA-graph decode calls it before
        replay because page allocation and Python block-table growth must stay
        outside the captured graph.
        """
        if request_id not in self.block_tables:
            raise KeyError(f"request {request_id} has no paged KV reservation")
        if end_pos < 0:
            raise ValueError(f"end_pos must be non-negative, got {end_pos}")
        self._ensure_capacity(request_id, end_pos)

    def release_request(self, request_id: int) -> None:
        """Free all physical pages and remove the request's logical block table."""
        self.allocator.release_request(request_id)
        self.block_tables.pop(request_id, None)
        self.lengths.pop(request_id, None)

    def reset(self) -> None:
        """Release all live page tables; pool contents are overwritten on reuse."""
        for request_id in list(self.block_tables):
            self.release_request(request_id)

    def bytes(self) -> int:
        """Physical K+V pool footprint, independent of live request lengths."""
        return self.k_cache.numel() * self.k_cache.element_size() * 2

    def _ensure_capacity(self, request_id: int, end_pos: int) -> None:
        required_blocks = self.blocks_for_tokens(end_pos)
        table = self.block_tables[request_id]
        while len(table) < required_blocks:
            table.append(self.allocator.allocate(request_id))

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} outside [0, {self.n_layers})")

    def _validate_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError(
                f"expected rank-3 K/V tensors, got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        expected = (self.n_heads_kv, k.shape[1], self.head_dim)
        if k.shape != expected or v.shape != expected:
            raise ValueError(
                f"expected K/V shape ({self.n_heads_kv}, T, {self.head_dim}), "
                f"got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        if k.device != self.device or v.device != self.device:
            raise ValueError("K/V tensors must be on the paged KV-cache device")
        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise ValueError("K/V tensors must match the paged KV-cache dtype")

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(layers={self.n_layers}, blocks={self.num_blocks}, "
            f"scratch_blocks={self.num_scratch_blocks}, "
            f"block_size={self.block_size}, allocated={self.num_blocks - self.allocator.free_blocks}, "
            f"reserved={self.allocator.reserved_blocks}, vram={self.bytes() / 1e9:.2f} GB)"
        )


def paged_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    key_lengths: torch.Tensor,
    num_kv_groups: int,
) -> torch.Tensor:
    """
    Run causal attention over gathered page-table K/V tensors.

    Args:
        q: `(B, Hq, Tq, D)`.
        k/v: `(B, Hkv, Tk, D)` materialized by `PagedKVCache.gather_batch`.
        query_positions: `(B, Tq)` absolute logical positions for every query.
        key_lengths: `(B,)` number of valid gathered keys for each request.
        num_kv_groups: number of Q heads sharing each KV head (GQA).

    The additive mask is the bridge from a ragged set of block tables to dense
    PyTorch SDPA. A later Triton paged-attention kernel will perform this
    logical-to-physical lookup inside its K/V loop and remove the gather.
    """
    B, _, Tq, _ = q.shape
    _, _, Tk, _ = k.shape
    if query_positions.shape != (B, Tq):
        raise ValueError(
            f"query_positions must be {(B, Tq)}, got {tuple(query_positions.shape)}"
        )
    if key_lengths.shape != (B,):
        raise ValueError(f"key_lengths must be {(B,)}, got {tuple(key_lengths.shape)}")

    key_positions = torch.arange(Tk, device=q.device).view(1, 1, Tk)
    allowed = (
        (key_positions < key_lengths.view(B, 1, 1))
        & (key_positions <= query_positions.unsqueeze(-1))
    )
    mask = torch.zeros(B, 1, Tq, Tk, dtype=q.dtype, device=q.device)
    mask.masked_fill_(~allowed.unsqueeze(1), float("-inf"))
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, enable_gqa=(num_kv_groups > 1)
    )
