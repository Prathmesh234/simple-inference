"""
Bucketed CUDA-graph decode over the paged KV cache.

The graph captures fixed tensor ADDRESSES and SHAPES:

    input_ids:   (bucket, 1)
    positions:   (bucket,)
    block_tables:(bucket, max_blocks_per_request)
    cos/sin:     (bucket, 1, 1, head_dim)

Before replay, Python allocates any newly-needed physical page, then updates
those fixed buffers in place with `copy_()`. The captured Triton attention
kernel reads the new block-table VALUES and follows them directly into the
pre-allocated physical K/V pool.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from model.cuda_graph import capture_graph
from ops.rope import rotate_half


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE on `(B, 1, H, D)` using fixed per-row metadata buffers."""
    return x * cos + rotate_half(x) * sin


class PagedDecodeBuffers:
    """Fixed-address inputs and output for one graph batch-size bucket."""

    def __init__(self, B, max_blocks, head_dim, dtype, device):
        self.input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.positions = torch.zeros(B, dtype=torch.long, device=device)
        self.block_tables = torch.full(
            (B, max_blocks), -1, dtype=torch.long, device=device
        )
        self.cos = torch.zeros(B, 1, 1, head_dim, dtype=dtype, device=device)
        self.sin = torch.zeros(B, 1, 1, head_dim, dtype=dtype, device=device)
        self.logits: torch.Tensor | None = None


class PagedGraphDecoder:
    """Capture and replay one-token decode while K/V lives in physical pages."""

    def __init__(
        self,
        model,
        kv_cache,
        *,
        max_running: int,
        max_seq_len: int,
        n_heads_q: int,
        n_heads_kv: int,
        head_dim: int,
        kv_groups: int,
    ):
        if not kv_cache.scratch_block_ids:
            raise ValueError("paged CUDA graphs require one reserved scratch block")
        self.model = model
        self.kv = kv_cache
        self.device = kv_cache.device
        self.dtype = kv_cache.dtype
        self.max_running = max_running
        self.nq = n_heads_q
        self.nkv = n_heads_kv
        self.D = head_dim
        self.kv_groups = kv_groups
        self.block_size = kv_cache.block_size
        self.max_seq_len = max_seq_len
        self.max_blocks = math.ceil(max_seq_len / self.block_size)
        self.scratch_block = kv_cache.scratch_block_ids[0]
        self.bos = model.cfg.bos_token_id

        self.buckets = self._make_buckets(max_running)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.bufs: dict[int, PagedDecodeBuffers] = {}

    @staticmethod
    def _make_buckets(max_running: int) -> list[int]:
        out, size = [], 1
        while size < max_running:
            out.append(size)
            size *= 2
        out.append(max_running)
        return sorted(set(out))

    def bucket_for(self, running: int) -> int | None:
        for bucket in self.buckets:
            if bucket >= running:
                return bucket
        return None

    def _run_layers(self, bufs: PagedDecodeBuffers, B: int) -> torch.Tensor:
        model = self.model
        h = model.embed(bufs.input_ids)
        for layer in model.layers:
            h = self._attn(h, layer, bufs, B)
            h = h + layer.mlp(layer.mlp_norm(h))
        h = model.norm(h)
        return model.head(h)[:, -1, :]

    def _attn(self, h, layer, bufs: PagedDecodeBuffers, B: int) -> torch.Tensor:
        attn = layer.attn
        layer_idx = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(B, 1, self.nq, self.D)
        k = F.linear(x, attn.wk).view(B, 1, self.nkv, self.D)
        v = F.linear(x, attn.wv).view(B, 1, self.nkv, self.D)
        q = _apply_rope(q, bufs.cos, bufs.sin)
        k = _apply_rope(k, bufs.cos, bufs.sin)

        logical_blocks = torch.div(
            bufs.positions, self.block_size, rounding_mode="floor"
        )
        block_offsets = torch.remainder(bufs.positions, self.block_size)
        rows = torch.arange(B, device=self.device)
        physical_blocks = bufs.block_tables[rows, logical_blocks]

        # The pool pointer is stable. Only the index tensors' values change
        # between replays, selecting each request's current page and offset.
        self.kv.k_cache[layer_idx][physical_blocks, :, block_offsets, :] = k[:, 0]
        self.kv.v_cache[layer_idx][physical_blocks, :, block_offsets, :] = v[:, 0]

        from kernels.attention_kernel import attention_paged_decode_triton

        out = attention_paged_decode_triton(
            q.transpose(1, 2),
            self.kv.k_cache[layer_idx],
            self.kv.v_cache[layer_idx],
            bufs.block_tables,
            bufs.positions,
            self.block_size,
        )
        out = out.transpose(1, 2).reshape(B, 1, self.nq * self.D)
        return h + F.linear(out, attn.wo)

    def _seed_for_capture(self, bufs: PagedDecodeBuffers, B: int) -> None:
        """Make every warmup/capture row write only to the scratch page."""
        bufs.input_ids.fill_(self.bos)
        bufs.positions.zero_()
        bufs.block_tables.fill_(-1)
        bufs.block_tables[:, 0].fill_(self.scratch_block)
        cos0 = self.model.rope_freqs.cos[0:1].to(self.dtype).view(1, 1, 1, self.D)
        sin0 = self.model.rope_freqs.sin[0:1].to(self.dtype).view(1, 1, 1, self.D)
        bufs.cos.copy_(cos0)
        bufs.sin.copy_(sin0)

    def _fill_for_decode(
        self,
        bufs: PagedDecodeBuffers,
        B: int,
        request_ids: list[int],
        positions: list[int],
        last_tokens: list[int],
    ) -> None:
        """
        Allocate pages outside the graph, then update fixed metadata in place.

        Padding rows point at the permanently reserved scratch page at position
        zero. Their logits are discarded after replay.
        """
        running = len(request_ids)
        for request_id, position in zip(request_ids, positions):
            self.kv.ensure_capacity(request_id, position + 1)

        table_rows = []
        for request_id in request_ids:
            table = self.kv.block_tables[request_id]
            table_rows.append(table + [-1] * (self.max_blocks - len(table)))
        scratch_row = [self.scratch_block] + [-1] * (self.max_blocks - 1)
        table_rows.extend([scratch_row] * (B - running))

        padded_positions = positions + [0] * (B - running)
        padded_tokens = last_tokens + [self.bos] * (B - running)
        bufs.block_tables.copy_(
            torch.tensor(table_rows, dtype=torch.long, device=self.device)
        )
        bufs.positions.copy_(
            torch.tensor(padded_positions, dtype=torch.long, device=self.device)
        )
        bufs.input_ids.copy_(
            torch.tensor(padded_tokens, dtype=torch.long, device=self.device).view(B, 1)
        )
        bufs.cos.copy_(
            self.model.rope_freqs.cos[bufs.positions]
            .to(self.dtype)
            .view(B, 1, 1, self.D)
        )
        bufs.sin.copy_(
            self.model.rope_freqs.sin[bufs.positions]
            .to(self.dtype)
            .view(B, 1, 1, self.D)
        )

    @torch.no_grad()
    def capture(self, B: int) -> None:
        bufs = PagedDecodeBuffers(
            B, self.max_blocks, self.D, self.dtype, self.device
        )
        self.bufs[B] = bufs
        self._seed_for_capture(bufs, B)
        self.graphs[B], bufs.logits = capture_graph(
            lambda: self._run_layers(bufs, B)
        )

    def capture_all(self) -> None:
        for bucket in self.buckets:
            if bucket not in self.graphs:
                self.capture(bucket)

    @torch.no_grad()
    def logits(
        self,
        request_ids: list[int],
        positions: list[int],
        last_tokens: list[int],
    ) -> torch.Tensor | None:
        running = len(request_ids)
        bucket = self.bucket_for(running)
        if bucket is None:
            return None
        if bucket not in self.graphs:
            self.capture(bucket)
        bufs = self.bufs[bucket]
        self._fill_for_decode(
            bufs, bucket, request_ids, positions, last_tokens
        )
        self.graphs[bucket].replay()
        for request_id, position in zip(request_ids, positions):
            self.kv.lengths[request_id] = max(
                self.kv.lengths[request_id], position + 1
            )
        return bufs.logits[:running].clone()
