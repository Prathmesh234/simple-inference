"""Optimized paged decode with optional CUDA-graph replay."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from model.cuda_graph import capture_graph
from ops.rope import rotate_half
from serving.paged_kv_cache import PagedKVCache, paged_attention_forward


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class PagedDecodeBuffers:
    """Reusable decode inputs for direct eager execution or graph replay."""

    def __init__(self, batch_size, max_blocks, head_dim, dtype, device):
        self.input_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )
        self.positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.block_tables = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.long, device=device
        )
        self.cos = torch.zeros(
            batch_size, 1, 1, head_dim, dtype=dtype, device=device
        )
        self.sin = torch.zeros_like(self.cos)
        self.logits: torch.Tensor | None = None


class PagedDecoder:
    """
    Run one direct page-table Triton decode forward.

    CUDA graphs change only the launch mode: graph-off calls `_run_layers`
    eagerly, while graph-on captures and replays that exact same function.
    """

    def __init__(
        self,
        model,
        kv_cache: PagedKVCache,
        *,
        max_running: int,
        max_seq_len: int,
        n_heads_q: int,
        n_heads_kv: int,
        head_dim: int,
        use_triton: bool,
        use_cuda_graphs: bool,
    ):
        self.use_triton = use_triton and kv_cache.device.type == "cuda"
        if use_cuda_graphs and not self.use_triton:
            use_cuda_graphs = False
        if use_cuda_graphs and not kv_cache.scratch_block_ids:
            raise ValueError("paged CUDA graphs require one reserved scratch block")
        self.model = model
        self.kv = kv_cache
        self.device = kv_cache.device
        self.dtype = kv_cache.dtype
        self.max_running = max_running
        self.nq = n_heads_q
        self.nkv = n_heads_kv
        self.kv_groups = n_heads_q // n_heads_kv
        self.head_dim = head_dim
        self.block_size = kv_cache.block_size
        self.max_blocks = math.ceil(max_seq_len / self.block_size)
        self.bos_token_id = model.cfg.bos_token_id
        self.use_cuda_graphs = use_cuda_graphs
        self.scratch_block = (
            kv_cache.scratch_block_ids[0]
            if kv_cache.scratch_block_ids
            else None
        )

        self.buckets = self._make_buckets(max_running)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.buffers: dict[int, PagedDecodeBuffers] = {}

    @property
    def backend(self) -> str:
        if not self.use_triton:
            return "paged_gathered_sdpa_fallback"
        launch = "cuda_graph" if self.use_cuda_graphs else "eager"
        return f"paged_triton_{launch}"

    @staticmethod
    def _make_buckets(max_running: int) -> list[int]:
        buckets, size = [], 1
        while size < max_running:
            buckets.append(size)
            size *= 2
        buckets.append(max_running)
        return sorted(set(buckets))

    def bucket_for(self, running: int) -> int | None:
        for bucket in self.buckets:
            if bucket >= running:
                return bucket
        return None

    def _get_buffers(self, batch_size: int) -> PagedDecodeBuffers:
        buffers = self.buffers.get(batch_size)
        if buffers is None:
            buffers = PagedDecodeBuffers(
                batch_size,
                self.max_blocks,
                self.head_dim,
                self.dtype,
                self.device,
            )
            self.buffers[batch_size] = buffers
        return buffers

    def _run_layers(
        self,
        buffers: PagedDecodeBuffers,
        batch_size: int,
    ) -> torch.Tensor:
        h = self.model.embed(buffers.input_ids)
        for layer in self.model.layers:
            h = self._attention(h, layer, buffers, batch_size)
            h = h + layer.mlp(layer.mlp_norm(h))
        h = self.model.norm(h)
        return self.model.head(h)[:, -1, :]

    def _attention(
        self,
        h: torch.Tensor,
        layer,
        buffers: PagedDecodeBuffers,
        batch_size: int,
    ) -> torch.Tensor:
        attn = layer.attn
        layer_idx = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(
            batch_size, 1, self.nq, self.head_dim
        )
        k = F.linear(x, attn.wk).view(
            batch_size, 1, self.nkv, self.head_dim
        )
        v = F.linear(x, attn.wv).view(
            batch_size, 1, self.nkv, self.head_dim
        )
        q = _apply_rope(q, buffers.cos, buffers.sin)
        k = _apply_rope(k, buffers.cos, buffers.sin)

        logical_blocks = torch.div(
            buffers.positions,
            self.block_size,
            rounding_mode="floor",
        )
        block_offsets = torch.remainder(buffers.positions, self.block_size)
        rows = torch.arange(batch_size, device=self.device)
        physical_blocks = buffers.block_tables[rows, logical_blocks]

        self.kv.write_decode_batch(
            layer_idx,
            physical_blocks,
            block_offsets,
            k[:, 0],
            v[:, 0],
        )

        from kernels.attention_kernel import attention_paged_decode_triton

        out = attention_paged_decode_triton(
            q.transpose(1, 2),
            self.kv.k_cache[layer_idx],
            self.kv.v_cache[layer_idx],
            buffers.block_tables,
            buffers.positions,
            self.block_size,
        )
        out = out.transpose(1, 2).reshape(
            batch_size, 1, self.nq * self.head_dim
        )
        return h + F.linear(out, attn.wo)

    def _seed_for_capture(
        self,
        buffers: PagedDecodeBuffers,
        batch_size: int,
    ) -> None:
        if self.scratch_block is None:
            raise RuntimeError("CUDA-graph capture requires a scratch block")
        buffers.input_ids.fill_(self.bos_token_id)
        buffers.positions.zero_()
        buffers.block_tables.fill_(-1)
        buffers.block_tables[:, 0].fill_(self.scratch_block)
        cos0 = self.model.rope_freqs.cos[0:1].to(self.dtype).view(
            1, 1, 1, self.head_dim
        )
        sin0 = self.model.rope_freqs.sin[0:1].to(self.dtype).view(
            1, 1, 1, self.head_dim
        )
        buffers.cos.copy_(cos0)
        buffers.sin.copy_(sin0)

    def _fill(
        self,
        buffers: PagedDecodeBuffers,
        batch_size: int,
        request_ids: list[int],
        positions: list[int],
        last_tokens: list[int],
    ) -> None:
        running = len(request_ids)
        for request_id, position in zip(request_ids, positions):
            self.kv.ensure_capacity(request_id, position + 1)

        table_rows = []
        for request_id in request_ids:
            table = self.kv.block_tables[request_id]
            table_rows.append(table + [-1] * (self.max_blocks - len(table)))

        padding = batch_size - running
        if padding:
            if self.scratch_block is None:
                raise RuntimeError("padded graph decode requires a scratch block")
            scratch_row = [self.scratch_block] + [-1] * (self.max_blocks - 1)
            table_rows.extend([scratch_row] * padding)

        buffers.block_tables.copy_(
            torch.tensor(table_rows, dtype=torch.long, device=self.device)
        )
        buffers.positions.copy_(
            torch.tensor(
                positions + [0] * padding,
                dtype=torch.long,
                device=self.device,
            )
        )
        buffers.input_ids.copy_(
            torch.tensor(
                last_tokens + [self.bos_token_id] * padding,
                dtype=torch.long,
                device=self.device,
            ).view(batch_size, 1)
        )
        buffers.cos.copy_(
            self.model.rope_freqs.cos[buffers.positions]
            .to(self.dtype)
            .view(batch_size, 1, 1, self.head_dim)
        )
        buffers.sin.copy_(
            self.model.rope_freqs.sin[buffers.positions]
            .to(self.dtype)
            .view(batch_size, 1, 1, self.head_dim)
        )

    @torch.no_grad()
    def capture(self, batch_size: int) -> None:
        if not self.use_cuda_graphs:
            return
        buffers = self._get_buffers(batch_size)
        self._seed_for_capture(buffers, batch_size)
        self.graphs[batch_size], buffers.logits = capture_graph(
            lambda: self._run_layers(buffers, batch_size)
        )

    def capture_all(self) -> None:
        if not self.use_cuda_graphs:
            return
        for bucket in self.buckets:
            if bucket not in self.graphs:
                self.capture(bucket)

    def disable_cuda_graphs(self) -> None:
        """Drop captured graphs while retaining the direct eager Triton path."""
        self.use_cuda_graphs = False
        self.graphs.clear()
        self.buffers.clear()

    def _run_gathered(
        self,
        request_ids: list[int],
        positions: list[int],
        last_tokens: list[int],
    ) -> torch.Tensor:
        """PyTorch SDPA fallback for CPU or explicit `USE_TRITON=false`."""
        batch_size = len(request_ids)
        position_tensor = torch.tensor(
            positions,
            dtype=torch.long,
            device=self.device,
        )
        token_tensor = torch.tensor(
            last_tokens,
            dtype=torch.long,
            device=self.device,
        ).view(batch_size, 1)
        h = self.model.embed(token_tensor)
        cos = (
            self.model.rope_freqs.cos[position_tensor]
            .to(self.dtype)
            .view(batch_size, 1, 1, self.head_dim)
        )
        sin = (
            self.model.rope_freqs.sin[position_tensor]
            .to(self.dtype)
            .view(batch_size, 1, 1, self.head_dim)
        )

        for layer in self.model.layers:
            attn = layer.attn
            layer_idx = attn.layer_idx
            x = layer.attn_norm(h)
            q = F.linear(x, attn.wq).view(
                batch_size, 1, self.nq, self.head_dim
            )
            k = F.linear(x, attn.wk).view(
                batch_size, 1, self.nkv, self.head_dim
            )
            v = F.linear(x, attn.wv).view(
                batch_size, 1, self.nkv, self.head_dim
            )
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

            lengths = [position + 1 for position in positions]
            for row, request_id in enumerate(request_ids):
                self.kv.append(
                    layer_idx,
                    request_id,
                    positions[row],
                    k[row, 0].unsqueeze(1).contiguous(),
                    v[row, 0].unsqueeze(1).contiguous(),
                )
            keys, values = self.kv.gather_batch(
                layer_idx,
                request_ids,
                lengths,
            )
            output = paged_attention_forward(
                q.transpose(1, 2),
                keys,
                values,
                position_tensor.view(batch_size, 1),
                torch.tensor(
                    lengths,
                    dtype=torch.long,
                    device=self.device,
                ),
                self.kv_groups,
            )
            output = output.transpose(1, 2).reshape(
                batch_size, 1, self.nq * self.head_dim
            )
            h = h + F.linear(output, attn.wo)
            h = h + layer.mlp(layer.mlp_norm(h))

        h = self.model.norm(h)
        return self.model.head(h)[:, -1, :]

    @torch.no_grad()
    def logits(
        self,
        request_ids: list[int],
        positions: list[int],
        last_tokens: list[int],
    ) -> torch.Tensor:
        running = len(request_ids)
        if not self.use_triton:
            return self._run_gathered(request_ids, positions, last_tokens)

        batch_size = self.bucket_for(running) if self.use_cuda_graphs else running
        if batch_size is None:
            raise ValueError(
                f"decode batch {running} exceeds max_running {self.max_running}"
            )

        buffers = self._get_buffers(batch_size)
        self._fill(
            buffers,
            batch_size,
            request_ids,
            positions,
            last_tokens,
        )
        if self.use_cuda_graphs:
            if batch_size not in self.graphs:
                self.capture(batch_size)
                self._fill(
                    buffers,
                    batch_size,
                    request_ids,
                    positions,
                    last_tokens,
                )
            self.graphs[batch_size].replay()
            logits = buffers.logits
        else:
            logits = self._run_layers(buffers, batch_size)

        for request_id, position in zip(request_ids, positions):
            self.kv.lengths[request_id] = max(
                self.kv.lengths[request_id],
                position + 1,
            )
        return logits[:running].clone()
