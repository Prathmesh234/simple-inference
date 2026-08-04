"""Batched prompt prefill with prefix-page reuse."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ops.rope import rotate_half
from serving.paged_kv_cache import paged_attention_forward
from serving.request import Request


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


@dataclass
class PrefillBatch:
    """Prepared unmatched prompt suffixes and their absolute positions."""

    requests: list[Request]
    token_ids: torch.Tensor
    prefix_lengths: list[int]
    suffix_lengths: list[int]
    prompt_lengths: list[int]
    positions: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def max_suffix_len(self) -> int:
        return self.token_ids.shape[1]


class PrefillRunner:
    """Compute only prompt tokens not already covered by prefix-cache pages."""

    def __init__(
        self,
        model,
        kv_cache,
        prefix_cache,
    ):
        self.model = model
        self.kv = kv_cache
        self.prefix_cache = prefix_cache
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.n_heads_q = model.cfg.num_attention_heads
        self.n_heads_kv = model.cfg.num_key_value_heads
        self.head_dim = model.cfg.head_dim
        self.kv_groups = model.cfg.num_kv_groups

    def logits(self, requests: list[Request]) -> torch.Tensor:
        """Write unmatched prompt K/V and return first-token logits per request."""
        batch = self._prepare(requests)
        h = self.model.embed(batch.token_ids)
        for layer in self.model.layers:
            h = self._attention(h, layer, batch)
            h = h + layer.mlp(layer.mlp_norm(h))

        last_idx = torch.tensor(
            [length - 1 for length in batch.suffix_lengths],
            dtype=torch.long,
            device=self.device,
        )
        h_last = h[torch.arange(batch.size, device=self.device), last_idx]
        return self.model.head(
            self.model.norm(h_last.unsqueeze(1))
        )[:, -1, :]

    def publish(self, requests: list[Request]) -> None:
        """Pin complete prompt pages in the radix tree for later requests."""
        if self.prefix_cache is None:
            return
        for request in requests:
            self.prefix_cache.insert(
                request.prompt_tokens,
                self.kv.block_tables[request.id],
            )

    def _prepare(self, requests: list[Request]) -> PrefillBatch:
        prefix_lengths = [request.cached_prefix_len for request in requests]
        suffix_lengths = [
            request.prompt_len - prefix_len
            for request, prefix_len in zip(requests, prefix_lengths)
        ]
        if min(suffix_lengths) < 1:
            raise RuntimeError("prefix cache must leave at least one token for prefill")

        prompt_lengths = [request.prompt_len for request in requests]
        batch_size = len(requests)
        max_suffix_len = max(suffix_lengths)
        token_ids = torch.zeros(
            batch_size,
            max_suffix_len,
            dtype=torch.long,
            device=self.device,
        )
        for row, (request, prefix_len, suffix_len) in enumerate(
            zip(requests, prefix_lengths, suffix_lengths)
        ):
            token_ids[row, :suffix_len] = torch.tensor(
                request.prompt_tokens[prefix_len:],
                dtype=torch.long,
                device=self.device,
            )

        positions = (
            torch.tensor(prefix_lengths, dtype=torch.long, device=self.device)
            .view(batch_size, 1)
            + torch.arange(max_suffix_len, device=self.device).view(1, max_suffix_len)
        )
        for row, suffix_len in enumerate(suffix_lengths):
            positions[row, suffix_len:] = prefix_lengths[row]

        cos = (
            self.model.rope_freqs.cos[positions]
            .to(self.dtype)
            .view(batch_size, max_suffix_len, 1, self.head_dim)
        )
        sin = (
            self.model.rope_freqs.sin[positions]
            .to(self.dtype)
            .view(batch_size, max_suffix_len, 1, self.head_dim)
        )
        return PrefillBatch(
            requests=requests,
            token_ids=token_ids,
            prefix_lengths=prefix_lengths,
            suffix_lengths=suffix_lengths,
            prompt_lengths=prompt_lengths,
            positions=positions,
            cos=cos,
            sin=sin,
        )

    def _attention(
        self,
        h: torch.Tensor,
        layer,
        batch: PrefillBatch,
    ) -> torch.Tensor:
        attn = layer.attn
        layer_idx = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(
            batch.size, batch.max_suffix_len, self.n_heads_q, self.head_dim
        )
        k = F.linear(x, attn.wk).view(
            batch.size, batch.max_suffix_len, self.n_heads_kv, self.head_dim
        )
        v = F.linear(x, attn.wv).view(
            batch.size, batch.max_suffix_len, self.n_heads_kv, self.head_dim
        )
        q = _apply_rope(q, batch.cos, batch.sin)
        k = _apply_rope(k, batch.cos, batch.sin)
        q_transposed = q.transpose(1, 2)

        for row, request in enumerate(batch.requests):
            real_length = batch.suffix_lengths[row]
            self.kv.append(
                layer_idx,
                request.id,
                batch.prefix_lengths[row],
                k[row, :real_length].transpose(0, 1).contiguous(),
                v[row, :real_length].transpose(0, 1).contiguous(),
            )
        keys, values = self.kv.gather_batch(
            layer_idx,
            [request.id for request in batch.requests],
            batch.prompt_lengths,
            pad_to=max(batch.prompt_lengths),
        )
        output = paged_attention_forward(
            q_transposed,
            keys,
            values,
            batch.positions,
            torch.tensor(
                batch.prompt_lengths,
                dtype=torch.long,
                device=self.device,
            ),
            self.kv_groups,
        ).transpose(1, 2)

        output = output.reshape(
            batch.size,
            batch.max_suffix_len,
            self.n_heads_q * self.head_dim,
        )
        return h + F.linear(output, attn.wo)
