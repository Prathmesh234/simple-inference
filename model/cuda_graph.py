"""
CUDAGraphDecoder — capture the autoregressive decode step into a CUDA graph.

The problem this solves
-----------------------
At decode the GPU is memory-bound and mostly idle between kernels; wall-clock is
dominated by the CPU launching hundreds of tiny kernels per token (one set per
op across all 28 layers). torch.profiler on this engine shows decode is ~3x
CPU-bound (Self CPU >> Self CUDA). A CUDA graph captures that whole sequence of
launches once and replays it as a SINGLE launch, removing the per-op CPU
dispatch overhead.

Why decode (and not prefill)
----------------------------
CUDA graphs require *static shapes* and *no host syncs* in the captured region.
The decode step is the perfect fit: every token is a fixed (B, 1) forward. Prefill
is variable-length, runs once, and isn't the per-token bottleneck — it stays eager.

What had to become static (and how)
-----------------------------------
A naive decode forward has three position-dependent inputs that a single captured
graph cannot vary. We move each into a static GPU buffer that is updated IN PLACE
(eagerly, outside the graph) before every replay; the captured kernels read the
buffers by address, so they see the new values without re-capture:

  - cur_pos   : (1,) long   — where in the KV cache this token's K/V is written
                              (KVCache.update_graph uses index_copy_ at this index)
  - cos / sin : (1, D)      — RoPE tables for the current absolute position
                              (still fed through the custom Triton RoPE kernel)
  - attn_mask : (1,1,1,L)   — additive mask, 0 for cache slots <= cur_pos and
                              -inf beyond, so attention over the FULL fixed-length
                              cache ignores not-yet-written slots

Sampling (multinomial / .item()) stays OUTSIDE the graph — it is data-dependent
and would force a host sync.

Custom kernels inside the graph
-------------------------------
RMSNorm, RoPE and SwiGLU still run via their hand-written Triton kernels inside
the captured region (they are static-shape and capturable). Only attention swaps
to SDPA(enable_gqa, attn_mask): the custom flash kernel derives the query
position from tensor shapes, which a single fixed-shape graph cannot express, so
the masked full-cache read is the correct capturable form.

Capture lifecycle
-----------------
`capture()` warms up a few iterations on a side stream (this also completes any
Triton autotuning so no benchmarking sync happens during capture), then records
one decode forward. After capture `kv_cache.graph` is cleared so subsequent
eager prefills are unaffected — replay() bypasses Python entirely, so the graph
state only needs to be live at capture time.
"""

from __future__ import annotations

import torch


class GraphDecodeState:
    """Static per-step inputs the graphed forward reads (updated in place)."""

    def __init__(self, max_seq_len: int, head_dim: int,
                 dtype: torch.dtype, device: torch.device):
        self.cur_pos   = torch.zeros(1, dtype=torch.long, device=device)
        self.cos       = torch.zeros(1, head_dim, dtype=dtype, device=device)
        self.sin       = torch.zeros(1, head_dim, dtype=dtype, device=device)
        # Additive mask over the full cache length, broadcast over (B, H, 1, L).
        self.attn_mask = torch.zeros(1, 1, 1, max_seq_len, dtype=dtype, device=device)


class CUDAGraphDecoder:
    """
    Captures one decode (T=1) forward of `model` and replays it per token.

    Usage:
        dec = CUDAGraphDecoder(model, kv_cache, batch_size)
        dec.capture(first_decode_pos)
        logits = dec.decode(token_ids, pos)   # (B, 1, vocab)
    """

    def __init__(self, model, kv_cache, batch_size: int):
        self.model = model
        self.kv = kv_cache
        self.B = batch_size
        self.device = kv_cache.device
        self.dtype = kv_cache.dtype
        self.head_dim = model.cfg.head_dim
        self.max_seq_len = kv_cache.max_seq_len

        # Static input/output buffers (fixed addresses for the captured graph).
        self.input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        self.state = GraphDecodeState(self.max_seq_len, self.head_dim, self.dtype, self.device)

        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_logits: torch.Tensor | None = None

    # ── per-step buffer updates (run eagerly, before replay) ──────────────────

    def _set_pos(self, pos: int) -> None:
        """Point all position-dependent static buffers at absolute position `pos`."""
        self.state.cur_pos.fill_(pos)
        # RoPE rows for this position (tables are fp32; cast to the model dtype).
        self.state.cos.copy_(self.model.rope_freqs.cos[pos:pos + 1].to(self.dtype))
        self.state.sin.copy_(self.model.rope_freqs.sin[pos:pos + 1].to(self.dtype))
        # Mask: keep cache slots [0, pos], drop [pos+1, max_seq_len).
        self.state.attn_mask.fill_(float("-inf"))
        self.state.attn_mask[..., :pos + 1] = 0.0

    # ── capture ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def capture(self, warmup_pos: int, n_warmup: int = 3) -> None:
        """
        Warm up (also finishing Triton autotuning) then capture one decode step.

        `warmup_pos` should be the first position that will actually be decoded;
        the warmup overwrites that KV slot with garbage, which the first real
        `decode(..., warmup_pos)` then overwrites with the correct K/V.
        """
        self.kv.graph = self.state
        self._set_pos(warmup_pos)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmup):
                self.model(self.input_ids, kv_cache=self.kv)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model(self.input_ids, kv_cache=self.kv)

        # Replay is pure CUDA — Python attention never re-runs — so the graph
        # state only needs to be live at capture time. Clear it so eager prefill
        # of the next sequence uses the normal path.
        self.kv.graph = None

    # ── replay ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def decode(self, token_ids: torch.Tensor, pos: int) -> torch.Tensor:
        """
        Replay the captured decode step for `token_ids` at absolute `pos`.

        Args:
            token_ids: (B, 1) or (B,) long tensor of the tokens to feed.
            pos:       absolute position of these tokens in the sequence.

        Returns:
            logits (B, 1, vocab) — a view of the static output buffer; consume
            (e.g. sample) before the next decode() call overwrites it.
        """
        self.input_ids.copy_(token_ids.view(self.B, 1))
        self._set_pos(pos)
        self.graph.replay()
        return self.static_logits
