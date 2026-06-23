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
import torch.nn.functional as F

from ops.rope import rotate_half


@torch.no_grad()
def capture_graph(forward_fn, n_warmup: int = 3):
    """
    Capture a fixed-shape callable into a CUDA graph and return (graph, output).

    The capture recipe both decoders share:
      1. Run `forward_fn` a few times on a SIDE STREAM. This primes the caching
         allocator and — critically — completes any Triton autotuning, so no
         benchmarking `cudaDeviceSynchronize` sneaks into the recorded region.
      2. Sync, then record exactly one `forward_fn()` into a `torch.cuda.CUDAGraph`.

    `forward_fn` must read/write only fixed-address, fixed-shape buffers and must
    not perform host syncs (no `.item()`, no `.cpu()`); its return value is the
    captured output tensor, which later `graph.replay()` calls overwrite in place.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmup):
            forward_fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward_fn()
    return graph, output


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE on x (B, T, H, D); cos/sin broadcastable to (B, T, 1, D)."""
    return x * cos + rotate_half(x) * sin


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

        # Reuse the shared side-stream-warmup + record recipe. The captured
        # callable is the model's own forward, which — because kv.graph is set —
        # routes attention through the graph-friendly masked full-cache path.
        self.graph, self.static_logits = capture_graph(
            lambda: self.model(self.input_ids, kv_cache=self.kv), n_warmup
        )

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


# ════════════════════════════════════════════════════════════════════════════
#  Batched / continuous-batching decode graphs (used by serving/engine.py)
# ════════════════════════════════════════════════════════════════════════════


class BatchedDecodeBuffers:
    """Static per-bucket buffers the batched decode graph reads/writes.

    Like `GraphDecodeState`, but PER ROW: each of the B rows has its own slot,
    position, RoPE row and mask, because a continuous-batching decode mixes
    requests that sit at different absolute positions in different KV slots.
    Every tensor has a fixed address; each step copies fresh values in place
    before `graph.replay()`.
    """

    def __init__(self, B, max_seq_len, head_dim, dtype, device):
        self.input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
        self.slots = torch.zeros(B, dtype=torch.long, device=device)
        self.positions = torch.zeros(B, dtype=torch.long, device=device)
        self.cos = torch.zeros(B, 1, 1, head_dim, dtype=dtype, device=device)
        self.sin = torch.zeros(B, 1, 1, head_dim, dtype=dtype, device=device)
        # Additive mask over the FULL cache length, broadcast over (B, H, 1, L).
        self.mask = torch.zeros(B, 1, 1, max_seq_len, dtype=dtype, device=device)
        self.logits: torch.Tensor | None = None  # captured graph output


class BatchedGraphDecoder:
    """
    CUDA-graph decode for the continuous-batching engine: one graph per
    batch-size bucket, replayed per step.

    Why this is harder than `CUDAGraphDecoder`
    ------------------------------------------
    A graph needs fixed shapes/addresses and no host syncs. The single-stream
    decoder gets that for free (always a fixed `(B, 1)` forward). The serving
    decode does NOT: the running-set size `R` changes every iteration, the read
    length `max(pos)+1` changes, and the eager path calls `.item()`. Modern
    engines (vLLM / SGLang) reconcile this exactly the way we do here:

      1. **Bucket the batch size.** Capture one graph per preset `B ∈
         {1,2,4,…,max_running}`; a real decode of `R` rounds UP to the smallest
         bucket `B ≥ R` and the extra `B−R` rows are padding.
      2. **Read the full cache length under a mask** (not a dynamic `[:Lmax]`
         slice) so attention shapes are constant — same trick `CUDAGraphDecoder`
         uses, generalised to per-row masks.
      3. **A reserved scratch KV slot** (`max_running`, never handed out by the
         scheduler) absorbs padding rows so they can't scribble on a real
         request's KV. Their logits are sliced off.
      4. **Sampling stays outside the graph** — the caller samples the returned
         logits.

    This class takes PRIMITIVES (model, kv cache, head dims), not the engine or
    `Request` objects, so `model/` never imports `serving/`.
    """

    def __init__(self, model, kv_cache, *, max_running: int, n_heads_q: int,
                 n_heads_kv: int, head_dim: int, kv_groups: int):
        self.model = model
        self.kv = kv_cache
        self.device = kv_cache.device
        self.dtype = kv_cache.dtype
        self.nq = n_heads_q
        self.nkv = n_heads_kv
        self.D = head_dim
        self.gqa = kv_groups > 1
        self.max_seq_len = kv_cache.max_seq_len
        self.bos = model.cfg.bos_token_id

        self.max_running = max_running
        # The scheduler hands out slots [0, max_running); this reserved row holds
        # padding. The engine must size its KVCache with max_running+1 rows.
        self.scratch_slot = max_running

        self.buckets = self._make_buckets(max_running)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.bufs: dict[int, BatchedDecodeBuffers] = {}

    # ── bucket selection ──────────────────────────────────────────────────

    @staticmethod
    def _make_buckets(max_running: int) -> list[int]:
        """Powers of two up to max_running, plus max_running itself."""
        out, s = [], 1
        while s < max_running:
            out.append(s)
            s *= 2
        out.append(max_running)
        return sorted(set(out))

    def bucket_for(self, R: int) -> int | None:
        """Smallest captured-bucket size ≥ R, or None if R exceeds all buckets."""
        for b in self.buckets:
            if b >= R:
                return b
        return None

    # ── captured forward (static shapes only — no host syncs) ─────────────

    def _run_layers(self, bufs: BatchedDecodeBuffers, B: int) -> torch.Tensor:
        m = self.model
        h = m.embed(bufs.input_ids)  # (B, 1, hidden)
        for layer in m.layers:
            h = self._attn(h, layer, bufs, B)
            h = h + layer.mlp(layer.mlp_norm(h))
        h = m.norm(h)
        return m.head(h)[:, -1, :]   # (B, vocab)

    def _attn(self, h, layer, bufs: BatchedDecodeBuffers, B: int) -> torch.Tensor:
        attn = layer.attn
        li = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(B, 1, self.nq, self.D)
        k = F.linear(x, attn.wk).view(B, 1, self.nkv, self.D)
        v = F.linear(x, attn.wv).view(B, 1, self.nkv, self.D)
        q = _apply_rope(q, bufs.cos, bufs.sin)
        k = _apply_rope(k, bufs.cos, bufs.sin)

        # Scatter each row's new K/V into (slot_r, position_r). slots/positions
        # are static buffers, so replay follows their updated values without
        # re-capture (mirrors CUDAGraphDecoder's index_copy_ on cur_pos).
        self.kv.k_cache[li][bufs.slots, :, bufs.positions, :] = k[:, 0]
        self.kv.v_cache[li][bufs.slots, :, bufs.positions, :] = v[:, 0]

        # Fixed-length read over the whole cache row; the additive mask hides
        # positions beyond each row's pos (and confines padding rows to scratch).
        K = self.kv.k_cache[li][bufs.slots]   # (B, nkv, max_seq_len, D)
        V = self.kv.v_cache[li][bufs.slots]
        qT = q.transpose(1, 2)                # (B, nq, 1, D)
        out = F.scaled_dot_product_attention(qT, K, V, attn_mask=bufs.mask, enable_gqa=self.gqa)
        out = out.transpose(1, 2).reshape(B, 1, self.nq * self.D)
        return h + F.linear(out, attn.wo)

    # ── per-step buffer fills (run eagerly, before replay) ────────────────

    def _seed_for_capture(self, bufs: BatchedDecodeBuffers, B: int) -> None:
        """Safe placeholder values so warmup/capture touches only scratch."""
        bufs.slots.fill_(self.scratch_slot)
        bufs.positions.zero_()
        bufs.input_ids.fill_(self.bos)
        cos0 = self.model.rope_freqs.cos[0:1].to(self.dtype).view(1, 1, 1, self.D)
        sin0 = self.model.rope_freqs.sin[0:1].to(self.dtype).view(1, 1, 1, self.D)
        bufs.cos.copy_(cos0)  # broadcasts (1,1,1,D) → (B,1,1,D)
        bufs.sin.copy_(sin0)
        bufs.mask.fill_(float("-inf"))
        bufs.mask[..., 0] = 0.0  # attend position 0 only

    def _fill_for_decode(self, bufs: BatchedDecodeBuffers, B: int,
                         slots: list[int], positions: list[int],
                         last_tokens: list[int]) -> None:
        """Copy the real running set into the static buffers; pad the rest with
        the scratch slot at position 0 (its outputs are sliced off)."""
        R = len(slots)
        dev = self.device
        pad = B - R
        s_list = list(slots) + [self.scratch_slot] * pad
        p_list = list(positions) + [0] * pad
        t_list = list(last_tokens) + [self.bos] * pad

        bufs.slots.copy_(torch.tensor(s_list, dtype=torch.long, device=dev))
        bufs.positions.copy_(torch.tensor(p_list, dtype=torch.long, device=dev))
        bufs.input_ids.copy_(torch.tensor(t_list, dtype=torch.long, device=dev).view(B, 1))

        pos = bufs.positions
        bufs.cos.copy_(self.model.rope_freqs.cos[pos].to(self.dtype).view(B, 1, 1, self.D))
        bufs.sin.copy_(self.model.rope_freqs.sin[pos].to(self.dtype).view(B, 1, 1, self.D))

        cols = torch.arange(self.max_seq_len, device=dev)
        allowed = cols[None, :] <= pos[:, None]            # (B, max_seq_len)
        neg = torch.zeros(B, self.max_seq_len, dtype=self.dtype, device=dev)
        neg.masked_fill_(~allowed, float("-inf"))
        bufs.mask.copy_(neg.view(B, 1, 1, self.max_seq_len))

    # ── capture / replay ──────────────────────────────────────────────────

    @torch.no_grad()
    def capture(self, B: int) -> None:
        """Record the decode graph for batch size `B` (reuses `capture_graph`)."""
        bufs = BatchedDecodeBuffers(B, self.max_seq_len, self.D, self.dtype, self.device)
        self.bufs[B] = bufs
        self._seed_for_capture(bufs, B)
        self.graphs[B], bufs.logits = capture_graph(lambda: self._run_layers(bufs, B))

    def capture_all(self) -> None:
        """Pre-capture every bucket (call during warmup to avoid serving stalls)."""
        for b in self.buckets:
            if b not in self.graphs:
                self.capture(b)

    @torch.no_grad()
    def logits(self, slots: list[int], positions: list[int],
               last_tokens: list[int]) -> torch.Tensor | None:
        """
        Replay the captured decode for the given rows, returning logits
        (R, vocab), or None if `R` exceeds the biggest bucket (caller falls back
        to eager). The KV write happens INSIDE the graph, so on return the cache
        already holds each row's new token.
        """
        R = len(slots)
        B = self.bucket_for(R)
        if B is None:
            return None
        if B not in self.graphs:
            self.capture(B)
        bufs = self.bufs[B]
        self._fill_for_decode(bufs, B, slots, positions, last_tokens)
        self.graphs[B].replay()
        return bufs.logits[:R].clone()
