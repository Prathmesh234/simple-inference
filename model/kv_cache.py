"""
KVCache — static, pre-allocated key/value cache for autoregressive decoding.

Why a KV cache?
---------------
During generation, each new token's attention needs the K and V vectors of
EVERY previous token. Without a cache, every decode step would re-run the
full forward pass over the entire prefix → O(T²) work to produce T tokens.

With a cache, prefill computes K/V for the prompt once, stores them, and
each decode step:
  - computes K/V only for the single new token  (O(1) per step per layer)
  - appends them to the cache
  - reads the entire cached K/V slice for attention

Net result: decode goes from O(T²) → O(T) total work, the difference between
"unusable" and "real-time" generation.

Layout
------
The cache is one big pre-allocated pool, sized for the worst case:

    k_cache: (n_layers, max_batch, n_heads_kv, max_seq_len, head_dim)
    v_cache: same shape

This shape matches the post-transpose layout in GroupedQueryAttention
(B, n_heads_kv, T, head_dim), so we can copy in/out via plain slicing
along dim=3 with zero reshape cost.

Memory
------
Llama 3.2-3B:
  n_layers=28, n_heads_kv=8, head_dim=128, bf16 (2 bytes)
  per-token-per-layer (K+V) = 2 * 8 * 128 * 2 = 4096 B = 4 KB
  per-token across all layers = 28 * 4 KB = 112 KB

So a single 8k-context request uses ~900 MB; a batch of 8 at 8k ≈ 7 GB.
That's why Phase 3 introduces PagedAttention — this pre-allocation is wasteful.

Position management
-------------------
The cache itself is stateless w.r.t. position — the *caller* (the model
forward) supplies `start_pos`. This keeps the cache trivially compatible
with two phases per generation step:

  Prefill   : start_pos=0,  T=prompt_len   → writes slots [0, T),       reads [0, T)
  Decode k  : start_pos=k,  T=1            → writes slot  [k, k+1),     reads [0, k+1)

`reset()` is just a logical no-op (the pool is overwritten on next prefill);
it exists for clarity when callers swap sequences in/out.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        n_layers: int,
        max_batch: int,
        max_seq_len: int,
        n_heads_kv: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        """
        Pre-allocate the full K and V pools.

        Args:
            n_layers:    number of transformer layers (28 for Llama 3.2-3B)
            max_batch:   maximum batch size this cache will ever hold
            max_seq_len: maximum sequence length (prompt + generated tokens)
            n_heads_kv:  number of KV heads (8 for GQA in Llama 3.2-3B)
            head_dim:    dim per head (128)
            dtype:       storage dtype — match the model (bfloat16)
            device:      cuda
        """
        self.n_layers    = n_layers
        self.max_batch   = max_batch
        self.max_seq_len = max_seq_len
        self.n_heads_kv  = n_heads_kv
        self.head_dim    = head_dim
        self.dtype       = dtype
        self.device      = torch.device(device)
        # Optional CUDA-graph decode state (see model/cuda_graph.py). When set to
        # a GraphDecodeState, `update` switches to a static, fixed-shape write so
        # the decode step can be captured into a torch.cuda.CUDAGraph. None for
        # the normal eager path.
        self.graph = None
        ##this is the limitation paged attention addresses 
        ## so right now we are blocking for the entire seq len, but in reality we never know what is going to be the user 
        ## sequence request. For all you know we blocked 1024 tokens worth of KV Cache memory but the user 
        ## sent a request with like 10 tokens wasting 1014 slots 
        shape = (n_layers, max_batch, n_heads_kv, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)

    # ── core API ─────────────────────────────────────────────────────────────

    def update(
        self,
        layer_idx: int,
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K/V into the cache at [start_pos, start_pos + T) and return
        the cached slice up to start_pos + T for attention.

        Args:
            layer_idx: which transformer layer this call belongs to
            start_pos: absolute position of the first new token (0 for prefill)
            k:         new K, shape (B, n_heads_kv, T, head_dim)
            v:         new V, shape (B, n_heads_kv, T, head_dim)

        Returns:
            k_full, v_full — shape (B, n_heads_kv, start_pos + T, head_dim)
            views into the underlying pool — DO NOT mutate.
        """
        B, H, T, D = k.shape
        end = start_pos + T

        assert B <= self.max_batch,  f"batch {B} > max_batch {self.max_batch}"
        assert H == self.n_heads_kv, f"n_heads_kv mismatch: got {H}, expected {self.n_heads_kv}"
        assert D == self.head_dim,   f"head_dim mismatch: got {D}, expected {self.head_dim}"
        assert end <= self.max_seq_len, (
            f"sequence overflow: start_pos+T={end} > max_seq_len={self.max_seq_len}"
        )

        # Write new K/V into their slots
        self.k_cache[layer_idx, :B, :, start_pos:end, :] = k
        self.v_cache[layer_idx, :B, :, start_pos:end, :] = v

        # Return the full prefix up to `end` for attention to read
        k_full = self.k_cache[layer_idx, :B, :, :end, :]
        v_full = self.v_cache[layer_idx, :B, :, :end, :]
        return k_full, v_full

    def update_graph(
        self,
        layer_idx: int,
        pos: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CUDA-graph-friendly decode write (T=1).

        Unlike `update`, this returns a FIXED-shape view of the whole cache
        (`[:, :, :max_seq_len, :]`) every step, so the captured graph never sees
        a changing shape. The single new K/V is written at the GPU-resident
        position `pos` via `index_copy_`, so the write target varies on replay
        without re-capture (a Python-int slice would bake the position in).

        Correctness during attention relies on a length mask (built in
        GraphDecodeState) that masks out cache slots beyond `pos`.

        Args:
            layer_idx: which transformer layer
            pos:       (1,) long tensor — absolute position of the decode token
            k:         (B, n_heads_kv, 1, head_dim)
            v:         (B, n_heads_kv, 1, head_dim)

        Returns:
            k_full, v_full — (B, n_heads_kv, max_seq_len, head_dim) static views.
        """
        B = k.shape[0]
        self.k_cache[layer_idx, :B].index_copy_(2, pos, k.contiguous())
        self.v_cache[layer_idx, :B].index_copy_(2, pos, v.contiguous())
        return self.k_cache[layer_idx, :B], self.v_cache[layer_idx, :B]

    def reset(self) -> None:
        """
        Logical reset between independent sequences.

        We don't actually zero the buffers — the next prefill overwrites
        from slot 0, and attention only reads slices up to start_pos + T,
        so any stale data beyond that window is unreachable. This method
        exists so callers have a clear hook when swapping sequences.
        """
        # If you ever need a hard wipe (e.g., debugging), uncomment:
        # self.k_cache.zero_()
        # self.v_cache.zero_()
        return

    # ── introspection helpers ────────────────────────────────────────────────

    def bytes(self) -> int:
        """Total VRAM footprint of the cache (K + V) in bytes."""
        return 2 * self.k_cache.numel() * self.k_cache.element_size()

    def __repr__(self) -> str:
        gb = self.bytes() / 1e9
        return (
            f"KVCache(n_layers={self.n_layers}, max_batch={self.max_batch}, "
            f"max_seq_len={self.max_seq_len}, n_heads_kv={self.n_heads_kv}, "
            f"head_dim={self.head_dim}, dtype={self.dtype}, vram={gb:.2f} GB)"
        )
