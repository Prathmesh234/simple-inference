"""
Grouped Query Attention (GQA).

Standard Multi-Head Attention recap
-------------------------------------
Every token produces Q, K, V vectors. For a sequence of T tokens:
  - Q: (T, n_heads, head_dim)  — what am I looking for?
  - K: (T, n_heads, head_dim)  — what do I contain?
  - V: (T, n_heads, head_dim)  — what do I pass forward if attended to?

Attention score for head h, query position i, key position j:
  score[h,i,j] = Q[h,i] · K[h,j] / sqrt(head_dim)

Apply causal mask (can't attend to future positions), softmax over j,
then weighted sum over V.

What GQA changes
-----------------
In standard MHA, every head has its own K and V projections.
GQA uses fewer KV heads than Q heads — here 8 KV heads vs 24 Q heads.
Each KV head is shared by 3 Q heads (num_kv_groups = 24 / 8 = 3).

Memory saving: KV cache stores K and V for every layer and every position.
  MHA:  n_heads_q  KV heads = 24 heads → 24 × head_dim per token per layer
  GQA:  n_heads_kv KV heads =  8 heads →  8 × head_dim per token per layer
  Saving: 3× less KV cache memory at no meaningful quality loss.

Implementation: repeat each KV head 3 times so the shape matches Q,
then run standard attention. This is done in-memory (no extra parameters).

Weight shapes for Llama 3.2-3B
--------------------------------
  wq: (n_heads_q  * head_dim, hidden) = (3072, 3072)
  wk: (n_heads_kv * head_dim, hidden) = (1024, 3072)   ← 3× smaller than wq
  wv: (n_heads_kv * head_dim, hidden) = (1024, 3072)
  wo: (hidden, n_heads_q * head_dim)  = (3072, 3072)

Forward pass shape flow
------------------------
  x                          (B, T, hidden)
  → q = x @ wq.T             (B, T, n_heads_q  * head_dim)
  → reshape                  (B, T, n_heads_q,  head_dim)
  → apply_rope               (B, T, n_heads_q,  head_dim)
  → transpose                (B, n_heads_q, T,  head_dim)   ← sdpa expects this

  → k,v same but n_heads_kv  (B, n_heads_kv, T, head_dim)
  → repeat kv heads          (B, n_heads_q,  T, head_dim)   ← now matches q

  → scaled_dot_product_attention → (B, n_heads_q, T, head_dim)
  → transpose + reshape      (B, T, n_heads_q * head_dim)
  → out = x @ wo.T           (B, T, hidden)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.rope import RopeFrequencies, apply_rope

# Triton kernel dispatch (Section 14d).
#   Controlled by the USE_TRITON env var (set in .env or shell).
#   Defaults to True — the Triton FlashAttention kernel is the production path.
#   Set USE_TRITON=false to force PyTorch's fused SDPA (useful for parity checks).
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        rope_freqs: RopeFrequencies,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_heads_q  = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim     = head_dim
        self.num_kv_groups = num_heads_q // num_heads_kv
        self.rope_freqs   = rope_freqs
        # Needed so each block's attention writes into the right slot of the
        # shared KVCache pool. Set at construction time by LlamaModel.
        self.layer_idx    = layer_idx

        # Projection weights — no bias in Llama
        self.wq = nn.Parameter(torch.empty(num_heads_q  * head_dim, hidden_size))
        self.wk = nn.Parameter(torch.empty(num_heads_kv * head_dim, hidden_size))
        self.wv = nn.Parameter(torch.empty(num_heads_kv * head_dim, hidden_size))
        self.wo = nn.Parameter(torch.empty(hidden_size, num_heads_q * head_dim))

    def load_weights(self, wq, wk, wv, wo):
        with torch.no_grad():
            self.wq.copy_(wq)
            self.wk.copy_(wk)
            self.wv.copy_(wv)
            self.wo.copy_(wo)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache=None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, hidden_size)
            start_pos: position offset — 0 during prefill, >0 during decode
            kv_cache:  optional KVCache object (Section 11) — None for now

        Returns:
            (B, T, hidden_size)
        """
        B, T, _ = x.shape

        # --- 1. Project to Q, K, V ---
        q = F.linear(x, self.wq)  # (B, T, n_heads_q  * head_dim)
        k = F.linear(x, self.wk)  # (B, T, n_heads_kv * head_dim)
        v = F.linear(x, self.wv)  # (B, T, n_heads_kv * head_dim)

        # --- 2. Reshape into heads ---
        q = q.view(B, T, self.num_heads_q,  self.head_dim)
        k = k.view(B, T, self.num_heads_kv, self.head_dim)
        v = v.view(B, T, self.num_heads_kv, self.head_dim)

        # --- 3. Apply RoPE to Q and K ---
        cos, sin = self.rope_freqs.get(seq_len=T, start_pos=start_pos)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # --- 4. Transpose to (B, n_heads, T, head_dim) for SDPA ---
        q = q.transpose(1, 2)  # (B, n_heads_q,  T, head_dim)
        k = k.transpose(1, 2)  # (B, n_heads_kv, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_heads_kv, T, head_dim)

        # --- 5. KV cache update (Section 11) ---
        # During prefill (start_pos=0): writes [0, T), returns [0, T) → standard self-attention.
        # During decode (start_pos>0, T=1): writes [pos, pos+1), returns [0, pos+1)
        # → the new query attends to the full cached prefix + itself.
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, start_pos, k, v)

        # --- 6. Attention ---
        # Triton FlashAttention handles GQA internally (maps each Q head to its
        # KV head), so we pass the un-repeated K/V. The PyTorch SDPA path needs
        # K/V repeated up to the Q head count first.
        #
        # Causal masking: prefill (T>1, Tq==Tk) is lower-triangular; decode
        # (T==1) attends to the whole cached prefix. The flash kernel derives
        # the per-row mask from a Tk-Tq offset, so we hand it the same is_causal
        # flag SDPA uses.
        causal = (T > 1)

        if USE_TRITON and q.is_cuda:
            from kernels.attention_kernel import attention_flash_triton
            out = attention_flash_triton(q, k, v, causal=causal)
        else:
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
            )
        # out: (B, n_heads_q, T, head_dim)

        # --- 8. Merge heads and project output ---
        out = out.transpose(1, 2).contiguous()       # (B, T, n_heads_q, head_dim)
        out = out.view(B, T, self.num_heads_q * self.head_dim)  # (B, T, hidden)
        out = F.linear(out, self.wo)                 # (B, T, hidden)

        return out
