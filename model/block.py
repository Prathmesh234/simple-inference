"""
Transformer Block — one layer of the Llama model.

Architecture recap
------------------
Each of the 28 Llama layers follows this structure:

    x → RMSNorm → Attention → + (residual) → x'
    x' → RMSNorm → MLP → + (residual) → output

This is called "pre-norm" because the norm happens *before* the op,
not after. Why?
  - Post-norm (norm after residual): gradients can explode as depth increases
  - Pre-norm: each op sees normalized input → more stable training

The residual connection is critical:
  - Without it, gradients vanish through 28 layers
  - With it, there's always a "straight path" for gradients to flow backward

Weight manifest for one block (layer 0 as example)
---------------------------------------------------
From HuggingFace checkpoint, each layer has 9 weight tensors:

  layers.0.input_layernorm.weight            (3072,)         ← attn_norm
  layers.0.self_attn.q_proj.weight           (3072, 3072)    ← wq
  layers.0.self_attn.k_proj.weight           (1024, 3072)    ← wk (GQA: 3× smaller)
  layers.0.self_attn.v_proj.weight           (1024, 3072)    ← wv
  layers.0.self_attn.o_proj.weight           (3072, 3072)    ← wo
  layers.0.post_attention_layernorm.weight   (3072,)         ← mlp_norm
  layers.0.mlp.gate_proj.weight              (8192, 3072)    ← w_gate
  layers.0.mlp.up_proj.weight                (8192, 3072)    ← w_up
  layers.0.mlp.down_proj.weight              (3072, 8192)    ← w_down

Total per layer: 2 RMSNorms + 4 attention matrices + 3 MLP matrices.

Shape flow for one forward pass
--------------------------------
  x                      (B, T, 3072)  ← from previous layer or embedding
  → attn_norm            (B, T, 3072)  ← normalize
  → attention            (B, T, 3072)  ← Q/K/V projection, rope, sdpa, output proj
  → + x (residual)       (B, T, 3072)
  → mlp_norm             (B, T, 3072)  ← normalize again
  → mlp                  (B, T, 3072)  ← gate/up expand to 8192, down to 3072
  → + x' (residual)      (B, T, 3072)  ← output
"""

import torch
import torch.nn as nn

from ops.rmsnorm import RMSNorm
from ops.attention import GroupedQueryAttention
from ops.mlp import SwiGLUMLP
from ops.rope import RopeFrequencies


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        rope_freqs: RopeFrequencies,
        norm_eps: float = 1e-5,
    ):
        """
        Args:
            hidden_size:       dimension of residual stream (3072)
            intermediate_size: expanded dimension in MLP (8192)
            num_heads_q:       number of query heads (24)
            num_heads_kv:      number of key/value heads (8, GQA)
            head_dim:          dimension per head (128)
            rope_freqs:        precomputed RoPE cos/sin tables
            norm_eps:          epsilon for RMSNorm (1e-5 or 1e-6)
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-attention norm
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)

        # Attention
        self.attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            rope_freqs=rope_freqs,
        )

        # Pre-MLP norm
        self.mlp_norm = RMSNorm(hidden_size, eps=norm_eps)

        # MLP
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def load_weights(
        self,
        attn_norm_weight: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        mlp_norm_weight: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        """Load all 9 weights for this block from the checkpoint."""
        self.attn_norm.load_weight(attn_norm_weight)
        self.attn.load_weights(wq, wk, wv, wo)
        self.mlp_norm.load_weight(mlp_norm_weight)
        self.mlp.load_weights(w_gate, w_up, w_down)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache=None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, hidden_size)
            start_pos: position offset for RoPE (0 during prefill, >0 during decode)
            kv_cache:  optional KVCache (Section 11) — None for now

        Returns:
            (B, T, hidden_size)
        """
        # --- 1. Attention block with residual ---
        # Pre-norm: normalize before passing to attention
        h = self.attn_norm(x)
        h = self.attn(h, start_pos=start_pos, kv_cache=kv_cache)
        x = x + h  # residual connection

        # --- 2. MLP block with residual ---
        # Pre-norm: normalize before passing to MLP
        h = self.mlp_norm(x)
        h = self.mlp(h)
        x = x + h  # residual connection

        return x
