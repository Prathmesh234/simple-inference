"""
SwiGLU MLP — the feed-forward block inside every Llama transformer layer.

Why SwiGLU instead of a plain two-layer MLP?
----------------------------------------------
A standard two-layer MLP is:
    y = activation(x @ W_up.T) @ W_down.T

SwiGLU replaces that with a gated variant using THREE weight matrices:
    gate = silu(x @ W_gate.T)      # values in (0, 1) after sigmoid-weighted linear
    up   = x @ W_up.T              # unconstrained projection
    y    = (gate * up) @ W_down.T  # gate selects which up-projected values pass through

The intuition: the gate learns to suppress irrelevant features element-wise.
This "soft selection" is empirically stronger than plain GELU (no gating) and
comes at the cost of one extra matrix (W_gate) — a worthwhile trade-off for
model quality.

SiLU (Sigmoid Linear Unit):
    silu(x) = x * sigmoid(x)
    - Smooth, non-monotonic activation
    - Unlike ReLU it doesn't hard-zero negative values — gentler gradient flow
    - Used here because the sigmoid part forms the "gate"

Expand-then-contract pattern
------------------------------
    hidden_size      = 3072
    intermediate_size = 8192

    x        : (B, T, 3072)   ← input from residual stream
    gate, up : (B, T, 8192)   ← expand ~2.67×
    y        : (B, T, 3072)   ← contract back down

The expansion lets the model learn richer feature interactions before
contracting back to the residual stream dimension.

Weight shapes for Llama 3.2-3B
---------------------------------
    W_gate : (intermediate_size, hidden_size) = (8192, 3072)
    W_up   : (intermediate_size, hidden_size) = (8192, 3072)
    W_down : (hidden_size, intermediate_size) = (3072, 8192)

No biases — consistent with the rest of the Llama architecture.

MLP is usually the most compute-heavy op per layer (bigger matrices than attn)
and is COMPUTE-bound at prefill and MEMORY-bound at decode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Args:
            hidden_size       : width of the residual stream (3072)
            intermediate_size : width of the expanded hidden dim (8192)
        """
        super().__init__()
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size

        # No bias — Llama uses bias=False for all linear layers
        self.w_gate = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.w_up   = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.w_down = nn.Parameter(torch.empty(hidden_size, intermediate_size))

    def load_weights(
        self,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        """Copy checkpoint tensors into parameters in-place."""
        with torch.no_grad():
            self.w_gate.copy_(w_gate)
            self.w_up.copy_(w_up)
            self.w_down.copy_(w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, hidden_size)

        Returns:
            (B, T, hidden_size)
        """
        # --- 1. Gated projection (expand) ---
        # silu(x W_gate^T) — values in approximately (-0.3, ∞) with soft gating
        gate = F.silu(F.linear(x, self.w_gate))  # (B, T, intermediate_size)

        # --- 2. Up projection (expand, no activation) ---
        up = F.linear(x, self.w_up)              # (B, T, intermediate_size)

        # --- 3. Element-wise gate * up, then contract ---
        # gate selects which up-projected features to pass forward
        return F.linear(gate * up, self.w_down)  # (B, T, hidden_size)
