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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton kernel dispatch (Section 14b — SwiGLU fusion).
#   Same env-var toggle as ops/rmsnorm.py. Controls whether the silu(gate)*up
#   elementwise step is run via the fused Triton kernel or two separate
#   PyTorch ops (silu followed by mul).
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Args:
            hidden_size       : width of the residual stream (3072)
            intermediate_size : width of the expanded hidden dim (8192)

        Weight layout (Section 14b Level-2 fusion)
        ------------------------------------------
        W_gate and W_up share the same input x and produce parallel outputs of
        shape (B, T, I). We store them stacked as a single (2*I, H) parameter
        so the forward pass does ONE matmul instead of two:

            combined = x @ W_gate_up.T     # (B, T, 2I)
            gate, up = combined.chunk(2, dim=-1)

        This is the same trick vLLM calls MergedColumnParallelLinear.
        One larger GEMM beats two smaller GEMMs because of better tile
        utilization and a single set of kernel-launch / setup overheads.
        """
        super().__init__()
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size

        # No bias — Llama uses bias=False for all linear layers
        # Combined gate+up: rows [0:I) are gate, rows [I:2I) are up.
        self.w_gate_up = nn.Parameter(torch.empty(2 * intermediate_size, hidden_size))
        self.w_down    = nn.Parameter(torch.empty(hidden_size, intermediate_size))

    def load_weights(
        self,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        """
        Copy checkpoint tensors into parameters in-place.

        We keep the (w_gate, w_up, w_down) external API so the loader code
        doesn't change. The concat happens here, once at load time.
        """
        with torch.no_grad():
            # Stack gate on top of up: cat along the output-feature dim.
            self.w_gate_up.copy_(torch.cat([w_gate, w_up], dim=0))
            self.w_down.copy_(w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, hidden_size)

        Returns:
            (B, T, hidden_size)
        """
        # --- 1. Fused gate + up projection (Level-2 fusion) ---
        # ONE matmul produces both halves; split before the activation.
        combined = F.linear(x, self.w_gate_up)        # (B, T, 2*intermediate_size)
        gate, up = combined.chunk(2, dim=-1)          # each (B, T, intermediate_size)

        # --- 2. Fused silu(gate) * up (Section 14b — Level-1 fusion) ---
        if USE_TRITON and gate.is_cuda:
            from kernels.swiglu_kernel import swiglu_triton
            fused = swiglu_triton(gate, up)
        else:
            fused = F.silu(gate) * up

        # --- 3. Down projection (contract) ---
        return F.linear(fused, self.w_down)
