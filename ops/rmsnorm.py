"""
RMSNorm — Root Mean Square Layer Normalization.

Why RMSNorm instead of LayerNorm?
----------------------------------
LayerNorm subtracts the mean then divides by std:
    y = (x - mean(x)) / std(x) * weight + bias

RMSNorm skips the mean subtraction and the bias:
    y = x / rms(x) * weight
    rms(x) = sqrt(mean(x²) + eps)

Llama uses RMSNorm because:
  - Cheaper: one fewer pass over the data (no mean computation)
  - Empirically matches LayerNorm quality on language modelling
  - No bias parameter to store or load

Where it appears in the model:
  - Before every attention block  (input_layernorm)
  - Before every MLP block        (post_attention_layernorm)
  - After the final layer         (model.norm)
  That is 2×28 + 1 = 57 RMSNorm calls per forward pass.

Shape contract:
  input:  (batch, seq_len, hidden_size)   any batch/seq dims work
  weight: (hidden_size,)                  learned scale, loaded from checkpoint
  output: (batch, seq_len, hidden_size)   same shape as input
"""

import os
import torch
import torch.nn as nn

# Triton kernel dispatch (Section 14a).
#   Controlled by the USE_TRITON env var (set in .env or shell).
#   Defaults to True — Triton is the production path. Set USE_TRITON=false
#   to force the pure-PyTorch reference (useful for debugging / parity checks).
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


def _pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return x_normed.to(input_dtype) * weight


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_TRITON and x.is_cuda:
            from kernels.rmsnorm_kernel import rmsnorm_triton
            return rmsnorm_triton(x, self.weight, self.eps)
        return _pytorch_rmsnorm(x, self.weight, self.eps)

    def load_weight(self, weight: torch.Tensor):
        """Copy a tensor from the checkpoint into this module's parameter."""
        with torch.no_grad():
            self.weight.copy_(weight)
