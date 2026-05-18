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

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep input dtype for the output but compute variance in float32
        # to avoid overflow with bfloat16 (bfloat16 max ~3.4e38, x² can explode)
        input_dtype = x.dtype
        x_f32 = x.float()

        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + self.eps)

        # Cast back and apply learned scale
        return (x_normed.to(input_dtype)) * self.weight

    def load_weight(self, weight: torch.Tensor):
        """Copy a tensor from the checkpoint into this module's parameter."""
        with torch.no_grad():
            self.weight.copy_(weight)
