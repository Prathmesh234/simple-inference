"""
kernels/rmsnorm_kernel.py — Section 14a

Triton fused RMSNorm kernel.

Why this is faster than PyTorch
---------------------------------
The PyTorch version in ops/rmsnorm.py:
    x_f32  = x.float()                   # read x, write temp
    var    = x_f32.pow(2).mean(-1)        # read temp, write var (scalar)
    normed = x_f32 * rsqrt(var + eps)     # read temp + var, write normed
    out    = normed.to(dtype) * weight    # read normed + weight, write out

That's 4+ round-trips through GPU memory (DRAM) for what is logically one
read + one write. The hidden size is only 3072 — RMSNorm is HEAVILY
memory-bound, not compute-bound.

The Triton kernel does this in two passes over the row, both staying in
registers / SRAM:
  Pass 1: load x tile by tile → accumulate sum of squares → compute rrms
  Pass 2: load x + weight tile by tile → multiply by rrms → write output

Net memory traffic: read x once, read weight once, write output once.
That is the theoretical minimum for this operation.

Kernel design
-------------
- Launch one Triton program per row (one row = one token's hidden vector).
- BLOCK_SIZE is set to the next power-of-2 >= hidden_size (4096 for H=3072)
  so the whole row fits in a single tile and we avoid the looping path.
- Computations over the row happen in float32 (matches PyTorch reference).
- Output is written back in the input dtype (bfloat16 for Llama).
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl

USE_AUTOTUNE = os.environ.get("USE_AUTOTUNE", "true").lower() in ("1", "true", "yes", "on")

def conditional_autotune(configs, key):
    if not USE_AUTOTUNE:
        configs = [configs[0]]
    return triton.autotune(configs, key)

@conditional_autotune(
    configs=[
        triton.Config({}, num_warps=1,  num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=3),
        triton.Config({}, num_warps=4,  num_stages=4),
        triton.Config({}, num_warps=8,  num_stages=4),
    ],
    key=["N"],   # re-tune whenever hidden size changes
)
@triton.jit
def _rmsnorm_fwd(
    x_ptr,        # input  (n_rows, N)
    w_ptr,        # weight (N,)
    out_ptr,      # output (n_rows, N)
    stride_row,   # x_ptr stride between rows (== N for contiguous)
    N,            # hidden_size (3072)
    eps,
    BLOCK_SIZE: tl.constexpr,   # next_power_of_2(N), e.g. 4096
):
    """
    One Triton program per row.

    Each program:
      1. Loads the row in a single tile (with masking for N < BLOCK_SIZE)
      2. Computes sum-of-squares → rrms in float32
      3. Applies scale weight and writes output in original dtype
    """
    pid = tl.program_id(0)
    #we get the x (our input row) and then we get the w

    """
    Input Matrix (x_ptr)                           Weight Vector (w_ptr)
 ┌───────────────────────────────┐              ┌───────────────────┐
 │ Row 0:  [x0, x1, x2, ...]     │              │ [w0, w1, w2, ...] │
 ├───────────────────────────────┤              └───────────────────┘
 │ Row 1:  [x0, x1, x2, ...]     │
 ├───────────────────────────────┤
 │ Row 2:  [x0, x1, x2, ...]     │ ◄── Program pid=2 processes this row
 └───────────────────────────────┘
    """
    row = x_ptr + pid*stride_row
    weight = w_ptr
    ##now we basically get the output too
    output = out_ptr + pid*stride_row
    #3let us get the cols too
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    ##LET US load the x and the weight
    x = tl.load(row + cols, mask=mask).to(tl.float32)
    w = tl.load(weight + cols, mask=mask).to(tl.float32)
    # now let's do the mean
    var = tl.sum(x*x, axis=0)/N
    rms = tl.rsqrt(var + eps)
    ##now we get the output
    out = x*w*rms
    ##now we need to write the output
    tl.store(output + cols, out.to(x_ptr.dtype.element_ty), mask=mask)

   


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Drop-in replacement for the PyTorch RMSNorm forward.

    Args:
        x:      (..., hidden_size) — any leading dims are flattened to rows
        weight: (hidden_size,)
        eps:    small constant for numerical stability

    Returns:
        Tensor of same shape and dtype as x.
    """
    orig_shape = x.shape
    #B, T,H
    ## now that we have the shape we have to get the number of rows
    #each pid is going to handle each row
    ##Remember right now it is B,T, H which is basically useless for us
    # our kernel does not understand B, T, h shit it only wants a tensor rows and cols
    ## we will make each of row something like this B*T and each col will be the hidden state
    x_2d = x.reshape(-1, x.shape[-1])
    n_rows, N = x_2d.shape           # N = hidden_size, e.g. 3072
    x_stride  = x_2d.stride(0)

    ##now we need an output tensor — same flat shape, reshape back at the end
    out = torch.empty_like(x_2d)

    ## block size: smallest power of 2 ≥ N so the row fits in one tile
    BLOCK_SIZE = triton.next_power_of_2(N)   # 3072 → 4096

    ## grid = ONE program per row.  BLOCK_SIZE is the per-row tile width,
    ## not a row-partitioning, so cdiv was wrong.  Just (n_rows,).
    grid = (n_rows,)

    ## Pass tensors directly — Triton's JIT pulls .data_ptr() internally.
    _rmsnorm_fwd[grid](
        x_2d,           # input  (n_rows, N)
        weight,         # weight (N,)
        out,            # output (n_rows, N)
        x_stride,       # stride between rows
        N,              # hidden_size
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)

