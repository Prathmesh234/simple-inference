"""
kernels/swiglu_kernel.py — Section 14b

Triton fused SwiGLU activation kernel.

What we're fusing
-----------------
The PyTorch MLP forward (ops/mlp.py) does:
    gate_proj = x @ W_gate.T          # GEMM  (cuBLAS — already optimal)
    up_proj   = x @ W_up.T            # GEMM  (cuBLAS — already optimal)
    tmp       = silu(gate_proj)       # elementwise: READ gate, WRITE tmp
    fused     = tmp * up_proj         # elementwise: READ tmp + up, WRITE fused
    out       = fused @ W_down.T      # GEMM  (cuBLAS — already optimal)

The two middle steps are two separate kernel launches and TWO round-trips
through DRAM for an (B, T, 8192) tensor. That's the waste we kill here.

After fusion: one kernel reads gate_proj + up_proj, computes
    silu(g) * u = (g * sigmoid(g)) * u
all in registers, and writes the fused result exactly once.

We do NOT fuse the matmuls — cuBLAS already saturates the GEMM and writing
our own matmul kernel would lose to it. This is elementwise-only fusion.

Why this is memory-bound
------------------------
Per output element we do: 1 sigmoid + 2 multiplies = ~3 FLOPs.
Per output element we move: 2 reads + 1 write = 3×2 = 6 bytes (bf16).
That's 0.5 FLOPs/byte — orders of magnitude below the GPU's compute:bandwidth
ratio (~1500 TFLOPS / 960 GB/s = 1.56 FLOPs/byte for break-even). So this
kernel is firmly bandwidth-bound and the ceiling is "read 2 + write 1 at
peak BW", same logic as RMSNorm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def _swiglu_fwd(
    gate_ptr,        # silu input  — (n_rows, n_cols), row stride = gu_row_stride
    up_ptr,          # gate input  — same layout as gate
    out_ptr,         # output      — contiguous (n_rows, n_cols), row stride = n_cols
    n_rows,
    n_cols,
    gu_row_stride,   # row stride of BOTH gate and up (= 2*n_cols when sliced from combined)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Elementwise: out = silu(gate) * up   where silu(g) = g * sigmoid(g)

    Combined Input Storage (gate_ptr and up_ptr):
     ┌─────────────────────────────────────────────────────────────┐
     │ Row 0: [gate_0, ..., gate_N-1]  [up_0, ..., up_N-1]         │
     ├─────────────────────────────────────────────────────────────┤
     │ Row 1: [gate_0, ..., gate_N-1]  [up_0, ..., up_N-1]         │
     ├─────────────────────────────────────────────────────────────┤
     │ Row 2: [gate_0, ..., gate_N-1]  [up_0, ..., up_N-1]         │ ◄── Program pid=2 processes this row
     └─────────────────────────────────────────────────────────────┘
     ◄────────────────────── gu_row_stride ────────────────────────►
     ◄─────────── n_cols ────────────►◄──────────── n_cols ────────►
 
    Output Storage (out_ptr):
     ┌───────────────────────────────┐
     │ Row 0: [out_0, ..., out_N-1]  │
     ├───────────────────────────────┤
     │ Row 1: [out_0, ..., out_N-1]  │
     ├───────────────────────────────┤
     │ Row 2: [out_0, ..., out_N-1]  │ ◄── Program pid=2 writes to this row
     └───────────────────────────────┘
     ◄─────────── n_cols ────────────►
    """
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    g_ptr = gate_ptr + row_pid * gu_row_stride
    u_ptr = up_ptr + row_pid * gu_row_stride
    o_ptr = out_ptr + row_pid * n_cols

    # ────────────────────────────────────────────────────────────────────
    # Column offsets — VERY easy to get wrong. The naive version:
    #
    #     cols = tl.arange(0, BLOCK_SIZE)        # ❌ WRONG
    #
    # looks fine and is what autocomplete will suggest. But our launch grid
    # is 2D — (n_rows, cdiv(n_cols, BLOCK_SIZE)) — so for n_cols=8192 and
    # BLOCK_SIZE=512 we get 16 programs PER row, each meant to handle a
    # different 512-wide column tile.
    #
    # With `cols = tl.arange(0, BLOCK_SIZE)`, EVERY one of those 16 programs
    # computes `cols = [0..511]` and writes to out[row, 0..511]:
    #
    #       ┌────────┬────────┬────────┬─────...─────┬────────┐
    #       │  0..511│ 512..  │ 1024.. │             │ 7680.. │
    #       ├────────┼────────┼────────┼─────────────┼────────┤
    #  row: │   ✓✓✓✓ │  empty │  empty │    empty    │  empty │  ← 16 progs
    #       └────────┴────────┴────────┴─────────────┴────────┘  all wrote to [0..511]
    #
    # Result:
    #   • out[row, 0..511]    — written 16× with identical values (wasted work)
    #   • out[row, 512..8191] — never written, contains torch.empty garbage
    #
    # (Not technically a classical race condition — all 16 programs write
    #  identical bits — but the output is still incorrect on the tail.)
    #
    # THE FIX: include the column-tile id in the offset so each program
    # writes a DIFFERENT slice of the row.
    # ────────────────────────────────────────────────────────────────────
    cols = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load gate and up values (compute in float32 for numerical stability)
    gate = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(u_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute: out = silu(gate) * up = (gate * sigmoid(gate)) * up
    silu_gate = gate * tl.sigmoid(gate)
    output = silu_gate * up

    # Cast back to the input's original element type (e.g. bfloat16) and store
    tl.store(o_ptr + cols, output.to(gate_ptr.dtype.element_ty), mask=mask)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU activation: silu(gate) * up.

    Handles the NON-CONTIGUOUS gate/up views produced by
    `combined.chunk(2, dim=-1)` without forcing a .contiguous() copy.

    Args:
        gate:  output of the gate projection — shape (..., n_cols).
               May be non-contiguous along the last dim only as long as
               innermost stride == 1 (which `chunk` guarantees).
        up:    output of the up projection — same shape, dtype, and strides as gate.

    Returns:
        Contiguous tensor of the same shape and dtype as gate.
    """
    assert gate.shape == up.shape, "gate and up must have the same shape"
    assert gate.dtype == up.dtype, "gate and up must have the same dtype"
    assert gate.is_cuda and up.is_cuda, "swiglu_triton expects CUDA tensors"
    assert gate.stride() == up.stride(), "gate and up must have identical strides"
    assert gate.stride(-1) == 1, "innermost dim must be unit-stride"

    orig_shape = gate.shape
    n_cols     = orig_shape[-1]
    n_rows     = gate.numel() // n_cols

    # View as 2D without copying. Use as_strided to preserve the row stride
    # of the underlying storage (which is 2*n_cols when gate came from chunk).
    gu_row_stride = gate.stride(-2) if gate.dim() >= 2 else n_cols
    gate_2d = gate.reshape(-1, n_cols) if gate.is_contiguous() else \
              gate.as_strided((n_rows, n_cols), (gu_row_stride, 1))
    up_2d   = up.reshape(-1, n_cols) if up.is_contiguous() else \
              up.as_strided((n_rows, n_cols), (gu_row_stride, 1))

    out = torch.empty(n_rows, n_cols, device=gate.device, dtype=gate.dtype)
    # as you can see we are launching n_cols and n_rows so two seperate pids basically 
    ##one for th erows and one for the cols 
    ##we also need lambda META - BECAUSE META IS FOR THE AUTOTUNING PURPOSES 
    grid = lambda META: (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))
    _swiglu_fwd[grid](  
        gate_2d, up_2d, out,
        n_rows, n_cols,
        gu_row_stride,
    )

    return out.reshape(orig_shape)
