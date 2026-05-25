"""
kernels/rope_kernel.py — Section 14c

Triton fused RoPE kernel.

What we're fusing
-----------------
The PyTorch version in ops/rope.py does, for each of Q and K:
    rotated = rotate_half(x)         # read x, write rotated (with -x slice copy)
    out     = x * cos                # read x + cos, write tmp1
    out    += rotated * sin          # read tmp1 + rotated + sin, write out

That's ~5 reads + 3 writes of (B, T, n_heads, head_dim) bf16 traffic per
tensor, plus an extra rotate_half() allocation. RoPE is firmly memory-bound:
6 FLOPs per element on a GPU with 1.56 FLOPs/byte break-even.

The fused kernel collapses all of that into:
    read x once, read cos+sin (one shared row across heads), write out once.

That is the theoretical minimum.

Kernel design
-------------
- Each program handles one (token, head) pair and computes the rotation
  for HALF = head_dim // 2 element-pairs at once.
- We exploit the symmetry of the rotation:
      out[i]        =  x[i]        * cos[i] - x[i + HALF] * sin[i]
      out[i + HALF] =  x[i + HALF] * cos[i] + x[i]        * sin[i]
  So one program loads both halves of x, plus the first half of cos/sin
  (cos/sin are stored duplicated — see RopeFrequencies — so cos[:HALF]
  equals cos[HALF:]). We avoid the entire `rotate_half` materialization.
- BLOCK_SIZE = next_power_of_2(HALF). For Llama (head_dim=128) HALF=64 so
  the whole half-row fits in one tile.
- Cos/sin are looked up by sequence position (token_pid % seq_len), which
  is the same for every batch element at the same position (cos is a
  per-position table broadcast over batch + heads).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=["n_heads", "HALF"],
)
@triton.jit
def _rope_fwd(
    x_ptr,        # input  (n_tokens, n_heads, head_dim) contiguous
    cos_ptr,      # cos    (seq_len, head_dim) contiguous, duplicated halves
    sin_ptr,      # sin    (seq_len, head_dim) contiguous, duplicated halves
    out_ptr,      # output (n_tokens, n_heads, head_dim) contiguous
    n_heads,
    seq_len,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # next_power_of_2(HALF)
):
    """
    One program per (token, head). Computes both halves of the rotated
    head in a single pass — no rotate_half materialization.

    Layout (n_tokens = B * T, contiguous along head_dim):

      Tensor x_ptr:
        token 0  ┌── head 0 ──┐┌── head 1 ──┐ ... ┌── head H-1 ──┐
        token 1  │   D dims   ││   D dims   │     │              │
        ...

      cos_ptr / sin_ptr (shared across batch + heads):
        position 0  [ first_half | first_half ]   ← halves are duplicated
        position 1  [ first_half | first_half ]
        ...
    """
    row_pid = tl.program_id(0)   # one row = one token (B*T flattened)
    col_pid = tl.program_id(1)   # one col = one head

    # cos/sin are indexed by sequence position, identical for every batch
    # element at the same position. row_pid runs over B*T in row-major
    # (token = b * T + t), so seq_pos = row_pid % seq_len recovers t.
    seq_pos = row_pid % seq_len

    row_off = (row_pid * n_heads + col_pid) * HEAD_DIM
    cs_off  = seq_pos * HEAD_DIM

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HALF

    # First and second half of the head vector
    x1 = tl.load(x_ptr + row_off + cols,        mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + row_off + HALF + cols, mask=mask, other=0.0).to(tl.float32)

    # Cos/sin — only the first half (duplicated layout)
    cos = tl.load(cos_ptr + cs_off + cols, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_off + cols, mask=mask, other=0.0).to(tl.float32)

    # Rotation. Equivalent to: out = x * cos + rotate_half(x) * sin
    #   rotate_half([x1, x2]) = [-x2, x1]
    #   ⇒ out_first  = x1 * cos + (-x2) * sin
    #     out_second = x2 * cos +   x1  * sin
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    out_dtype = x_ptr.dtype.element_ty
    tl.store(out_ptr + row_off + cols,        out1.to(out_dtype), mask=mask)
    tl.store(out_ptr + row_off + HALF + cols, out2.to(out_dtype), mask=mask)


def _apply_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a single tensor of shape (B, T, n_heads, head_dim)."""
    B, T, n_heads, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    seq_len = cos.shape[0]
    assert sin.shape[0] == seq_len and cos.shape[-1] == head_dim == sin.shape[-1]

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)

    n_tokens   = B * T
    HALF       = head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(HALF)

    grid = (n_tokens, n_heads)
    _rope_fwd[grid](
        x, cos, sin, out,
        n_heads,
        seq_len,
        HEAD_DIM=head_dim,
        HALF=HALF,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for ops.rope.apply_rope.

    Args:
        q:   (B, T, n_heads_q,  head_dim)
        k:   (B, T, n_heads_kv, head_dim)
        cos: (T, head_dim)   — duplicated halves (matches RopeFrequencies)
        sin: (T, head_dim)

    Returns:
        (q_rot, k_rot) — same shapes and dtypes as inputs.
    """
    q_rot = _apply_one(q, cos, sin)
    k_rot = _apply_one(k, cos, sin)
    return q_rot, k_rot
