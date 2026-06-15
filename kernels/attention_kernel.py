"""
kernels/attention_kernel.py — Section 14d

A FlashAttention-2 forward kernel in Triton, written to teach the single most
important idea in GPU inference: **the memory wall**, and how FlashAttention
climbs over it.

The problem
-----------
Attention for one head is:

    S = Q @ K^T / sqrt(d)     # (T, T)   scores
    P = softmax(S)            # (T, T)   probabilities (row-wise)
    O = P @ V                 # (T, d)   output

The trap is the T×T score matrix `S`. For T = 8192 and bf16 that is
8192 * 8192 * 2 = 128 MB **per head**. Llama-3.2-3B has 24 heads × 28 layers,
so writing `S` out would push terabytes through HBM during a single prefill.
The textbook formulation is O(T²) in both compute *and* memory traffic.

`attention_prefill_triton` — FlashAttention-2 forward — never materialises `S`.
It walks the keys/values in tiles, keeping a running max `m`, running
denominator `l`, and running output `acc` in registers (the *online softmax*
trick). HBM traffic drops from O(T²) to O(T·d): you read Q, K, V once and write
O once. This is what vLLM / SGLang / PyTorch SDPA actually run.

Online softmax (the heart of FlashAttention)
--------------------------------------------
Softmax needs the row max for numerical stability, which normally means seeing
the whole row before you can normalise. Online softmax updates the result as
each new tile of scores arrives:

    m_new = max(m_old, max(tile))                      # new running max
    alpha = exp(m_old - m_new)                          # rescale factor for old state
    l     = l * alpha + sum(exp(tile - m_new))          # fix the denominator
    acc   = acc * alpha + exp(tile - m_new) @ V_tile    # fix the numerator

After the last tile, `O = acc / l`. Mathematically identical to a full softmax,
but it only ever holds one tile of scores at a time — so the T×T matrix never
exists.

GQA
---
Llama uses Grouped-Query Attention: 24 query heads share 8 KV heads
(KV_GROUP = 3). Instead of `repeat_interleave`-ing K/V in HBM (the PyTorch path
in ops/attention.py does this), the kernel just maps query head `h` to KV head
`h // KV_GROUP` when it indexes K and V. No KV duplication, less HBM traffic.

Causal masking with a KV offset
--------------------------------
Query row `i` sits at absolute position `(Tk - Tq) + i`. During prefill
Tk == Tq so the offset is 0 (standard lower-triangular mask). During decode the
single query (Tq=1) sits at position Tk-1 and must attend to the entire cached
prefix — the offset handles both cases with one code path. This matches the
semantics of `F.scaled_dot_product_attention(is_causal=...)` used by the
PyTorch reference, including chunked-prefill (Tq < Tk) shapes.
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



# ===========================================================================
# PREFILL flash: variable-length FlashAttention-2, online softmax, never
# materialise T×T.
#
# NAMING NOTE — "prefill" describes the kernel's shape regime (arbitrary query
# length Tq, key length Tk), NOT the only phase it runs in. This SAME kernel is
# also used for EAGER (non-CUDA-graph) decode, where Tq=1: vary-length inputs are
# fine in eager mode. It is the `_flash_decode_fwd` kernel below — fixed-length,
# position-from-tensor — that is used ONLY inside a captured CUDA graph. So:
#   * prefill phase            -> _flash_prefill_fwd
#   * decode, USE_CUDA_GRAPHS=0 -> _flash_prefill_fwd   (same kernel, Tq=1)
#   * decode, USE_CUDA_GRAPHS=1 -> _flash_decode_fwd    (capturable variant)
# ===========================================================================

def _flash_prefill_configs():
    configs = []
    for BM in (64, 128):
        for BN in (32, 64):
            for w in (4, 8):
                for s in (2, 3):
                    configs.append(
                        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN},
                                      num_warps=w, num_stages=s)
                    )
    return configs


# Autotune key: ONLY (D, CAUSAL) — deliberately NOT Tk.
#   Tk (the key/cache length) grows by 1 every decode step. If Tk were in the
#   key, Triton would re-run the full config sweep on EVERY decode token (a
#   ~1.7s stall per token measured on RTX 6000 Ada), destroying TPOT. head_dim
#   and causal-ness are what actually determine the best block/warp config, so
#   we tune once per (D, CAUSAL) and reuse the result for all sequence lengths.
@conditional_autotune(configs=_flash_prefill_configs(), key=["D", "CAUSAL"])
@triton.jit
def _flash_prefill_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr, sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Hq, Tq, Tk, KV_GROUP,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    One program per (batch*head, M-block). Keeps the running softmax state
    (m_i, l_i) and the running output (acc) in registers, looping over K/V in
    BLOCK_N tiles. The T×T score matrix never touches HBM.

    Memory map (for one (batch b, query-head hq) — set by program_id(1))
    -------------------------------------------------------------------
    The launch grid is 2D: program_id(0) picks a BLOCK_M slab of queries,
    program_id(1) picks which (batch, head) we're on. GQA means head hq reads
    KV head hkv = hq // KV_GROUP — so 3 query heads share one K/V block.

      Q[b, hq]   (Tq, D)                K[b, hkv] (Tk, D)     V[b, hkv] (Tk, D)
     ┌──────────────────────┐         ┌──────────────┐      ┌──────────────┐
     │ Q row 0:  [q0..qD-1] │         │ K row 0      │      │ V row 0      │
     │   ...                │         │ K row 1      │      │ V row 1      │
     ├──────────────────────┤ ◄─┐     │   ...        │      │   ...        │
     │ Q row m  ┐           │   │     │ K row n      │      │ V row n      │
     │   ...    │ BLOCK_M   │   │     │   ...        │      │   ...        │
     │ Q row m' ┘ slab      │   │     │ K row Tk-1   │      │ V row Tk-1   │
     ├──────────────────────┤   │     └──────────────┘      └──────────────┘
     │   ...                │   │      ◄── D ──►             ◄── D ──►
     └──────────────────────┘   │
      ◄────── D ──────►         start_m = program_id(0) loads THIS BLOCK_M slab
                                of Q once, into registers (stays resident).

    Inner loop — stream K/V in BLOCK_N tiles, never materialise scores:
    -------------------------------------------------------------------
      for start_n in 0, BLOCK_N, 2*BLOCK_N, ... (up to `hi`, the causal bound):

          qk = Qslab @ Ktile^T          # (BLOCK_M, BLOCK_N)  ← lives in SRAM only
                                        #   never written to HBM
                 ┌───────── BLOCK_N ─────────┐
        BLOCK_M  │  s00  s01  s02 ... s0,N-1 │   one score tile;
                 │  s10  s11  ...            │   consumed immediately by the
                 │  ...                      │   online-softmax update, then
                 └───────────────────────────┘   discarded.

          ┌─ online softmax (per query row, kept in registers across tiles) ─┐
          │  m_new = max(m_i, rowmax(qk))      running max                   │
          │  p     = exp(qk - m_new)           tile probs                    │
          │  alpha = exp(m_i - m_new)          rescale the old partial sums  │
          │  l_i   = l_i*alpha + rowsum(p)     running denominator           │
          │  acc   = acc*alpha + p @ Vtile     running numerator  (BLOCK_M,D)│
          └──────────────────────────────────────────────────────────────────┘

    After the last tile:  O = acc / l_i   ── written out ONCE to O[b, hq].

      O[b, hq]   (Tq, D)
     ┌──────────────────────┐
     │   ...                │
     ├──────────────────────┤ ◄── program_id(0) writes its BLOCK_M slab here
     │ O row m  ┐ BLOCK_M   │     (rows ≥ Tq are masked off — Tq may not be a
     │ O row m' ┘ slab      │      multiple of BLOCK_M)
     ├──────────────────────┤
     │   ...                │
     └──────────────────────┘
      ◄────── D ──────►

    The win: the (BLOCK_M, BLOCK_N) score tile is the ONLY scores that ever
    exist, and it lives in SRAM. HBM traffic is read Q,K,V once + write O once
    = O(T·D), versus the textbook O(T²) that writes the whole (Tq, Tk) matrix.
    """
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)
    b  = off_bh // Hq
    hq = off_bh % Hq
    hkv = hq // KV_GROUP

    q_base = q_ptr + b * stride_qb + hq * stride_qh
    k_base = k_ptr + b * stride_kb + hkv * stride_kh
    v_base = v_ptr + b * stride_vb + hkv * stride_vh
    o_base = o_ptr + b * stride_ob + hq * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < Tq, other=0.0,
    )  # (BLOCK_M, D)

    q_pos = (Tk - Tq) + offs_m  # absolute position of each query row

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Causal blocks past the diagonal contribute nothing — skip them.
    if CAUSAL:
        hi = tl.minimum(Tk, (Tk - Tq) + (start_m + 1) * BLOCK_M)
    else:
        hi = Tk

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=offs_n[:, None] < Tk, other=0.0,
        )  # (BLOCK_N, D)

        qk = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)

        valid = offs_n[None, :] < Tk
        if CAUSAL:
            valid = valid & (q_pos[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        # --- online softmax update ---
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])                 # (BLOCK_M, BLOCK_N)
        alpha = tl.exp(m_i - m_new)                     # rescale old state
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < Tk, other=0.0,
        )  # (BLOCK_N, D)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # A fully-masked row (l_i == 0) would divide by zero; guard it.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
        mask=offs_m[:, None] < Tq,
    )


def attention_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float | None = None,
    assume_contiguous: bool = False,
    return_transposed: bool = False,
) -> torch.Tensor:
    """
    FlashAttention-2 forward in Triton. Drop-in for
    F.scaled_dot_product_attention over (B, H, T, D) layout with GQA support.

    Args:
        q: (B, Hq,  Tq, D)
        k: (B, Hkv, Tk, D)
        v: (B, Hkv, Tk, D)
        causal:   apply the causal mask (with a Tk-Tq query offset)
        sm_scale: 1/sqrt(D) if None
        assume_contiguous: skip the per-call `.contiguous()` on q/k/v. The
            kernel indexes with explicit strides, so it only requires the tensors
            to be contiguous in their last dimension (head_dim) for coalesced access.
            Hot paths that already guarantee contiguous inputs can pass True to
            drop the per-call CPU checks.
        return_transposed: if True, writes output directly to (B, Tq, Hq, D) shape
            and returns it (zero-copy transposition), avoiding subsequent .transpose().contiguous().
    Returns:
        If return_transposed is True: (B, Tq, Hq, D) contiguous tensor
        Else: (B, Hq, Tq, D) tensor
    """
    B, Hq, Tq, D = q.shape
    _, Hkv, Tk, _ = k.shape
    assert k.shape[-1] == D and v.shape[-1] == D
    assert D in (16, 32, 64, 128, 256), f"unsupported head_dim {D}"
    KV_GROUP = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    if not assume_contiguous:
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()
        if v.stride(-1) != 1:
            v = v.contiguous()

    grid = lambda meta: (triton.cdiv(Tq, meta["BLOCK_M"]), B * Hq)

    if return_transposed:
        out = torch.empty((B, Tq, Hq, D), dtype=q.dtype, device=q.device)
        _flash_prefill_fwd[grid](
            q, k, v, out, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(2), out.stride(1), out.stride(3),
            Hq, Tq, Tk, KV_GROUP,
            D=D, CAUSAL=causal,
        )
    else:
        out = torch.empty_like(q)
        _flash_prefill_fwd[grid](
            q, k, v, out, sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Hq, Tq, Tk, KV_GROUP,
            D=D, CAUSAL=causal,
        )
    return out


# ===========================================================================
# DECODE flash (CUDA-graph only): single-query (Tq=1), fixed-length, capturable.
#
# This variant is used ONLY for decode when USE_CUDA_GRAPHS=1. Plain eager
# decode reuses _flash_prefill_fwd above (Tq=1) — see its naming note.
# ===========================================================================
#
# Why a separate kernel from _flash_prefill_fwd?
# ----------------------------------------------
# _flash_prefill_fwd is written for variable-length, eager execution: it takes the
# key length `Tk` as a scalar launch arg, loops `range(0, Tk)`, and infers the
# query's absolute position from the tensor SHAPE (`q_pos = (Tk - Tq) + offs_m`).
# All of that changes every decode step (Tk grows by 1), so a single captured
# CUDA graph — which freezes launch args, grid, and loop trip counts — cannot
# replay it correctly.
#
# `_flash_decode_fwd` is the static-shape form required for graph capture:
#   * KV_LEN is a COMPILE-TIME constant (the full pre-allocated cache length),
#     so the loop trip count is fixed and identical on every replay.
#   * The current position is read from a TENSOR pointer (`pos_ptr`), not derived
#     from shape — so replay sees the new position by reading the updated buffer,
#     with no re-capture.
#   * It always reads the WHOLE fixed-length cache and masks keys at positions
#     beyond `cur_pos` to -inf (the causal/length mask), exactly like the SDPA
#     fallback it replaces.
# Tq is hard-wired to 1 (one query row per program), which is the decode case.

def _flash_decode_configs():
    configs = []
    for BN in (32, 64, 128):
        for w in (2, 4):
            configs.append(triton.Config({"BLOCK_N": BN}, num_warps=w, num_stages=2))
    return configs


# Autotune key: (D, KV_LEN). NOT the position — position is a runtime tensor, so
# the same compiled kernel serves every decode step (and stays capturable).
@conditional_autotune(configs=_flash_decode_configs(), key=["D", "KV_LEN"])
@triton.jit
def _flash_decode_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr, pos_ptr, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_od,
    Hq, KV_GROUP,
    D: tl.constexpr,
    KV_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    One program per (batch, query-head). Single query row (Tq=1).

    The cache holds keys/values for positions [0, KV_LEN); only [0, cur_pos] are
    valid this step (the new token's K/V was written at cur_pos before launch).
    Online softmax over the fixed KV_LEN with a length mask — never materialises
    a score row in HBM, same as the prefill kernel.
    """
    off_bh = tl.program_id(0)
    b  = off_bh // Hq
    hq = off_bh % Hq
    hkv = hq // KV_GROUP

    # Current decode position, read from a tensor so it varies across graph
    # replays without re-capture.
    cur = tl.load(pos_ptr).to(tl.int32)

    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + b * stride_qb + hq * stride_qh + offs_d * stride_qd).to(tl.float32)  # (D,)

    k_base = k_ptr + b * stride_kb + hkv * stride_kh
    v_base = v_ptr + b * stride_vb + hkv * stride_vh

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for start_n in range(0, KV_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid = offs_n <= cur  # keep [0, cur]; mask the rest (future + padding)

        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=valid[:, None], other=0.0,
        ).to(tl.float32)  # (BLOCK_N, D)

        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale  # (BLOCK_N,)
        qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)                 # (BLOCK_N,)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=valid[:, None], other=0.0,
        ).to(tl.float32)  # (BLOCK_N, D)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)  # (D,)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe

    tl.store(
        o_ptr + b * stride_ob + hq * stride_oh + offs_d * stride_od,
        acc.to(o_ptr.dtype.element_ty),
    )


def attention_decode_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cur_pos: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """
    Single-step decode attention (Tq=1) over the full fixed-length KV cache,
    written to be safe inside a captured CUDA graph.

    Args:
        q:       (B, Hq, 1, D)            — the single decode query per head
        k:       (B, Hkv, KV_LEN, D)      — the WHOLE pre-allocated key cache
        v:       (B, Hkv, KV_LEN, D)      — the WHOLE pre-allocated value cache
        cur_pos: (1,) long tensor         — absolute position of this token; keys
                                            at positions (0..cur_pos) are attended.
        sm_scale: 1/sqrt(D) if None.

    Returns:
        (B, Hq, 1, D) attention output.
    """
    B, Hq, Tq, D = q.shape
    _, Hkv, KV_LEN, _ = k.shape
    assert Tq == 1, "attention_decode_triton is decode-only (Tq must be 1)"
    assert k.shape[-1] == D and v.shape[-1] == D
    assert D in (16, 32, 64, 128, 256), f"unsupported head_dim {D}"
    KV_GROUP = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    out = torch.empty((B, Hq, 1, D), dtype=q.dtype, device=q.device)
    grid = (B * Hq,)
    _flash_decode_fwd[grid](
        q, k, v, out, cur_pos, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        Hq, KV_GROUP,
        D=D, KV_LEN=KV_LEN,
    )
    return out

