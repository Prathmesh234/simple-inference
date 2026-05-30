"""
Shared helpers for the attention-kernel profiling deep-dive (Section 13.5).

Everything in profiling/ targets ONE kernel: the FlashAttention-2 Triton kernel
in kernels/attention_kernel.py. The goal is to answer, with hard data:

  - Where does decode-time go, op by op? (torch.profiler)
  - Is attention compute-bound or memory-bound at each shape? (roofline)
  - What does the GPU timeline actually look like? (chrome trace / nsys)
  - Why is the kernel slow *internally*? (ncu — register/cache/warp metrics)

WARMUP DISCIPLINE (the golden rule of profiling)
------------------------------------------------
The first few kernel calls carry one-time costs that inflate latency 10-100x:
  - Triton JIT compilation of each new (shape, dtype) specialization
  - The autotune sweep (16 configs) on the first call per (D, CAUSAL) key
  - cuBLAS/cuDNN plan selection for the SDPA reference
ALWAYS call warmup() before any measured/captured region. We discard >=10 iters.

ENV NOTE (how to launch these scripts)
--------------------------------------
System python has no torch/triton. Run everything through uv, e.g.:

    PATH="$HOME/.local/bin:$PATH" \
    XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
    UV_CACHE_DIR="$HOME/.cache/uv" \
    uv run python profiling/profile_attention_torch.py

(XDG_CONFIG_HOME/UV_CACHE_DIR are only a uv launcher workaround on this box —
~/.config is root-owned — they have nothing to do with the kernels.)
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

# Make `kernels`, `benchmarks`, etc. importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model / hardware constants (Llama-3.2-3B on RTX 6000 Ada)
# ---------------------------------------------------------------------------
DEVICE          = "cuda"
DTYPE           = torch.bfloat16
HEAD_DIM        = 128
N_Q             = 24          # query heads
N_KV            = 8           # KV heads  -> GQA group = 3
KV_GROUP        = N_Q // N_KV
PEAK_BW_GB_S    = 960.0       # HBM bandwidth ceiling
PEAK_TFLOPS_BF16 = 1457.0     # bf16 tensor-core ceiling
# Ridge point of the roofline: intensity (FLOP/byte) where the machine flips
# from memory-bound to compute-bound.  peak_flops / peak_bw.
RIDGE_FLOP_PER_BYTE = (PEAK_TFLOPS_BF16 * 1e12) / (PEAK_BW_GB_S * 1e9)


# ---------------------------------------------------------------------------
# Input builders: the two regimes that look completely different when profiled
# ---------------------------------------------------------------------------
def make_prefill_qkv(B: int, T: int):
    """Prefill: a full prompt of length T attends causally over itself.
    Large square score tiles -> expect COMPUTE-bound behaviour as T grows."""
    q = torch.randn(B, N_Q,  T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N_KV, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N_KV, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, v


def make_decode_qkv(B: int, Tk: int):
    """Decode: ONE new query token attends over a Tk-long KV cache.
    Tall-skinny work, tiny compute, lots of KV reads -> expect MEMORY-bound."""
    q = torch.randn(B, N_Q,  1,  HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N_KV, Tk, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N_KV, Tk, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, v


# ---------------------------------------------------------------------------
# Roofline math
# ---------------------------------------------------------------------------
def attn_flops(B: int, Tq: int, Tk: int, causal: bool) -> int:
    """FLOPs for QK^T + softmax-weighted P@V. MACs counted as 2 FLOPs.
    Causal prefill: ~half the score entries are masked out, so * 0.5."""
    full = 2 * 2 * B * N_Q * Tq * Tk * HEAD_DIM   # QK^T and P@V
    if causal and Tq > 1:
        full = int(full * 0.5)
    return full


def attn_io_bytes(B: int, Tq: int, Tk: int, dtype_bytes: int = 2) -> int:
    """HBM traffic for the flash kernel: read Q, read K, read V, write O once.
    K/V live in N_KV heads (GQA), Q/O in N_Q heads. Flash never writes the
    T x T score matrix to HBM — that is the whole point of the algorithm."""
    q_el = B * N_Q  * Tq * HEAD_DIM
    o_el = B * N_Q  * Tq * HEAD_DIM
    kv_el = 2 * B * N_KV * Tk * HEAD_DIM
    return (q_el + o_el + kv_el) * dtype_bytes


def arithmetic_intensity(B, Tq, Tk, causal) -> float:
    """FLOP per byte of HBM traffic. Compare to RIDGE_FLOP_PER_BYTE:
       below ridge -> memory-bound; above ridge -> compute-bound."""
    return attn_flops(B, Tq, Tk, causal) / attn_io_bytes(B, Tq, Tk)


def achieved_tflops(flops: int, latency_ms: float) -> float:
    return flops / (latency_ms * 1e-3) / 1e12


def achieved_bw_gb_s(bytes_moved: int, latency_ms: float) -> float:
    return bytes_moved / (latency_ms * 1e-3) / 1e9


def roofline_bound(intensity: float) -> str:
    """Classify a shape as compute- or memory-bound by its arithmetic intensity."""
    return "compute-bound" if intensity >= RIDGE_FLOP_PER_BYTE else "memory-bound"


# ---------------------------------------------------------------------------
# Warmup + timing
# ---------------------------------------------------------------------------
def warmup(fn, iters: int = 15):
    """Run fn() iters times and sync. Absorbs Triton JIT + autotune sweep +
    cuBLAS plan selection so the captured/timed region is steady-state."""
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# NVTX ranges — labels for the nsys/ncu timeline. No-op cost without a profiler,
# so it is always safe to leave these in.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def nvtx_range(label: str):
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def cuda_profiler_region():
    """Wrap the ONE region you want nsys/ncu to capture. Pair with
        nsys profile --capture-range=cudaProfilerApi ...
    so model-load and warmup are excluded from the trace."""
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()


def banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
