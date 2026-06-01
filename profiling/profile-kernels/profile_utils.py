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
    uv run python profiling/profile_kernel_torch.py attention

(XDG_CONFIG_HOME/UV_CACHE_DIR are only a uv launcher workaround on this box —
~/.config is root-owned — they have nothing to do with the kernels.)
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

# Make `kernels`, `benchmarks`, etc. importable when run as a script.
# This file lives at <repo>/profiling/profile-kernels/, so the repo root is
# three parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "torch-profiler" / "out"
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
HIDDEN          = 3072        # residual-stream width (RMSNorm / RoPE inputs)
INTERMEDIATE    = 8192        # MLP expanded width (SwiGLU gate/up width)
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
# Input builders for the elementwise / per-row kernels (rmsnorm, rope, swiglu).
# All three are MEMORY-bound: trivial arithmetic per element, so latency is set
# by HBM traffic. Prefill (T tokens) vs decode (T=1) only changes the row count.
# ---------------------------------------------------------------------------
def make_rmsnorm_inputs(B: int, T: int):
    """RMSNorm over the residual stream. x:(B,T,HIDDEN), weight:(HIDDEN,).
    One Triton program per (B*T) row, each normalising HIDDEN elements."""
    x = torch.randn(B, T, HIDDEN, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(HIDDEN, device=DEVICE, dtype=DTYPE)
    return x, weight


def make_rope_inputs(B: int, T: int):
    """RoPE inputs in (B, T, n_heads, head_dim) layout (pre-transpose), plus
    duplicated cos/sin of shape (T, head_dim) as the HF/Llama path produces."""
    q = torch.randn(B, T, N_Q,  HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, T, N_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    cos = torch.randn(T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    sin = torch.randn(T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, cos, sin


def make_swiglu_inputs(B: int, T: int):
    """SwiGLU gate/up. Built via chunk(2, dim=-1) of a (B,T,2*INTERMEDIATE)
    tensor so gate/up are the SAME non-contiguous views the MLP path feeds the
    kernel (innermost stride 1, row stride 2*INTERMEDIATE) — profiling the real
    no-copy fast path, not an idealised contiguous one."""
    combined = torch.randn(B, T, 2 * INTERMEDIATE, device=DEVICE, dtype=DTYPE)
    gate, up = combined.chunk(2, dim=-1)
    return gate, up


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


# ---------------------------------------------------------------------------
# Kernel registry — one place that knows how to build the prefill/decode inputs
# and the callable for every Triton kernel we profile. Add a kernel here once
# and BOTH the torch and nsys drivers below can profile it by name; no new
# per-kernel script needed.
#
# Each entry is a zero-arg builder returning a list of "regimes":
#     (regime_label, nvtx/record_function span name, no-arg callable)
# The callable closes over freshly-built CUDA inputs, so calling it repeatedly
# re-runs exactly one kernel launch. Kernel imports are lazy (inside the
# builder) so importing profile_utils on a CPU-only box stays cheap.
# ---------------------------------------------------------------------------
def _attention_regimes():
    from kernels.attention_kernel import attention_flash_triton
    qp, kp, vp = make_prefill_qkv(B=16, T=1024)
    qd, kd, vd = make_decode_qkv(B=16, Tk=2048)
    return [
        ("PREFILL B=1 T=1024 causal", "attn_prefill",
         lambda: attention_flash_triton(qp, kp, vp, causal=True, assume_contiguous=True)),
        ("DECODE  B=1 Tq=1 Tk=2048", "attn_decode",
         lambda: attention_flash_triton(qd, kd, vd, causal=False, assume_contiguous=True)),
    ]


def _rmsnorm_regimes():
    from kernels.rmsnorm_kernel import rmsnorm_triton
    xp, wp = make_rmsnorm_inputs(B=16, T=1024)
    xd, wd = make_rmsnorm_inputs(B=16, T=1)
    return [
        ("PREFILL B=1 T=1024 HIDDEN=3072", "rmsnorm_prefill",
         lambda: rmsnorm_triton(xp, wp)),
        ("DECODE  B=1 T=1 HIDDEN=3072", "rmsnorm_decode",
         lambda: rmsnorm_triton(xd, wd)),
    ]


def _rope_regimes():
    from kernels.rope_kernel import rope_triton
    qp, kp, cp, sp = make_rope_inputs(B=16, T=1024)
    qd, kd, cd, sd = make_rope_inputs(B=16, T=1)
    return [
        ("PREFILL B=1 T=1024", "rope_prefill",
         lambda: rope_triton(qp, kp, cp, sp)),
        ("DECODE  B=1 T=1", "rope_decode",
         lambda: rope_triton(qd, kd, cd, sd)),
    ]


def _swiglu_regimes():
    from kernels.swiglu_kernel import swiglu_triton
    gp, up = make_swiglu_inputs(B=16, T=1024)
    gd, ud = make_swiglu_inputs(B=16, T=1)
    return [
        ("PREFILL B=1 T=1024 INTERMEDIATE=8192", "swiglu_prefill",
         lambda: swiglu_triton(gp, up)),
        ("DECODE  B=1 T=1 INTERMEDIATE=8192", "swiglu_decode",
         lambda: swiglu_triton(gd, ud)),
    ]


# name -> regime builder. The keys are what you pass on the command line.
KERNELS = {
    "attention": _attention_regimes,
    "rmsnorm":   _rmsnorm_regimes,
    "rope":      _rope_regimes,
    "swiglu":    _swiglu_regimes,
}


def kernel_names() -> list[str]:
    return list(KERNELS)


def build_regimes(name: str):
    """Return the list of (label, span, fn) regimes for a registered kernel."""
    if name not in KERNELS:
        raise SystemExit(
            f"unknown kernel '{name}'. choose one of: {', '.join(kernel_names())}")
    return KERNELS[name]()


# ---------------------------------------------------------------------------
# Generic drivers — profile ANY registered kernel by name.
# ---------------------------------------------------------------------------
def run_torch_profile(name: str, iters: int = 20, row_limit: int = 15):
    """torch.profiler key_averages over every regime of `name`, printed and
    saved to profiling/out/profiler_<name>_1.txt. Each regime is sorted by both
    cuda_time_total and self_cuda_time_total (where the cost sits vs the op's
    own leaf-kernel cost)."""
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    assert torch.cuda.is_available(), "CUDA required for kernel profiling"
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    report_path = OUT_DIR / f"profiler_{name}_1.txt"
    lines: list[str] = []

    def emit(text: str):
        print(text)
        lines.append(text)

    for label, span, fn in build_regimes(name):
        warmup(fn, iters=15)                   # absorb JIT + autotune sweep
        for sort_key in ("cuda_time_total", "self_cuda_time_total"):
            warmup(fn, iters=5)
            with profile(activities=activities, record_shapes=True,
                         with_flops=True) as prof:
                for _ in range(iters):
                    with record_function(span):
                        fn()
                torch.cuda.synchronize()
            emit("\n" + "=" * 78)
            emit(f"  {name.upper()}  {label}  (sorted by {sort_key})")
            emit("=" * 78)
            emit(prof.key_averages().table(sort_by=sort_key, row_limit=row_limit))

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nProfiler report saved: {report_path}")
    return report_path


def run_nsys_capture(name: str, n_steps: int = 10):
    """NVTX-annotated capture region for nsys/ncu over every regime of `name`.
    Warms up OUTSIDE the cudaProfiler region so only steady-state launches are
    traced. Launch this under: nsys profile --capture-range=cudaProfilerApi."""
    import torch
    assert torch.cuda.is_available(), "CUDA required"

    regimes = build_regimes(name)
    for _, _, fn in regimes:                   # warm up every shape first
        warmup(fn, iters=15)

    with cuda_profiler_region():
        for i in range(n_steps):
            for _, span, fn in regimes:
                with nvtx_range(f"{span}_step{i}"):
                    fn()
        torch.cuda.synchronize()

    print(f"Captured {n_steps} steps/regime for '{name}' "
          f"({len(regimes)} regimes) inside the cudaProfilerApi range.")
