"""
Step 2 & 3 — scheduled capture, Chrome/Perfetto trace, and memory profiling.

Builds on Step 1 with three additions from the plan:

  1. schedule(wait, warmup, active) — the profiler itself skips the first
     iterations (initialization / JIT / autotune) and only keeps clean
     steady-state steps. This is warmup discipline enforced by the API.

  2. export_chrome_trace(...) — writes a .json timeline you open in
     chrome://tracing or https://ui.perfetto.dev . Visually:
       prefill = long matmul tiles back-to-back   (compute-bound)
       decode  = short kernels with gaps between   (memory-bound)
     That picture IS the compute- vs memory-bound distinction made concrete.

  3. profile_memory=True — attributes CUDA allocations to ops, so you can sort
     key_averages by self_cuda_memory_usage and see which op allocates most.

Traces are written to profiling/out/.

Run:
    PATH="$HOME/.local/bin:$PATH" XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
    UV_CACHE_DIR="$HOME/.cache/uv" \
    uv run python profiling/profile-kernels/torch-profiler/profile_attention_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# This script lives in profile-kernels/torch-profiler/; profile_utils.py is one
# directory up. Make it importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.profiler import (
    profile, schedule, record_function, ProfilerActivity,
)

from profile_utils import (
    make_prefill_qkv, make_decode_qkv, warmup, banner, OUT_DIR,
)
from kernels.attention_kernel import attention_prefill_triton

ACTIVITIES = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

# wait=1 (skip), warmup=1 (profiler warmup), active=3 (kept) -> 5 steps/cycle.
SCHED = schedule(wait=1, warmup=1, active=3, repeat=1)


def _trace(label: str, span: str, fn, trace_name: str):
    # Manual warmup too: absorbs the one-time Triton JIT + autotune sweep so the
    # profiler's own warmup window isn't polluted by a 1.7s compile.
    warmup(fn, iters=15)

    trace_path = OUT_DIR / trace_name
    with profile(
        activities=ACTIVITIES,
        schedule=SCHED,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for _ in range(5):                     # wait+warmup+active = 5
            with record_function(span):
                fn()
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    banner(f"{label}  (top ops by self GPU memory)")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=12))
    print(f"\n  Chrome/Perfetto trace written: {trace_path}")
    print(f"  Open at chrome://tracing or https://ui.perfetto.dev")


def main():
    assert torch.cuda.is_available(), "CUDA required for kernel profiling"

    qp, kp, vp = make_prefill_qkv(B=1, T=1024)
    _trace("PREFILL B=1 T=1024 causal", "flash_prefill",
           lambda: attention_prefill_triton(qp, kp, vp, causal=True, assume_contiguous=True),
           "attn_prefill_trace.json")

    qd, kd, vd = make_decode_qkv(B=1, Tk=2048)
    _trace("DECODE  B=1 Tq=1 Tk=2048", "flash_decode",
           lambda: attention_prefill_triton(qd, kd, vd, causal=False, assume_contiguous=True),
           "attn_decode_trace.json")

    print("\nTip: load both traces in Perfetto and compare the GPU stream row. "
          "Prefill shows dense back-to-back kernels; decode shows a short "
          "kernel surrounded by idle gaps (memory-latency stalls).")


if __name__ == "__main__":
    main()
