"""
Step 4 target — NVTX-annotated attention run for Nsight Systems / Compute.

This is the script you launch UNDER nsys or ncu (see profiling/run_nsys.sh and
profiling/run_ncu.sh). It:

  1. Warms up OUTSIDE the captured region (so JIT + autotune are excluded).
  2. Calls cudaProfilerStart()/Stop() around the region of interest, so
     `nsys profile --capture-range=cudaProfilerApi` captures ONLY that region
     instead of model-load + warmup (which would bury the signal).
  3. Wraps each logical phase in an NVTX range, so the nsys/ncu timeline shows
     "prefill_T1024" / "decode_Tk2048" next to the raw Triton kernel names.

NVTX ranges and cudaProfilerStart are no-ops (negligible cost) when no profiler
is attached, so running this script bare is harmless and validates it works.

Bare run (sanity check, no profiler):
    uv run python profiling/profile_attention_nsys.py

Under nsys:
    bash profiling/run_nsys.sh
Under ncu (single-kernel deep dive):
    bash profiling/run_ncu.sh
"""

from __future__ import annotations

import torch

from profile_utils import (
    make_prefill_qkv, make_decode_qkv, warmup,
    nvtx_range, cuda_profiler_region,
)
from kernels.attention_kernel import attention_flash_triton

N_CAPTURED_STEPS = 10


def main():
    assert torch.cuda.is_available(), "CUDA required"

    qp, kp, vp = make_prefill_qkv(B=1, T=1024)
    qd, kd, vd = make_decode_qkv(B=1, Tk=2048)

    prefill = lambda: attention_flash_triton(qp, kp, vp, causal=True)
    decode  = lambda: attention_flash_triton(qd, kd, vd, causal=False)

    # Warm up BOTH shapes outside the capture window (JIT + autotune sweep).
    warmup(prefill, iters=15)
    warmup(decode, iters=15)

    # Only this region is captured when --capture-range=cudaProfilerApi is used.
    with cuda_profiler_region():
        for i in range(N_CAPTURED_STEPS):
            with nvtx_range(f"prefill_T1024_step{i}"):
                prefill()
            with nvtx_range(f"decode_Tk2048_step{i}"):
                decode()
        torch.cuda.synchronize()

    print(f"Captured {N_CAPTURED_STEPS} prefill + {N_CAPTURED_STEPS} decode "
          f"steps inside the cudaProfilerApi range.")


if __name__ == "__main__":
    main()
