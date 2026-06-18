"""
EAGER baseline — run the workload the normal PyTorch way, once per step.

What "eager" means at the driver level
--------------------------------------
Every line in workload() is dispatched the moment Python reaches it:

    Python op  →  ATen kernel pick  →  cuLaunchKernel (CPU→driver call)  →  queued

So one step = N_OPS launches, and EACH launch pays the fixed CPU dispatch cost
(~5-10 us: build the launch descriptor, cross into the driver, enqueue). With
tiny tensors the GPU finishes each kernel in ~1-3 us and then sits idle waiting
for the CPU to issue the next one. The CPU is the bottleneck; the GPU starves.

We time STEPS steps and also dump a torch.profiler trace so you can literally
count the launches and see Self CPU >> Self CUDA.

Run:
    uv run python profiling/cudagraphs-test/eager_op.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from workload import N_OPS, workload

OUT = Path(__file__).parent / "out"
STEPS = 1000
WARMUP = 50
DEVICE = "cuda"
DTYPE = torch.float32
DIM = int(os.environ.get("DIM", "256"))  # tensor is DIM x DIM


def make_inputs():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, generator=g)
    a = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, generator=g)
    b = torch.randn(DIM, DIM, device=DEVICE, dtype=DTYPE, generator=g)
    return x, a, b


@torch.no_grad()
def main():
    assert torch.cuda.is_available(), "CUDA required"
    x, a, b = make_inputs()

    # Warmup (let cuBLAS/caching allocator/first-launch JIT settle).
    for _ in range(WARMUP):
        workload(x, a, b)
    torch.cuda.synchronize()

    # ── timed loop ────────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        workload(x, a, b)
    torch.cuda.synchronize()          # one sync at the very end, not per step
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_step_us = elapsed_ms * 1000.0 / STEPS

    # ── profiler trace (a handful of steps is enough to see the launches) ──────
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(20):
            workload(x, a, b)
        torch.cuda.synchronize()

    OUT.mkdir(exist_ok=True)
    trace_path = OUT / "eager_trace.json"
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=12)

    print("=" * 70)
    print(f"  EAGER  ({N_OPS} ops/step, {STEPS} steps, tensor {DIM}x{DIM})")
    print("=" * 70)
    print(f"  total           : {elapsed_ms:8.2f} ms")
    print(f"  per step        : {per_step_us:8.2f} us")
    print(f"  per launch      : {per_step_us / N_OPS:8.3f} us   "
          f"(= per-op CPU dispatch cost)")
    print(f"  kernel launches : {N_OPS} per step  x  {STEPS} steps "
          f"= {N_OPS * STEPS} total")
    print(f"  trace           : {trace_path}")
    print("\n  --- torch.profiler (sorted by CPU total) ---")
    print(table)


if __name__ == "__main__":
    main()
