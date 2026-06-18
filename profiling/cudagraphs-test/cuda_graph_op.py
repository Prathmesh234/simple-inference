"""
CUDA-GRAPH version — capture the workload ONCE, replay it per step.

This is the same workload() as the eager demo, but instead of letting Python
re-dispatch all N_OPS kernels every step, we record the kernel sequence into a
torch.cuda.CUDAGraph one time and then call graph.replay() each step. Replay
re-issues the whole recorded sequence with a SINGLE driver call — the per-op CPU
dispatch cost vanishes.

────────────────────────────────────────────────────────────────────────────
HOW THIS FILE BUILDS THE GRAPH (the 4 rules of CUDA graphs)
────────────────────────────────────────────────────────────────────────────
A captured graph freezes kernel args, grid dims, and — crucially — the *pointer
values* of every tensor. It does NOT freeze tensor CONTENTS. That single fact
drives the entire design:

  RULE 1 — STATIC INPUT BUFFERS.
    The graph records "read from address &x_static". So inputs must live in
    tensors allocated ONCE, and new data is written into them IN PLACE with
    copy_() before each replay. We never reassign x_static = something_new;
    that would change the address and the graph would read stale memory.

  RULE 2 — STATIC OUTPUT BUFFER.
    The last recorded kernel writes to a fixed address. We keep the handle
    (static_out) and read results out of it after replay. It is overwritten
    every replay, so consume it before the next one.

  RULE 3 — WARM UP ON A SIDE STREAM BEFORE CAPTURE.
    Any lazy init (cuBLAS handles, allocator growth, autotuning) does a host
    sync, which is ILLEGAL during capture and would abort it. We run the
    workload a few times on a separate stream first so all that finishes, then
    capture sees only clean launches.

  RULE 4 — NO HOST SYNC INSIDE THE CAPTURED REGION.
    workload() is pure GPU elementwise math: no .item(), .cpu(), or print().
    (In the LLM, this is why sampling stays OUTSIDE the graph.)

Capture vs replay at the driver level
-------------------------------------
  capture:  cudaStreamBeginCapture → every launch is RECORDED as a graph node
            (kernel + grid + frozen pointer args + dependency edges), not run →
            cudaStreamEndCapture → cudaGraphInstantiate compiles the DAG into a
            replayable executable (cudaGraphExec_t).
  replay:   cudaGraphLaunch — the driver walks the prebuilt node list and
            submits all N_OPS kernels with ~one CPU call. No per-op descriptor
            rebuild, no per-op CPU→driver crossing.

Run:
    uv run python profiling/cudagraphs-test/cuda_graph_op.py
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

    # ── RULE 1: allocate the STATIC input buffers ONCE. The graph will bake in
    #    THESE addresses; we feed new data by copy_()-ing into them, never by
    #    rebinding the names.
    x_static, a_static, b_static = make_inputs()

    # ── RULE 3: warm up on a side stream so any lazy init / host sync happens
    #    BEFORE capture (a sync during capture aborts it).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(WARMUP):
            workload(x_static, a_static, b_static)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # ── CAPTURE: record one pass of the workload into the graph. Nothing runs
    #    here — each launch becomes a node in the graph's DAG. RULE 2: we keep
    #    the returned tensor as the static output buffer.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = workload(x_static, a_static, b_static)
    torch.cuda.synchronize()

    # ── REPLAY helper. Per step we (optionally) refresh the static inputs in
    #    place, then fire the whole recorded sequence with ONE call.
    def step(new_x=None, new_a=None, new_b=None):
        if new_x is not None:
            ##this is exactly what is happening in our kv cache  - we are using copy_() in order to add the key and valye outputs to the buffer
            x_static.copy_(new_x)      # RULE 1: in-place, address unchanged
        if new_a is not None:
            a_static.copy_(new_a)
        if new_b is not None:
            b_static.copy_(new_b)
        graph.replay()                  # the whole N_OPS chain, one driver call
        return static_out               # RULE 2: read before next replay

    # ── timed loop (replay only; inputs already in the static buffers) ─────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_step_us = elapsed_ms * 1000.0 / STEPS

    # ── profiler trace: notice there is now essentially ONE launch per step
    #    (cudaGraphLaunch) instead of N_OPS, and Self CPU collapses.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(20):
            step()
        torch.cuda.synchronize()

    OUT.mkdir(exist_ok=True)
    trace_path = OUT / "cuda_graph_trace.json"
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=12)

    print("=" * 70)
    print(f"  CUDA GRAPH  ({N_OPS} ops/step captured, {STEPS} steps, tensor {DIM}x{DIM})")
    print("=" * 70)
    print(f"  total           : {elapsed_ms:8.2f} ms")
    print(f"  per step        : {per_step_us:8.2f} us   (= one graph.replay())")
    print(f"  launches issued : {N_OPS} kernels recorded ONCE, replayed as 1 call/step")
    print(f"  trace           : {trace_path}")
    print("\n  --- torch.profiler (sorted by CPU total) ---")
    print(table)


if __name__ == "__main__":
    main()
