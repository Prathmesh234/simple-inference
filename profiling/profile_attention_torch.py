"""
Step 1 — torch.profiler key_averages on the FlashAttention Triton kernel.

The simplest profiler: no external tools needed (torch.profiler ships with
torch). Produces an op-by-op table sorted by GPU time, separately for the
PREFILL and DECODE regimes — because they have completely different op
distributions:

  - prefill: one big causal attention over T tokens (large square score tiles)
  - decode : one query token over a long KV cache (short, memory-bound kernel)

Read two columns:
  - cuda_time_total      : GPU time including child ops (where the cost sits)
  - self_cuda_time_total : GPU time excluding children (the op's *own* cost)
The gap between them tells you at which level (Python wrapper vs leaf kernel)
the time is actually spent.

`with_flops=True` lets the profiler estimate FLOPs for matmul-like ops; divide
by latency for an achieved-TFLOPS sanity check against the roofline.

Run:
    PATH="$HOME/.local/bin:$PATH" XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
    UV_CACHE_DIR="$HOME/.cache/uv" \
    uv run python profiling/profile_attention_torch.py
"""

from __future__ import annotations

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from profile_utils import (
    make_prefill_qkv, make_decode_qkv, warmup, banner,
)
from kernels.attention_kernel import attention_flash_triton

ACTIVITIES = [ProfilerActivity.CPU, ProfilerActivity.CUDA]


def _profile_call(label: str, fn, span_name: str, sort_key: str):
    warmup(fn, iters=15)                       # absorb JIT + autotune sweep
    with profile(activities=ACTIVITIES, record_shapes=True, with_flops=True) as prof:
        for _ in range(20):                    # steady-state captured iters
            with record_function(span_name):
                fn()
        torch.cuda.synchronize()

    banner(f"{label}  (sorted by {sort_key})")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))


def main():
    assert torch.cuda.is_available(), "CUDA required for kernel profiling"

    # ---- PREFILL: T x T causal attention ----
    qp, kp, vp = make_prefill_qkv(B=1, T=1024)
    prefill = lambda: attention_flash_triton(qp, kp, vp, causal=True)
    _profile_call("PREFILL  B=1 T=1024 causal",
                  prefill, "flash_prefill", "cuda_time_total")
    _profile_call("PREFILL  B=1 T=1024 causal",
                  prefill, "flash_prefill", "self_cuda_time_total")

    # ---- DECODE: 1 query token over a 2048-long KV cache ----
    qd, kd, vd = make_decode_qkv(B=1, Tk=2048)
    decode = lambda: attention_flash_triton(qd, kd, vd, causal=False)
    _profile_call("DECODE   B=1 Tq=1 Tk=2048",
                  decode, "flash_decode", "cuda_time_total")
    _profile_call("DECODE   B=1 Tq=1 Tk=2048",
                  decode, "flash_decode", "self_cuda_time_total")

    print("\nTip: compare PREFILL vs DECODE 'self_cuda_time_total'. Prefill is "
          "dominated by the dense score-tile matmuls; decode is a short, "
          "memory-bound kernel — the cost shape that motivates CUDA graphs "
          "and KV-cache layout work later.")


if __name__ == "__main__":
    main()
