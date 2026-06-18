"""
The shared workload both demos run — deliberately built to expose CPU launch
overhead, which is the ONLY thing CUDA graphs remove.

Design choices (why this shape):
  - SMALL tensors (256x256). Each kernel finishes on the GPU in ~1-3 us, so the
    ~5-10 us the CPU spends *launching* each kernel dominates wall-clock. This is
    the same regime as LLM decode: tiny per-step math, hundreds of launches.
  - MANY ops per step (N_OPS chained elementwise ops). One "step" issues N_OPS
    kernel launches. Stack enough of them and the CPU dispatch cost is the wall.
  - Pure elementwise, in-place where possible: keeps shapes/addresses static so
    the exact same sequence is capturable into a CUDA graph with no changes.

The function is written to run ENTIRELY on the GPU with NO host syncs (no .item(),
no .cpu(), no print) — a hard requirement for CUDA-graph capture.
"""

from __future__ import annotations

import torch

N_OPS = 50  # chained kernels per step → 50 launches/step


def workload(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    A chain of N_OPS tiny elementwise kernels.

    Each line below is a separate CUDA kernel launch. None of them need a host
    sync, and every tensor keeps a fixed shape/address, so this whole function
    is capturable verbatim into a CUDA graph.
    """
    for _ in range(N_OPS):
        x = x * a + b          # fused-multiply-add elementwise kernel(s)
        x = torch.tanh(x)      # elementwise kernel
    return x
