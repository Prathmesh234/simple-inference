"""
profile_kernel.py — Section 14f (kernel profiling)

Profile ONE Triton kernel under NVIDIA Nsight Systems and save a
`.nsys-rep` report you can download off the VM and open in the Nsight
Systems GUI on your local machine.

Why this exists
---------------
On a headless GPU VM you can't open the Nsight Systems UI. But `nsys`
writes a self-contained `.nsys-rep` file — download it (scp / the IDE's
download button) and open it locally with **File → Open**. You get the
full timeline (CUDA kernels, memcpy, launch gaps, NVTX ranges) without
the GPU ever being on your machine.

What the script does
--------------------
It wraps `nsys profile` around an in-process workload that:
  1. builds Llama 3.2-3B-shaped inputs on the GPU,
  2. warms up (triggers Triton autotuning so the one-time sweep does NOT
     pollute the trace),
  3. runs the kernel `--iters` times inside a named NVTX range so the hot
     loop is trivial to find on the timeline.

Usage (on the GPU VM):
    python benchmarks/profile_kernel.py --kernel rmsnorm
    python benchmarks/profile_kernel.py --kernel swiglu --iters 200
    python benchmarks/profile_kernel.py --kernel rope --seq-len 2048 --out rope_2k

Output: <out>.nsys-rep  (default out = profile_<kernel>)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

# repo root on the path so `kernels.*` imports work no matter the cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Llama 3.2-3B dims
H        = 3072    # hidden size
I        = 8192    # MLP intermediate size
N_Q      = 24      # query heads
N_KV     = 8       # kv heads (GQA)
HEAD_DIM = 128
EPS      = 1e-5


def _build_rmsnorm(T):
    import torch
    from kernels.rmsnorm_kernel import rmsnorm_triton
    x = torch.randn(1, T, H, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    return lambda: rmsnorm_triton(x, w, EPS)


def _build_swiglu(T):
    import torch
    from kernels.swiglu_kernel import swiglu_triton
    gate = torch.randn(1, T, I, device="cuda", dtype=torch.bfloat16)
    up   = torch.randn(1, T, I, device="cuda", dtype=torch.bfloat16)
    return lambda: swiglu_triton(gate, up)


def _build_rope(T):
    import torch
    from kernels.rope_kernel import rope_triton
    q   = torch.randn(1, T, N_Q,  HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k   = torch.randn(1, T, N_KV, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return lambda: rope_triton(q, k, cos, sin)


KERNELS = {
    "rmsnorm": _build_rmsnorm,
    "swiglu":  _build_swiglu,
    "rope":    _build_rope,
}


def run_workload(kernel: str, seq_len: int, iters: int):
    """The thing nsys actually profiles. Runs in a child process under nsys."""
    import torch

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available — profiling needs a GPU.")

    torch.manual_seed(0)
    call = KERNELS[kernel](seq_len)

    # Warmup: trigger Triton autotuning + JIT BEFORE the timed range so the
    # one-time config sweep doesn't show up as a giant spike in the trace.
    for _ in range(25):
        call()
    torch.cuda.synchronize()

    # Hot loop inside a named NVTX range → easy to spot on the timeline.
    torch.cuda.nvtx.range_push(f"{kernel}_x{iters}")
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def profile(kernel: str, seq_len: int, iters: int, out: str):
    """Re-launch this script under `nsys profile` to capture the workload."""
    if shutil.which("nsys") is None:
        sys.exit(
            "ERROR: `nsys` not found on PATH.\n"
            "Install NVIDIA Nsight Systems CLI on the GPU VM, or load its "
            "module, then re-run. (Download the .nsys-rep and open it in the "
            "Nsight Systems GUI on your local machine.)"
        )

    report = out if out.endswith(".nsys-rep") else f"{out}.nsys-rep"
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",        # CUDA kernels + our NVTX ranges
        "--force-overwrite=true",
        "--output", report,
        sys.executable, os.path.abspath(__file__),
        "--kernel", kernel,
        "--seq-len", str(seq_len),
        "--iters", str(iters),
        "--_workload",              # tells the child to just run the loop
    ]
    print(f"Profiling '{kernel}' (T={seq_len}, iters={iters}) ...")
    subprocess.run(cmd, check=True)
    print(f"\nSaved trace → {report}")
    print("Download it and open in Nsight Systems (File → Open).")


def main():
    p = argparse.ArgumentParser(description="Profile a Triton kernel with Nsight Systems.")
    p.add_argument("--kernel", choices=sorted(KERNELS), required=True)
    p.add_argument("--seq-len", type=int, default=512, help="sequence length T (default 512)")
    p.add_argument("--iters", type=int, default=100, help="hot-loop iterations to profile (default 100)")
    p.add_argument("--out", default=None, help="output basename (default profile_<kernel>)")
    p.add_argument("--_workload", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._workload:
        run_workload(args.kernel, args.seq_len, args.iters)
    else:
        out = args.out or f"profile_{args.kernel}"
        profile(args.kernel, args.seq_len, args.iters, out)


if __name__ == "__main__":
    main()
