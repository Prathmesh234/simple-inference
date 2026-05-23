"""
SwiGLU activation: Triton fused kernel vs PyTorch.

Compares two things:
  1. The elementwise step in isolation  →  silu(gate) * up
  2. The full MLP forward pass          →  3 matmuls + the activation
     (lets us see what fraction of MLP latency the fusion actually saves)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import torch
import torch.nn.functional as F

import ops.mlp as mlp_mod
from ops.mlp import SwiGLUMLP
from kernels.swiglu_kernel import swiglu_triton
from benchmarks.bench_utils import bench_fn


DEVICE   = "cuda"
DTYPE    = torch.bfloat16
H        = 3072      # Llama 3.2-3B hidden_size
I        = 8192      # Llama 3.2-3B intermediate_size
PEAK_BW  = 960.0     # GB/s, RTX 6000 Ada


def _pytorch_swiglu(gate, up):
    return F.silu(gate) * up


def main():
    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
        ("big     T=8192", 1, 8192),
    ]

    # ── Trigger Triton autotune sweep up front ────────────────────────────
    print("\n--- Warmup: triggering autotune sweep (one-time cost) ---")
    torch.manual_seed(0)
    g = torch.randn(1, 512, I, device=DEVICE, dtype=DTYPE)
    u = torch.randn(1, 512, I, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    swiglu_triton(g, u)
    torch.cuda.synchronize()
    sweep_ms = (time.perf_counter() - t0) * 1000
    from kernels.swiglu_kernel import _swiglu_fwd
    chosen = next(iter(_swiglu_fwd.cache.values()))
    print(f"  Autotune sweep took {sweep_ms:.0f} ms")
    print(f"  Selected: BLOCK_SIZE={chosen.kwargs['BLOCK_SIZE']}, "
          f"num_warps={chosen.num_warps}, num_stages={chosen.num_stages}")

    # ── Correctness ───────────────────────────────────────────────────────
    print("\n--- Correctness ---")
    ref = _pytorch_swiglu(g, u)
    got = swiglu_triton(g, u)
    max_diff  = (ref - got).abs().max().item()
    mean_diff = (ref - got).abs().mean().item()
    print(f"  max  |pytorch - triton| = {max_diff:.2e}   "
          f"[{'PASS' if max_diff < 1e-1 else 'FAIL'}]")
    print(f"  mean |pytorch - triton| = {mean_diff:.2e}")

    # ── Elementwise step in isolation ─────────────────────────────────────
    print(f"\n--- Elementwise silu(gate)*up only  (peak {PEAK_BW:.0f} GB/s) ---")
    print(f"  {'Config':<18} {'PyTorch':>12} {'Triton':>12} {'Speedup':>10} "
          f"{'PT BW%':>8} {'TR BW%':>8}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

    for label, B, T in shapes:
        g = torch.randn(B, T, I, device=DEVICE, dtype=DTYPE)
        u = torch.randn(B, T, I, device=DEVICE, dtype=DTYPE)
        # bytes: read gate + read up + write out, all bf16 (2 bytes)
        bytes_moved = 3 * B * T * I * 2

        for _ in range(5):
            _pytorch_swiglu(g, u); swiglu_triton(g, u)

        lat_pt = bench_fn(lambda: _pytorch_swiglu(g, u))
        lat_tr = bench_fn(lambda: swiglu_triton(g, u))
        bw_pt = (bytes_moved / 1e9) / (lat_pt / 1000) / PEAK_BW * 100
        bw_tr = (bytes_moved / 1e9) / (lat_tr / 1000) / PEAK_BW * 100

        print(f"  {label:<18} {lat_pt*1000:>9.2f} µs {lat_tr*1000:>9.2f} µs "
              f"{lat_pt/lat_tr:>8.2f}× {bw_pt:>7.1f}% {bw_tr:>7.1f}%")

    # ── Level-2 gain: 2 separate matmuls vs 1 fused matmul ───────────────
    # This compares the gate+up GEMM(s) only. The activation and down-proj
    # are NOT included here — we want to isolate the matmul-fusion effect.
    print(f"\n--- Level-2 gain: 2 GEMMs (W_gate, W_up) vs 1 fused GEMM (W_gate_up) ---")
    print(f"  {'Config':<18} {'2 GEMMs':>12} {'1 GEMM':>12} {'Speedup':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")

    Wg = torch.randn(I, H, device=DEVICE, dtype=DTYPE) * 0.02
    Wu = torch.randn(I, H, device=DEVICE, dtype=DTYPE) * 0.02
    Wgu = torch.cat([Wg, Wu], dim=0)   # (2*I, H)

    for label, B, T in shapes:
        x = torch.randn(B, T, H, device=DEVICE, dtype=DTYPE)

        def two_gemms():
            gate = F.linear(x, Wg)
            up   = F.linear(x, Wu)
            return gate, up

        def one_gemm():
            combined = F.linear(x, Wgu)
            return combined.chunk(2, dim=-1)

        for _ in range(5): two_gemms(); one_gemm()
        lat_2 = bench_fn(two_gemms)
        lat_1 = bench_fn(one_gemm)
        print(f"  {label:<18} {lat_2*1000:>9.2f} µs {lat_1*1000:>9.2f} µs "
              f"{lat_2/lat_1:>8.2f}×")

    # ── Full MLP forward (3 matmuls + activation) ─────────────────────────
    # This is what actually matters for the model. Matmuls dominate the time;
    # the fusion saves only the activation step, so end-to-end gain is smaller.
    print(f"\n--- Full MLP forward  (Triton SwiGLU on/off; gate+up always fused) ---")
    print(f"  {'Config':<18} {'PyTorch':>12} {'Triton':>12} {'Speedup':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")

    mlp = SwiGLUMLP(H, I).to(DEVICE, DTYPE)
    # Init the params with something reasonable
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0, 0.02)

    for label, B, T in shapes:
        x = torch.randn(B, T, H, device=DEVICE, dtype=DTYPE)

        # PyTorch backend
        mlp_mod.USE_TRITON = False
        for _ in range(5): mlp(x)
        lat_pt = bench_fn(lambda: mlp(x))

        # Triton backend
        mlp_mod.USE_TRITON = True
        for _ in range(5): mlp(x)
        lat_tr = bench_fn(lambda: mlp(x))

        print(f"  {label:<18} {lat_pt*1000:>9.2f} µs {lat_tr*1000:>9.2f} µs "
              f"{lat_pt/lat_tr:>8.2f}×")

    mlp_mod.USE_TRITON = True


if __name__ == "__main__":
    main()
