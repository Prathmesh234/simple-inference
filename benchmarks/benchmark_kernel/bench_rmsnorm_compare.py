"""
Triton RMSNorm vs PyTorch RMSNorm — quick A/B comparison across shapes.

Started life as a 3-way rsqrt-variant test; now just two contenders:
  - PyTorch reference (_pytorch_rmsnorm)
  - Our Triton kernel (rmsnorm_triton)

For a full benchmark with correctness vs transformers, see bench_rmsnorm.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from ops.rmsnorm import _pytorch_rmsnorm
from kernels.rmsnorm_kernel import rmsnorm_triton
from benchmarks.bench_utils import bench_fn


DEVICE   = "cuda"
DTYPE    = torch.bfloat16
H        = 3072        # Llama 3.2-3B hidden size
EPS      = 1e-5
PEAK_BW  = 960.0       # GB/s, RTX 6000 Ada


def main():
    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
        ("big     T=8192", 1, 8192),
    ]

    # ── Trigger Triton autotuning FIRST (before any timed work) ───────────
    # The autotuner key is N (hidden_size), so one call per dtype trains the
    # cache for every shape we'll bench afterwards. Doing it here means the
    # correctness check and the per-shape benches are not polluted by the
    # 12-config sweep cost.
    print("\n--- Warmup: triggering autotune sweep (one-time cost) ---")
    import time
    torch.manual_seed(0)
    x = torch.randn(1, 512, H, device=DEVICE, dtype=DTYPE)
    w = torch.randn(H, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rmsnorm_triton(x, w, EPS)
    torch.cuda.synchronize()
    sweep_ms = (time.perf_counter() - t0) * 1000

    from kernels.rmsnorm_kernel import _rmsnorm_fwd
    chosen = next(iter(_rmsnorm_fwd.cache.values()))
    print(f"  Autotune sweep took {sweep_ms:.0f} ms  (cached → free thereafter)")
    print(f"  Selected config: num_warps={chosen.num_warps}, num_stages={chosen.num_stages}")

    # ── Correctness sanity check ──────────────────────────────────────────
    # Note: with random N(0,1) weights, output magnitudes reach ~3 where
    # bf16 spacing is ~0.03, so max diff up to ~7e-2 is normal rounding noise.
    # bench_rmsnorm.py uses real Llama weights → tighter ~2e-2.
    print("\n--- Correctness (max diff, bf16 rounding noise ~7e-2 with random w) ---")
    ref = _pytorch_rmsnorm(x, w, EPS)
    got = rmsnorm_triton(x, w, EPS)
    max_diff  = (ref - got).abs().max().item()
    mean_diff = (ref - got).abs().mean().item()
    print(f"  max  |pytorch - triton| = {max_diff:.2e}   "
          f"[{'PASS' if max_diff < 1e-1 else 'FAIL'}]")
    print(f"  mean |pytorch - triton| = {mean_diff:.2e}")

    # ── Latency ───────────────────────────────────────────────────────────
    print(f"\n--- Latency / Bandwidth (peak {PEAK_BW:.0f} GB/s) ---")
    print(f"  {'Config':<18} {'PyTorch':>12} {'Triton':>12} {'Speedup':>10} "
          f"{'PT BW%':>8} {'TR BW%':>8}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

    for label, B, T in shapes:
        x = torch.randn(B, T, H, device=DEVICE, dtype=DTYPE)
        w = torch.randn(H, device=DEVICE, dtype=DTYPE)
        bytes_moved = 2 * B * T * H * 2     # read x + write out, bf16

        # Per-shape warmup (autotune already cached, this just warms caches/JIT)
        for _ in range(5):
            _pytorch_rmsnorm(x, w, EPS)
            rmsnorm_triton(x, w, EPS)

        lat_pt = bench_fn(lambda: _pytorch_rmsnorm(x, w, EPS))
        lat_tr = bench_fn(lambda: rmsnorm_triton(x, w, EPS))

        bw_pt = (bytes_moved / 1e9) / (lat_pt / 1000) / PEAK_BW * 100
        bw_tr = (bytes_moved / 1e9) / (lat_tr / 1000) / PEAK_BW * 100

        print(f"  {label:<18} {lat_pt*1000:>9.2f} µs {lat_tr*1000:>9.2f} µs "
              f"{lat_pt/lat_tr:>8.2f}× {bw_pt:>7.1f}% {bw_tr:>7.1f}%")


if __name__ == "__main__":
    main()
