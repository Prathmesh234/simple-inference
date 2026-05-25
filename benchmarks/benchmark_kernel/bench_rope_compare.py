"""
Triton RoPE vs PyTorch RoPE — quick A/B comparison across shapes.

For full correctness vs transformers + JSON recording, see bench_rope.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

# Build a PyTorch-only reference by force-disabling Triton dispatch before import.
os.environ["USE_TRITON"] = "false"
from ops.rope import apply_rope as apply_rope_pt, RopeFrequencies

from kernels.rope_kernel import rope_triton
from benchmarks.bench_utils import bench_fn


DEVICE   = "cuda"
DTYPE    = torch.bfloat16
HEAD_DIM = 128
N_Q      = 24
N_KV     = 8
PEAK_BW  = 960.0       # GB/s, RTX 6000 Ada


def bytes_moved(B, T):
    # Read Q, K + read cos, sin (broadcast across batch & heads — count once)
    # + write Q_rot, K_rot. All bf16 (2 bytes).
    nq = B * T * N_Q  * HEAD_DIM
    nk = B * T * N_KV * HEAD_DIM
    nc = T * HEAD_DIM
    return (nq + nk + nc + nc + nq + nk) * 2


def main():
    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
        ("big     T=8192", 1, 8192),
    ]

    # ── Trigger Triton autotuning FIRST (before any timed work) ───────────
    print("\n--- Warmup: triggering autotune sweep (one-time cost) ---")
    import time
    torch.manual_seed(0)
    freqs = RopeFrequencies(head_dim=HEAD_DIM, max_seq_len=8192, device=torch.device(DEVICE))
    q = torch.randn(1, 512, N_Q,  HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, 512, N_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    cos, sin = freqs.get(512)
    cos = cos.to(DTYPE); sin = sin.to(DTYPE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rope_triton(q, k, cos, sin)
    torch.cuda.synchronize()
    sweep_ms = (time.perf_counter() - t0) * 1000

    from kernels.rope_kernel import _rope_fwd
    chosen = next(iter(_rope_fwd.cache.values()))
    print(f"  Autotune sweep took {sweep_ms:.0f} ms  (cached → free thereafter)")
    print(f"  Selected config: num_warps={chosen.num_warps}, num_stages={chosen.num_stages}")

    # ── Correctness sanity check ──────────────────────────────────────────
    print("\n--- Correctness (max diff, bf16 rounding ~3e-2 with random Q/K) ---")
    q_ref, k_ref = apply_rope_pt(q, k, cos, sin)
    q_got, k_got = rope_triton(q, k, cos, sin)
    q_diff = (q_ref - q_got).abs().max().item()
    k_diff = (k_ref - k_got).abs().max().item()
    print(f"  Q max |pytorch - triton| = {q_diff:.2e}   "
          f"[{'PASS' if q_diff < 5e-2 else 'FAIL'}]")
    print(f"  K max |pytorch - triton| = {k_diff:.2e}   "
          f"[{'PASS' if k_diff < 5e-2 else 'FAIL'}]")

    # ── Latency ───────────────────────────────────────────────────────────
    print(f"\n--- Latency / Bandwidth (peak {PEAK_BW:.0f} GB/s) ---")
    print(f"  {'Config':<18} {'PyTorch':>12} {'Triton':>12} {'Speedup':>10} "
          f"{'PT BW%':>8} {'TR BW%':>8}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

    for label, B, T in shapes:
        q = torch.randn(B, T, N_Q,  HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T, N_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        cos, sin = freqs.get(T)
        cos = cos.to(DTYPE); sin = sin.to(DTYPE)
        bm = bytes_moved(B, T)

        for _ in range(5):
            apply_rope_pt(q, k, cos, sin)
            rope_triton(q, k, cos, sin)

        lat_pt = bench_fn(lambda: apply_rope_pt(q, k, cos, sin))
        lat_tr = bench_fn(lambda: rope_triton(q, k, cos, sin))

        bw_pt = (bm / 1e9) / (lat_pt / 1000) / PEAK_BW * 100
        bw_tr = (bm / 1e9) / (lat_tr / 1000) / PEAK_BW * 100

        print(f"  {label:<18} {lat_pt*1000:>9.2f} µs {lat_tr*1000:>9.2f} µs "
              f"{lat_pt/lat_tr:>8.2f}× {bw_pt:>7.1f}% {bw_tr:>7.1f}%")


if __name__ == "__main__":
    main()
