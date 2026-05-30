"""
Attention: Flash-Triton vs PyTorch SDPA — and a look at the memory wall.

Two implementations of the *same* GQA causal attention:

  - flash : FlashAttention-2 Triton kernel (online softmax, tiled). Never
            materialises the T×T score matrix.   HBM traffic ∝ T.
  - sdpa  : torch.nn.functional.scaled_dot_product_attention (the production
            reference; itself a flash kernel under the hood).

The textbook formulation would materialise the full (B,Hq,T,T) score matrix in
HBM (traffic ∝ T²). The table at the end quantifies how much HBM that costs
relative to flash's total I/O as T grows — that gap *is* the reason
FlashAttention exists.

For full correctness vs transformers + JSON recording, see bench_attention.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from kernels.attention_kernel import attention_flash_triton
from benchmarks.bench_utils import bench_fn

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
HEAD_DIM = 128
N_Q      = 24          # Llama-3.2-3B query heads
N_KV     = 8           # KV heads (GQA group = 3)
KV_GROUP = N_Q // N_KV
PEAK_BW       = 960.0    # GB/s,  RTX 6000 Ada
PEAK_TFLOPS   = 1457.0   # bf16 tensor-core


def sdpa_ref(q, k, v, causal):
    kk = k.repeat_interleave(KV_GROUP, dim=1)
    vv = v.repeat_interleave(KV_GROUP, dim=1)
    return F.scaled_dot_product_attention(q, kk, vv, is_causal=causal)


def make_qkv(B, T):
    q = torch.randn(B, N_Q,  T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N_KV, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N_KV, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, v


def attn_flops(B, T):
    # QK^T + P@V, each 2*B*Hq*T*T*d, causal ≈ half the score entries are live.
    full = 2 * 2 * B * N_Q * T * T * HEAD_DIM
    return full * 0.5


def io_bytes_flash(B, T):
    # read Q,K,V once + write O once (bf16). K/V are GQA (N_KV heads).
    q = B * N_Q  * T * HEAD_DIM
    kv = 2 * B * N_KV * T * HEAD_DIM
    o = B * N_Q  * T * HEAD_DIM
    return (q + kv + o) * 2


def io_bytes_materialised_scores(B, T):
    # what the textbook path would pay: write + read the fp32 (B,Hq,T,T) scores.
    return 2 * (B * N_Q * T * T) * 4


def main():
    print("\n--- Correctness vs SDPA (bf16, causal) ---")
    torch.manual_seed(0)
    q, k, v = make_qkv(1, 256)
    ref = sdpa_ref(q, k, v, causal=True)
    got = attention_flash_triton(q, k, v, causal=True)
    diff = (got - ref).abs().max().item()
    print(f"  flash  max |triton - sdpa| = {diff:.2e}   "
          f"[{'PASS' if diff < 5e-2 else 'FAIL'}]")

    # decode shape (Tq=1 over a cached prefix): build separately.
    print("\n--- Correctness: decode (Tq=1, attend full prefix) ---")
    Tk = 512
    qd = torch.randn(1, N_Q,  1,  HEAD_DIM, device=DEVICE, dtype=DTYPE)
    kd = torch.randn(1, N_KV, Tk, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    vd = torch.randn(1, N_KV, Tk, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    kk = kd.repeat_interleave(KV_GROUP, dim=1); vv = vd.repeat_interleave(KV_GROUP, dim=1)
    ref_d = F.scaled_dot_product_attention(qd, kk, vv, is_causal=False)
    got_d = attention_flash_triton(qd, kd, vd, causal=False)
    diff = (got_d - ref_d).abs().max().item()
    print(f"  flash  max |triton - sdpa| = {diff:.2e}   "
          f"[{'PASS' if diff < 5e-2 else 'FAIL'}]")

    print(f"\n--- Latency (prefill, causal). peak {PEAK_BW:.0f} GB/s / "
          f"{PEAK_TFLOPS:.0f} TFLOPS ---")
    print(f"  {'Config':<14} {'flash':>11} {'sdpa':>11} "
          f"{'flash/sdpa':>11} {'flash TC%':>10}")
    print(f"  {'-'*14} {'-'*11} {'-'*11} {'-'*11} {'-'*10}")

    shapes = [
        ("T=128",  1, 128),
        ("T=512",  1, 512),
        ("T=1024", 1, 1024),
        ("T=2048", 1, 2048),
        ("T=4096", 1, 4096),
    ]
    for label, B, T in shapes:
        q, k, v = make_qkv(B, T)

        for _ in range(3):
            attention_flash_triton(q, k, v, causal=True)
            sdpa_ref(q, k, v, causal=True)

        lat_flash = bench_fn(lambda: attention_flash_triton(q, k, v, causal=True))
        lat_sdpa  = bench_fn(lambda: sdpa_ref(q, k, v, causal=True))

        tc = attn_flops(B, T) / (lat_flash / 1000) / 1e12 / PEAK_TFLOPS * 100
        print(f"  {label:<14} {lat_flash*1000:>8.1f} \u00b5s {lat_sdpa*1000:>8.1f} \u00b5s "
              f"{lat_sdpa/lat_flash:>10.2f}\u00d7 {tc:>9.1f}%")

    print("\n--- The memory wall: HBM a materialised-score path would pay ---")
    print(f"  {'Config':<14} {'flash I/O':>12} {'scores I/O':>12} {'x bigger':>9}")
    print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*9}")
    for label, B, T in shapes:
        flash_io = io_bytes_flash(B, T)
        scores_io = io_bytes_materialised_scores(B, T)
        print(f"  {label:<14} {flash_io/1e6:>9.1f} MB {scores_io/1e6:>9.1f} MB "
              f"{scores_io/flash_io:>8.1f}\u00d7")

    print("\n  Roofline note (RTX 6000 Ada): prefill attention is compute-bound")
    print("  (tensor-core %, above) \u2014 flash wins by NOT paying the O(T\u00b2) HBM")
    print("  tax a materialised score matrix imposes. Decode (Tq=1) is")
    print("  memory-bound; flash's win there is avoiding the KV repeat copy.")


if __name__ == "__main__":
    main()
