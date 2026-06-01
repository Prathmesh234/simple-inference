"""
Roofline: is the attention kernel compute-bound or memory-bound at each shape?

For every shape we measure median latency (triton.testing.do_bench) and compute:
  - arithmetic intensity  = FLOPs / HBM bytes          (x-axis of the roofline)
  - achieved TFLOPS       = FLOPs / latency
  - achieved GB/s         = bytes / latency
  - % of peak compute and % of peak bandwidth

The roofline ridge point on RTX 6000 Ada is peak_flops / peak_bw ~= 1518 FLOP/B.
A shape below that intensity is fundamentally MEMORY-bound (you can't go faster
without moving fewer bytes); above it is COMPUTE-bound (need more FLOP/s).

Two regimes are swept:
  - PREFILL  (Tq = Tk = T, causal): intensity grows with T -> heads toward
    compute-bound. This is why prefill benefits from big tensor-core tiles.
  - DECODE   (Tq = 1, Tk grows): intensity stays tiny -> permanently
    memory-bound. This is why decode is dominated by KV-cache reads and is the
    target for KV layout / quantization / CUDA-graph work, not bigger tiles.

If matplotlib is available a PNG is written to profiling/out/; otherwise a text
roofline table is printed (no extra dependency required).

Run:
    PATH="$HOME/.local/bin:$PATH" XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
    UV_CACHE_DIR="$HOME/.cache/uv" \
    uv run python profiling/roofline_attention.py
"""

from __future__ import annotations

import torch
import triton

from profile_utils import (
    make_prefill_qkv, make_decode_qkv, warmup, banner, OUT_DIR,
    attn_flops, attn_io_bytes, arithmetic_intensity, roofline_bound,
    achieved_tflops, achieved_bw_gb_s,
    PEAK_TFLOPS_BF16, PEAK_BW_GB_S, RIDGE_FLOP_PER_BYTE,
)
from kernels.attention_kernel import attention_flash_triton


def _measure(fn) -> float:
    warmup(fn, iters=15)
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def _row(regime, B, Tq, Tk, causal, fn):
    lat = _measure(fn)
    flops = attn_flops(B, Tq, Tk, causal)
    byts  = attn_io_bytes(B, Tq, Tk)
    ai    = arithmetic_intensity(B, Tq, Tk, causal)
    tflop = achieved_tflops(flops, lat)
    bw    = achieved_bw_gb_s(byts, lat)
    return {
        "regime": regime, "shape": f"B{B} Tq{Tq} Tk{Tk}",
        "lat_ms": lat, "ai": ai, "tflops": tflop, "bw": bw,
        "compute_pct": tflop / PEAK_TFLOPS_BF16 * 100,
        "bw_pct": bw / PEAK_BW_GB_S * 100,
        "bound": roofline_bound(ai),
    }


def main():
    assert torch.cuda.is_available(), "CUDA required"
    rows = []

    for T in (128, 512, 1024, 2048, 4096):
        q, k, v = make_prefill_qkv(B=1, T=T)
        rows.append(_row("prefill", 1, T, T, True,
                         lambda q=q, k=k, v=v: attention_flash_triton(q, k, v, causal=True, assume_contiguous=True)))

    for Tk in (128, 512, 1024, 2048, 4096):
        q, k, v = make_decode_qkv(B=1, Tk=Tk)
        rows.append(_row("decode", 1, 1, Tk, False,
                         lambda q=q, k=k, v=v: attention_flash_triton(q, k, v, causal=False, assume_contiguous=True)))

    banner(f"Attention roofline  (ridge = {RIDGE_FLOP_PER_BYTE:.0f} FLOP/byte, "
           f"peak {PEAK_TFLOPS_BF16:.0f} TFLOPS / {PEAK_BW_GB_S:.0f} GB/s)")
    hdr = (f"{'regime':<8} {'shape':<16} {'lat(ms)':>8} {'AI(F/B)':>9} "
           f"{'TFLOPS':>8} {'%comp':>6} {'GB/s':>7} {'%bw':>6}  bound")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['regime']:<8} {r['shape']:<16} {r['lat_ms']:>8.3f} "
              f"{r['ai']:>9.1f} {r['tflops']:>8.1f} {r['compute_pct']:>5.1f}% "
              f"{r['bw']:>7.0f} {r['bw_pct']:>5.1f}%  {r['bound']}")

    _maybe_plot(rows)


def _maybe_plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("\n(matplotlib not installed — skipping PNG. "
              "`uv pip install matplotlib` to enable the roofline plot.)")
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 6))
    ai = np.logspace(-1, 4, 200)
    roof = np.minimum(PEAK_TFLOPS_BF16, PEAK_BW_GB_S * ai / 1000.0)  # GB/s*FLOP/B -> TFLOPS
    ax.plot(ai, roof, "k-", label="roofline")
    ax.axvline(RIDGE_FLOP_PER_BYTE, ls="--", color="gray", label="ridge point")
    for r in rows:
        m = "o" if r["regime"] == "prefill" else "s"
        ax.scatter(r["ai"], r["tflops"], marker=m,
                   label=f"{r['regime']} {r['shape']}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved TFLOPS (bf16)")
    ax.set_title("FlashAttention Triton kernel — roofline (RTX 6000 Ada)")
    ax.legend(fontsize=7, loc="lower right")
    out = OUT_DIR / "attention_roofline.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\n  Roofline plot written: {out}")


if __name__ == "__main__":
    main()
