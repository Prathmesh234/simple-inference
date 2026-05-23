"""
Section 4 / 14a benchmark: RMSNorm — PyTorch vs Triton

Runs:
  1. Correctness check — our RMSNorm (PyTorch) vs transformers LlamaRMSNorm
  2. Correctness check — Triton kernel vs PyTorch (bfloat16 tolerance ~2e-2)
  3. Benchmarks at decode/prefill shapes for both backends
  4. Records results to benchmarks/results_baseline.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from transformers import AutoModelForCausalLM

from config import ModelConfig
from loader import WeightLoader
import ops.rmsnorm as rm_mod
from ops.rmsnorm import RMSNorm, _pytorch_rmsnorm
from kernels.rmsnorm_kernel import rmsnorm_triton
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, tensor_core_util_pct, record, print_results

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"
PEAK_BW  = 960.0  # GB/s, RTX 6000 Ada


# ---------------------------------------------------------------------------
# 1. Correctness: our PyTorch vs transformers
# ---------------------------------------------------------------------------

def check_correctness_vs_transformers(loader: WeightLoader, cfg: ModelConfig) -> bool:
    print("\n--- Correctness: PyTorch vs transformers ---")

    our_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DEVICE, DTYPE)
    our_norm.load_weight(loader.get("layers.0.attn_norm", device=DEVICE))

    print("  Loading transformers model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
    )
    ref_norm = ref_model.model.layers[0].input_layernorm
    ref_model.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 128, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

    rm_mod.USE_TRITON = False
    with torch.no_grad():
        our_out = our_norm(x)
        ref_out = ref_norm(x)

    max_diff  = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()
    status = "PASS" if max_diff < 1e-2 else "FAIL"

    print(f"  max  |pytorch - ref| = {max_diff:.2e}   [{status}]")
    print(f"  mean |pytorch - ref| = {mean_diff:.2e}")

    del ref_model
    torch.cuda.empty_cache()
    return status == "PASS"


# ---------------------------------------------------------------------------
# 2. Correctness: Triton vs PyTorch (bfloat16 arithmetic can differ by ~2e-2)
# ---------------------------------------------------------------------------

def check_correctness_triton(loader: WeightLoader, cfg: ModelConfig) -> bool:
    print("\n--- Correctness: Triton vs PyTorch ---")

    norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DEVICE, DTYPE)
    norm.load_weight(loader.get("layers.0.attn_norm", device=DEVICE))

    torch.manual_seed(42)
    x = torch.randn(1, 512, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

    ref = _pytorch_rmsnorm(x, norm.weight, norm.eps)
    got = rmsnorm_triton(x, norm.weight, norm.eps)

    max_diff  = (ref - got).abs().max().item()
    mean_diff = (ref - got).abs().mean().item()
    # Triton fuses the weight multiply in float32; PyTorch does it in bfloat16.
    # This causes ~2e-2 max diff on realistic activation magnitudes — normal.
    status = "PASS" if max_diff < 3e-2 else "FAIL"

    print(f"  max  |triton - pytorch| = {max_diff:.2e}   [{status}]")
    print(f"  mean |triton - pytorch| = {mean_diff:.2e}")
    print(f"  (Triton fuses weight-mul in f32; PyTorch does it in bf16 → tiny rounding diff)")
    return status == "PASS"


# ---------------------------------------------------------------------------
# 3. Benchmarks: both backends side-by-side
# ---------------------------------------------------------------------------

def run_benchmarks(cfg: ModelConfig):
    print("\n--- Benchmarks: PyTorch vs Triton ---")
    print(f"  Peak memory BW: {PEAK_BW:.0f} GB/s  |  RMSNorm is memory-bound\n")

    norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DEVICE, DTYPE)

    shapes = [
        ("decode  T=1",    1, 1,    cfg.hidden_size),
        ("prefill T=128",  1, 128,  cfg.hidden_size),
        ("prefill T=512",  1, 512,  cfg.hidden_size),
        ("prefill T=2048", 1, 2048, cfg.hidden_size),
    ]

    print(f"  {'Config':<22} {'Backend':<12} {'Latency':>10}  {'BW GB/s':>10}  {'BW%peak':>8}  {'Speedup':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for label, B, T, H in shapes:
        x = torch.randn(B, T, H, device=DEVICE, dtype=DTYPE)
        bytes_moved = 2 * B * T * H * 2  # read x + write out, bfloat16
        flops = 5 * B * T * H

        lat_pt = lat_tr = None
        for backend, flag in [("pytorch", False), ("triton", True)]:
            rm_mod.USE_TRITON = flag
            lat = bench_fn(lambda: norm(x))
            bw  = bandwidth_gb_s(bytes_moved, lat)
            bw_pct = bw / PEAK_BW * 100
            tc_pct = tensor_core_util_pct(flops, lat)

            if backend == "pytorch":
                lat_pt = lat
                speedup_str = "  —"
            else:
                lat_tr = lat
                speedup_str = f"{lat_pt / lat_tr:>6.2f}×"

            print(f"  {label:<22} {backend:<12} {lat:>9.4f}ms  {bw:>10.1f}  {bw_pct:>7.1f}%  {speedup_str:>8}")
            record(
                op_name="rmsnorm",
                backend=backend,
                label=f"B={B} T={T} H={H}",
                latency_ms=lat,
                bandwidth_gb_s_val=bw,
                extra={"batch": B, "seq_len": T, "hidden": H,
                       "peak_bw_pct": round(bw_pct, 1),
                       "tc_util_pct": round(tc_pct, 1)},
            )
        print()

    rm_mod.USE_TRITON = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig.llama_3_2_3b()

    print(f"\n{'='*65}")
    print("  Section 14a — RMSNorm Benchmark  (PyTorch vs Triton)")
    print(f"{'='*65}")

    loader = WeightLoader.from_pretrained(MODEL_ID)

    ok1 = check_correctness_vs_transformers(loader, cfg)
    ok2 = check_correctness_triton(loader, cfg)

    if not (ok1 and ok2):
        print("\n[ERROR] Correctness check(s) failed — fix before benchmarking")
        sys.exit(1)

    run_benchmarks(cfg)
    print_results("rmsnorm")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
    print(f"  USE_TRITON=True is now active in ops/rmsnorm.py")
