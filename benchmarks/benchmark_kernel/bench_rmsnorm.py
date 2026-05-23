"""
Section 4 benchmark: RMSNorm

Runs:
  1. Correctness check — our RMSNorm vs transformers LlamaRMSNorm
  2. Benchmarks at decode shape (T=1) and prefill shapes (T=128, T=2048)
  3. Records results to benchmarks/results_baseline.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM

from config import ModelConfig
from loader import WeightLoader
from ops.rmsnorm import RMSNorm
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, tensor_core_util_pct, record, print_results

DEVICE = "cuda"
DTYPE  = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------

def check_correctness(loader: WeightLoader, cfg: ModelConfig):
    print("\n--- Correctness check ---")

    # Our implementation
    our_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DEVICE, DTYPE)
    our_norm.load_weight(loader.get("layers.0.attn_norm", device=DEVICE))

    # Reference: pull LlamaRMSNorm from transformers (no full model load needed)
    print("  Loading transformers model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    ref_norm = ref_model.model.layers[0].input_layernorm
    ref_model.eval()

    # Random input — same seed so both see identical data
    torch.manual_seed(42)
    x = torch.randn(1, 128, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        our_out = our_norm(x)
        ref_out = ref_norm(x)

    max_diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()
    status = "PASS" if max_diff < 1e-2 else "FAIL"

    print(f"  max  |our - ref| = {max_diff:.2e}   [{status}]")
    print(f"  mean |our - ref| = {mean_diff:.2e}")
    print(f"  bfloat16 tolerance is ~1e-2 — anything below that is numerical noise")

    del ref_model  # free VRAM before benchmarking
    torch.cuda.empty_cache()

    return status == "PASS"


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(cfg: ModelConfig):
    print("\n--- Benchmarks ---")
    print(f"  RTX 6000 Ada peak memory bandwidth: ~960 GB/s")
    print(f"  RMSNorm is memory-bound: we read x, write output — bandwidth tells us how close we are to the limit\n")

    norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DEVICE, DTYPE)

    shapes = [
        ("decode  T=1",    1, 1,    cfg.hidden_size),
        ("prefill T=128",  1, 128,  cfg.hidden_size),
        ("prefill T=512",  1, 512,  cfg.hidden_size),
        ("prefill T=2048", 1, 2048, cfg.hidden_size),
    ]

    print(f"  {'Config':<24} {'Latency':>10}  {'BW GB/s':>10}  {'BW%peak':>8}  {'TC%peak':>8}")
    print(f"  {'-'*24} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    PEAK_BW = 960.0  # GB/s, RTX 6000 Ada

    for label, B, T, H in shapes:
        x = torch.randn(B, T, H, device=DEVICE, dtype=DTYPE)

        # bytes moved: read x (BF16 = 2 bytes/elem) + write output
        # weight is tiny (H elems) — negligible vs x for large T
        bytes_moved = 2 * B * T * H * 2  # 2× for read + write, 2 bytes per bfloat16

        # FLOPs: x^2 (H) + mean reduction (H) + rsqrt (1) + scale x (H) + mul weight (H)
        # Rough total: ~5 ops per element
        flops = 5 * B * T * H

        latency_ms = bench_fn(lambda: norm(x))
        bw = bandwidth_gb_s(bytes_moved, latency_ms)
        bw_pct = bw / PEAK_BW * 100
        tc_pct = tensor_core_util_pct(flops, latency_ms)

        short_label = f"B={B} T={T} H={H}"
        print(f"  {label:<24} {latency_ms:>9.4f}ms  {bw:>10.1f}  {bw_pct:>7.1f}%  {tc_pct:>7.1f}%")

        record(
            op_name="rmsnorm",
            backend="pytorch",
            label=short_label,
            latency_ms=latency_ms,
            bandwidth_gb_s_val=bw,
            extra={"batch": B, "seq_len": T, "hidden": H, "peak_bw_pct": round(bw_pct, 1),
                   "tc_util_pct": round(tc_pct, 1)},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig.llama_3_2_3b()

    print(f"\n{'='*60}")
    print("  Section 4 — RMSNorm Benchmark")
    print(f"{'='*60}")

    loader = WeightLoader.from_pretrained(MODEL_ID)

    ok = check_correctness(loader, cfg)
    if not ok:
        print("\n[ERROR] Correctness check failed — fix before benchmarking")
        sys.exit(1)

    run_benchmarks(cfg)
    print_results("rmsnorm")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
