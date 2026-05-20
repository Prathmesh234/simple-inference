"""
Section 8 benchmark: SwiGLUMLP

Runs:
  1. Correctness — our SwiGLUMLP vs transformers LlamaMLP layer 0
  2. Benchmarks at decode (T=1) and prefill shapes
  3. Records results to benchmarks/results_baseline.json

Why MLP is the most compute-heavy op
--------------------------------------
Attention weights (GQA):
  wq (3072×3072) + wk (1024×3072) + wv (1024×3072) + wo (3072×3072)
  Total: ~38M params per layer

MLP weights (SwiGLU):
  w_gate (8192×3072) + w_up (8192×3072) + w_down (3072×8192)
  Total: ~75M params per layer  ← 2× more than attention

More parameters = more FLOPs per token = more time at both prefill and decode.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM

from config import ModelConfig
from loader import WeightLoader
from ops.mlp import SwiGLUMLP
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, tensor_core_util_pct, record, print_results

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------

def check_correctness(loader: WeightLoader, cfg: ModelConfig):
    print("\n--- Correctness check ---")

    # Our implementation
    mlp = SwiGLUMLP(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
    ).to(DEVICE, DTYPE)

    mlp.load_weights(
        w_gate=loader.get("layers.0.mlp.w_gate"),
        w_up=loader.get("layers.0.mlp.w_up"),
        w_down=loader.get("layers.0.mlp.w_down"),
    )
    mlp.eval()

    # Load transformers reference
    print("  Loading transformers model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    )
    ref_model.eval()
    ref_mlp = ref_model.model.layers[0].mlp

    # Random input — same seed for both
    B, T = 1, 64
    torch.manual_seed(42)
    x = torch.randn(B, T, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        our_out = mlp(x)
        ref_out = ref_mlp(x)

    diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()
    status = "PASS" if diff < 1e-2 else "FAIL"

    print(f"  max  |our - ref| = {diff:.2e}   [{status}]")
    print(f"  mean |our - ref| = {mean_diff:.2e}")
    print(f"  (bfloat16 tolerance is ~1e-2)")

    del ref_model
    torch.cuda.empty_cache()

    return status == "PASS"


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(cfg: ModelConfig):
    print("\n--- Benchmarks ---")
    print("  MLP is COMPUTE-bound at prefill (large matmuls dominate)")
    print("  MLP is MEMORY-bound at decode (all weight bytes loaded for 1 token)\n")

    mlp = SwiGLUMLP(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
    ).to(DEVICE, DTYPE)
    mlp.eval()

    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
    ]

    # Weight bytes: w_gate + w_up + w_down (all loaded once per forward pass)
    weight_bytes = (
        cfg.intermediate_size * cfg.hidden_size   # w_gate
        + cfg.intermediate_size * cfg.hidden_size  # w_up
        + cfg.hidden_size * cfg.intermediate_size  # w_down
    ) * 2  # bfloat16

    PEAK_BW = 960.0  # GB/s, RTX 6000 Ada

    print(f"  {'Config':<24} {'Latency':>10}  {'BW GB/s':>10}  {'BW%peak':>8}  {'TC%peak':>8}")
    print(f"  {'-'*24} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for label, B, T in shapes:
        x = torch.randn(B, T, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        # Bytes moved: weights + read x + write output (activations are small vs weights)
        act_bytes = B * T * cfg.hidden_size * 2 * 2  # read x + write out
        bytes_moved = weight_bytes + act_bytes

        # FLOPs: 3 matmuls — w_gate, w_up, w_down
        # Each matmul (B*T, in) @ (in, out) = 2 * B * T * in * out FLOPs
        flops = 2 * B * T * (
            cfg.hidden_size * cfg.intermediate_size   # x @ w_gate.T
            + cfg.hidden_size * cfg.intermediate_size  # x @ w_up.T
            + cfg.intermediate_size * cfg.hidden_size  # (gate*up) @ w_down.T
        )

        lat_ms = bench_fn(lambda: mlp(x))
        bw     = bandwidth_gb_s(bytes_moved, lat_ms)
        bw_pct = bw / PEAK_BW * 100
        tc_pct = tensor_core_util_pct(flops, lat_ms)

        short = f"B={B} T={T} H={cfg.hidden_size}"
        print(f"  {label:<24} {lat_ms:>9.4f}ms  {bw:>10.1f}  {bw_pct:>7.1f}%  {tc_pct:>7.1f}%")
        record("mlp", "pytorch", short, lat_ms, bw,
               extra={"batch": B, "seq_len": T, "hidden": cfg.hidden_size,
                      "intermediate": cfg.intermediate_size, "tc_util_pct": round(tc_pct, 1)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    print(f"\n{'='*60}")
    print("  Section 8 — MLP (SwiGLU) Benchmark")
    print(f"{'='*60}")

    ok = check_correctness(loader, cfg)
    if not ok:
        print("\n[ERROR] Correctness check failed")
        sys.exit(1)

    run_benchmarks(cfg)
    print_results("mlp")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
