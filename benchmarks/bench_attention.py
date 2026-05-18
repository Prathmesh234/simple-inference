"""
Section 7 benchmark: GroupedQueryAttention

Runs:
  1. Correctness — our GQA vs transformers LlamaAttention layer 0
  2. Benchmarks at decode (T=1) and prefill shapes
  3. Records results to benchmarks/results_baseline.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM

from config import ModelConfig
from loader import WeightLoader
from ops.rope import RopeFrequencies
from ops.attention import GroupedQueryAttention
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, record, print_results

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------

def check_correctness(loader: WeightLoader, cfg: ModelConfig):
    print("\n--- Correctness check ---")

    rope_cfg = cfg.rope_scaling
    freqs = RopeFrequencies(
        head_dim=cfg.head_dim,
        max_seq_len=cfg.max_position_embeddings,
        rope_theta=cfg.rope_theta,
        rope_type=rope_cfg.rope_type,
        factor=rope_cfg.factor,
        low_freq_factor=rope_cfg.low_freq_factor,
        high_freq_factor=rope_cfg.high_freq_factor,
        original_max_seq_len=rope_cfg.original_max_position_embeddings,
        device=torch.device(DEVICE),
    )

    # Build our attention layer
    attn = GroupedQueryAttention(
        hidden_size=cfg.hidden_size,
        num_heads_q=cfg.num_attention_heads,
        num_heads_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rope_freqs=freqs,
    ).to(DEVICE, DTYPE)

    attn.load_weights(
        wq=loader.get("layers.0.attn.wq"),
        wk=loader.get("layers.0.attn.wk"),
        wv=loader.get("layers.0.attn.wv"),
        wo=loader.get("layers.0.attn.wo"),
    )
    attn.eval()

    # Load transformers reference
    print("  Loading transformers model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    )
    ref_model.eval()
    ref_attn = ref_model.model.layers[0].self_attn

    # Random input
    B, T = 1, 64
    torch.manual_seed(3)
    x = torch.randn(B, T, cfg.hidden_size, device=DEVICE, dtype=DTYPE)
    position_ids = torch.arange(T, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        our_out = attn(x, start_pos=0)
        ref_out, _, _ = ref_attn(x, position_ids=position_ids)

    diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()
    status = "PASS" if diff < 5e-2 else "FAIL"

    print(f"  max  |our - ref| = {diff:.2e}   [{status}]")
    print(f"  mean |our - ref| = {mean_diff:.2e}")
    print(f"  (bfloat16 accumulates error across matmuls — tolerance is 5e-2)")

    del ref_model
    torch.cuda.empty_cache()

    return status == "PASS"


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(cfg: ModelConfig):
    print("\n--- Benchmarks ---")
    print("  Attention is COMPUTE-bound at prefill (matmuls dominate)")
    print("  Attention is MEMORY-bound at decode (weights loaded for 1 token)\n")

    rope_cfg = cfg.rope_scaling
    freqs = RopeFrequencies(
        head_dim=cfg.head_dim,
        max_seq_len=4096,
        rope_theta=cfg.rope_theta,
        rope_type=rope_cfg.rope_type,
        factor=rope_cfg.factor,
        low_freq_factor=rope_cfg.low_freq_factor,
        high_freq_factor=rope_cfg.high_freq_factor,
        original_max_seq_len=rope_cfg.original_max_position_embeddings,
        device=torch.device(DEVICE),
    )

    attn = GroupedQueryAttention(
        hidden_size=cfg.hidden_size,
        num_heads_q=cfg.num_attention_heads,
        num_heads_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rope_freqs=freqs,
    ).to(DEVICE, DTYPE)
    attn.eval()

    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
    ]

    # Weight bytes: wq + wk + wv + wo (loaded once per forward pass)
    weight_bytes = (
        cfg.num_attention_heads  * cfg.head_dim * cfg.hidden_size  # wq
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wk
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wv
        + cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim  # wo
    ) * 2  # bfloat16

    PEAK_BW = 960.0

    print(f"  {'Config':<24} {'Latency':>10}  {'Bandwidth':>12}  {'% of peak':>10}")
    print(f"  {'-'*24} {'-'*10}  {'-'*12}  {'-'*10}")

    for label, B, T in shapes:
        x = torch.randn(B, T, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        # At decode (T=1), weight loading dominates — report as weight bandwidth
        # At prefill, activations + weights contribute, weights are amortised
        act_bytes = B * T * cfg.hidden_size * 2 * 2  # read x + write out
        bytes_moved = weight_bytes + act_bytes

        lat_ms = bench_fn(lambda: attn(x, start_pos=0))
        bw     = bandwidth_gb_s(bytes_moved, lat_ms)
        pct    = bw / PEAK_BW * 100

        short = f"B={B} T={T} H={cfg.hidden_size}"
        print(f"  {label:<24} {lat_ms:>9.4f}ms  {bw:>10.1f} GB/s  {pct:>9.1f}%")
        record("attention", "pytorch", short, lat_ms, bw,
               extra={"batch": B, "seq_len": T, "hidden": cfg.hidden_size})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    print(f"\n{'='*60}")
    print("  Section 7 — Attention Benchmark")
    print(f"{'='*60}")

    ok = check_correctness(loader, cfg)
    if not ok:
        print("\n[ERROR] Correctness check failed")
        sys.exit(1)

    run_benchmarks(cfg)
    print_results("attention")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
