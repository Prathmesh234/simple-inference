"""
Section 9 benchmark: TransformerBlock

Runs:
  1. Correctness — our TransformerBlock vs transformers LlamaDecoderLayer (layer 0)
  2. Benchmarks at decode (T=1) and prefill shapes
  3. Records results to benchmarks/results_baseline.json

Why benchmark the full block?
-------------------------------
Individual op benchmarks (attention, MLP) measure each sub-component in isolation.
The block benchmark captures the combined cost including:
  - Two RMSNorms (attn_norm + mlp_norm)
  - Attention (projections + RoPE + SDPA)
  - MLP (SwiGLU)
  - Two residual additions

This is the fundamental repeating unit of the model — 28 of these make up the
full Llama 3.2-3B. Its latency × 28 gives a lower bound on full model cost.

Weight budget per block
------------------------
  attn_norm:  (3072,)          ← tiny
  wq:         (3072, 3072)     = 9.44M params
  wk:         (1024, 3072)     = 3.14M params
  wv:         (1024, 3072)     = 3.14M params
  wo:         (3072, 3072)     = 9.44M params
  mlp_norm:   (3072,)          ← tiny
  w_gate:     (8192, 3072)     = 25.17M params
  w_up:       (8192, 3072)     = 25.17M params
  w_down:     (3072, 8192)     = 25.17M params
  ─────────────────────────────────────────────
  Total:      ~100.7M params × 2 bytes = ~200 MB per layer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM

from config import ModelConfig
from loader import WeightLoader
from ops.rope import RopeFrequencies
from model.block import TransformerBlock
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, tensor_core_util_pct, record, print_results

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

    block = TransformerBlock(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_heads_q=cfg.num_attention_heads,
        num_heads_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rope_freqs=freqs,
        norm_eps=cfg.rms_norm_eps,
    ).to(DEVICE, DTYPE)

    block.load_weights(
        attn_norm_weight=loader.get("layers.0.attn_norm"),
        wq=loader.get("layers.0.attn.wq"),
        wk=loader.get("layers.0.attn.wk"),
        wv=loader.get("layers.0.attn.wv"),
        wo=loader.get("layers.0.attn.wo"),
        mlp_norm_weight=loader.get("layers.0.mlp_norm"),
        w_gate=loader.get("layers.0.mlp.w_gate"),
        w_up=loader.get("layers.0.mlp.w_up"),
        w_down=loader.get("layers.0.mlp.w_down"),
    )
    block.eval()

    print("  Loading transformers model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    )
    ref_model.eval()
    ref_layer = ref_model.model.layers[0]

    B, T = 1, 64
    torch.manual_seed(7)
    x = torch.randn(B, T, cfg.hidden_size, device=DEVICE, dtype=DTYPE)
    position_ids = torch.arange(T, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        # Build position embeddings the same way transformers does
        ref_rope = ref_model.model.rotary_emb
        position_embeddings = ref_rope(x.transpose(1, 2), position_ids)

        our_out = block(x, start_pos=0)
        ref_out = ref_layer(x, position_embeddings=position_embeddings)[0]

    diff = (our_out - ref_out).abs().max().item()
    mean_diff = (our_out - ref_out).abs().mean().item()
    # Block accumulates error across attn + mlp — allow up to 5e-2
    status = "PASS" if diff < 5e-2 else "FAIL"

    print(f"  max  |our - ref| = {diff:.2e}   [{status}]")
    print(f"  mean |our - ref| = {mean_diff:.2e}")
    print(f"  (bfloat16 accumulates error across attn + mlp — tolerance is 5e-2)")

    del ref_model
    torch.cuda.empty_cache()

    return status == "PASS"


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(cfg: ModelConfig):
    print("\n--- Benchmarks ---")
    print("  Block = attention + 2×RMSNorm + MLP + 2×residual")
    print("  At decode:  memory-bound (weights dominate, 1 token processed)")
    print("  At prefill: compute-bound (large matmuls over T tokens)\n")

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

    block = TransformerBlock(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_heads_q=cfg.num_attention_heads,
        num_heads_kv=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rope_freqs=freqs,
        norm_eps=cfg.rms_norm_eps,
    ).to(DEVICE, DTYPE)
    block.eval()

    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
    ]

    # ── bytes moved ──────────────────────────────────────────────────────────
    # Attention weights: wq + wk + wv + wo
    attn_weight_bytes = (
        cfg.num_attention_heads  * cfg.head_dim * cfg.hidden_size   # wq
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wk
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wv
        + cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim  # wo
    ) * 2  # bfloat16

    # MLP weights: w_gate + w_up + w_down
    mlp_weight_bytes = (
        cfg.intermediate_size * cfg.hidden_size   # w_gate
        + cfg.intermediate_size * cfg.hidden_size  # w_up
        + cfg.hidden_size * cfg.intermediate_size  # w_down
    ) * 2  # bfloat16

    # Norms: two small weight vectors — negligible but included for completeness
    norm_weight_bytes = 2 * cfg.hidden_size * 2  # bfloat16

    total_weight_bytes = attn_weight_bytes + mlp_weight_bytes + norm_weight_bytes

    # ── FLOPs per token ──────────────────────────────────────────────────────
    # Attention projections
    proj_flops_per_BT = 2 * (
        cfg.num_attention_heads  * cfg.head_dim * cfg.hidden_size   # wq
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wk
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wv
        + cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size  # wo
    )
    # MLP projections
    mlp_flops_per_BT = 2 * (
        cfg.hidden_size * cfg.intermediate_size   # x @ w_gate.T
        + cfg.hidden_size * cfg.intermediate_size  # x @ w_up.T
        + cfg.intermediate_size * cfg.hidden_size  # (gate*up) @ w_down.T
    )

    PEAK_BW = 960.0  # GB/s, RTX 6000 Ada

    print(f"  {'Config':<24} {'Latency':>10}  {'BW GB/s':>10}  {'BW%peak':>8}  {'TC%peak':>8}")
    print(f"  {'-'*24} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for label, B, T in shapes:
        x = torch.randn(B, T, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        act_bytes = B * T * cfg.hidden_size * 2 * 2  # read x + write output
        bytes_moved = total_weight_bytes + act_bytes

        # QK^T and softmax@V FLOPs (attention score computation)
        attn_score_flops = 2 * 2 * B * cfg.num_attention_heads * T * T * cfg.head_dim
        flops = B * T * (proj_flops_per_BT + mlp_flops_per_BT) + attn_score_flops

        lat_ms = bench_fn(lambda: block(x, start_pos=0))
        bw     = bandwidth_gb_s(bytes_moved, lat_ms)
        bw_pct = bw / PEAK_BW * 100
        tc_pct = tensor_core_util_pct(flops, lat_ms)

        short = f"B={B} T={T} H={cfg.hidden_size}"
        print(f"  {label:<24} {lat_ms:>9.4f}ms  {bw:>10.1f}  {bw_pct:>7.1f}%  {tc_pct:>7.1f}%")
        record("block", "pytorch", short, lat_ms, bw,
               extra={"batch": B, "seq_len": T, "hidden": cfg.hidden_size,
                      "intermediate": cfg.intermediate_size, "tc_util_pct": round(tc_pct, 1)})

    # Extra insight: estimated full-model cost = block × 28
    print(f"\n  --- Full model estimate (28 blocks) ---")
    print(f"  {'Config':<24} {'Est. 28-layer':>14}")
    print(f"  {'-'*24} {'-'*14}")

    from benchmarks.bench_utils import _load, BASELINE_FILE
    data = _load(BASELINE_FILE)
    for entry in data.get("block", {}).get("pytorch", []):
        est = entry["latency_ms"] * 28
        print(f"  {entry['label']:<24} {est:>12.2f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    print(f"\n{'='*60}")
    print("  Section 9 — TransformerBlock Benchmark")
    print(f"{'='*60}")

    ok = check_correctness(loader, cfg)
    if not ok:
        print("\n[ERROR] Correctness check failed")
        sys.exit(1)

    run_benchmarks(cfg)
    print_results("block")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
