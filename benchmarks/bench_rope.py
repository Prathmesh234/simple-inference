"""
Section 6 benchmark: RoPE

Runs:
  1. Correctness — our apply_rope vs transformers LlamaRotaryEmbedding
  2. Benchmarks at decode (T=1) and prefill shapes
  3. Records results to benchmarks/results_baseline.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM

from config import ModelConfig
from ops.rope import RopeFrequencies, apply_rope
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, tensor_core_util_pct, record, print_results

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------

def check_correctness(cfg: ModelConfig):
    print("\n--- Correctness check ---")

    # Build our RopeFrequencies from the config
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

    # Load transformers model to get its rotary embedding layer
    print("  Loading transformers model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    )
    ref_model.eval()

    # Random Q and K tensors — same shape as what attention would produce
    B, T = 1, 64
    torch.manual_seed(7)
    q = torch.randn(B, T, cfg.num_attention_heads,  cfg.head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, T, cfg.num_key_value_heads,  cfg.head_dim, device=DEVICE, dtype=DTYPE)

    # Our implementation
    cos, sin = freqs.get(seq_len=T, start_pos=0)
    with torch.no_grad():
        q_rot, k_rot = apply_rope(q, k, cos.to(DTYPE), sin.to(DTYPE))

    # Transformers reference — rotary_emb moved to model level in transformers 5.x
    # transformers expects (B, n_heads, T, head_dim) layout
    ref_rope = ref_model.model.rotary_emb
    position_ids = torch.arange(T, device=DEVICE).unsqueeze(0)  # (1, T)
    q_ref_in = q.transpose(1, 2)   # (B, n_heads, T, head_dim)
    k_ref_in = k.transpose(1, 2)
    with torch.no_grad():
        cos_ref, sin_ref = ref_rope(q_ref_in, position_ids)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        q_ref_out, k_ref_out = apply_rotary_pos_emb(q_ref_in, k_ref_in, cos_ref, sin_ref)

    # Our output is (B, T, n_heads, head_dim) — transpose to match
    q_rot_t = q_rot.transpose(1, 2)
    k_rot_t = k_rot.transpose(1, 2)

    q_diff = (q_rot_t - q_ref_out).abs().max().item()
    k_diff = (k_rot_t - k_ref_out).abs().max().item()

    q_status = "PASS" if q_diff < 1e-2 else "FAIL"
    k_status = "PASS" if k_diff < 1e-2 else "FAIL"

    print(f"  Q max |our - ref| = {q_diff:.2e}   [{q_status}]")
    print(f"  K max |our - ref| = {k_diff:.2e}   [{k_status}]")

    del ref_model
    torch.cuda.empty_cache()

    return q_status == "PASS" and k_status == "PASS"


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(cfg: ModelConfig):
    print("\n--- Benchmarks ---")
    print("  RoPE is memory-bound: reads Q and K, writes rotated Q and K\n")

    rope_cfg = cfg.rope_scaling
    freqs = RopeFrequencies(
        head_dim=cfg.head_dim,
        max_seq_len=8192,           # enough for benchmark shapes
        rope_theta=cfg.rope_theta,
        rope_type=rope_cfg.rope_type,
        factor=rope_cfg.factor,
        low_freq_factor=rope_cfg.low_freq_factor,
        high_freq_factor=rope_cfg.high_freq_factor,
        original_max_seq_len=rope_cfg.original_max_position_embeddings,
        device=torch.device(DEVICE),
    )

    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
    ]

    PEAK_BW = 960.0

    print(f"  {'Config':<24} {'Latency':>10}  {'BW GB/s':>10}  {'BW%peak':>8}  {'TC%peak':>8}")
    print(f"  {'-'*24} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for label, B, T in shapes:
        q = torch.randn(B, T, cfg.num_attention_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T, cfg.num_key_value_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE)
        cos, sin = freqs.get(T)
        cos = cos.to(DTYPE)
        sin = sin.to(DTYPE)

        # bytes: read Q, K, cos, sin + write Q_rot, K_rot
        # cos/sin are (T, head_dim) — small relative to Q at large T
        nq = B * T * cfg.num_attention_heads * cfg.head_dim
        nk = B * T * cfg.num_key_value_heads  * cfg.head_dim
        nc = T * cfg.head_dim
        bytes_moved = (nq + nk + nc + nc + nq + nk) * 2  # bfloat16

        # FLOPs: RoPE rotates pairs of elements with 4 mults + 2 adds = 6 FLOPs/element
        # Applied to Q (nq elements) and K (nk elements)
        flops = 6 * (nq + nk)

        lat_ms = bench_fn(lambda: apply_rope(q, k, cos, sin))
        bw     = bandwidth_gb_s(bytes_moved, lat_ms)
        bw_pct = bw / PEAK_BW * 100
        tc_pct = tensor_core_util_pct(flops, lat_ms)

        short = f"B={B} T={T} H={cfg.head_dim}"
        print(f"  {label:<24} {lat_ms:>9.4f}ms  {bw:>10.1f}  {bw_pct:>7.1f}%  {tc_pct:>7.1f}%")
        record("rope", "pytorch", short, lat_ms, bw,
               extra={"batch": B, "seq_len": T, "head_dim": cfg.head_dim,
                      "tc_util_pct": round(tc_pct, 1)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig.llama_3_2_3b()

    print(f"\n{'='*60}")
    print("  Section 6 — RoPE Benchmark")
    print(f"{'='*60}")

    ok = check_correctness(cfg)
    if not ok:
        print("\n[ERROR] Correctness check failed")
        sys.exit(1)

    run_benchmarks(cfg)
    print_results("rope")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
