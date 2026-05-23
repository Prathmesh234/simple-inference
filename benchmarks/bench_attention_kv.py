"""
Section 11 benchmark: GroupedQueryAttention WITH KVCache (decode path).

`bench_attention.py` measures the prefill path: one forward at start_pos=0,
no cache. This file measures the decode path that matters during generation:

  1. Prefill `ctx_len` tokens into the KV cache (start_pos=0, T=ctx_len)
  2. Single-token decode steps at start_pos=ctx_len, ctx_len+1, ...
     → this is what every generated token costs

Sweep across `ctx_len` so you can see decode latency grow with context
(attention is O(ctx_len) per step; the projections are constant cost).

For comparison we also report the prefill cost of the same ctx_len as a
single forward — that's what a NO-cache engine would pay to produce one
token at that position (O(ctx_len) prompt re-processing every step).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from config import ModelConfig
from ops.rope import RopeFrequencies
from ops.attention import GroupedQueryAttention
from model.kv_cache import KVCache
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, record, print_results

DEVICE = "cuda"
DTYPE  = torch.bfloat16


def run_benchmarks(cfg: ModelConfig):
    print("\n--- Decode-step attention latency vs cached context length ---")
    print("  Decode reads ALL cached K/V each step → cost grows linearly with ctx_len")
    print("  Projections (wq/wk/wv/wo) are constant per token; QK^T and softmax@V scale with ctx_len\n")

    MAX_CTX = 8192 + 64

    freqs = RopeFrequencies(
        head_dim             = cfg.head_dim,
        max_seq_len          = MAX_CTX,
        rope_theta           = cfg.rope_theta,
        rope_type            = cfg.rope_scaling.rope_type,
        factor               = cfg.rope_scaling.factor,
        low_freq_factor      = cfg.rope_scaling.low_freq_factor,
        high_freq_factor     = cfg.rope_scaling.high_freq_factor,
        original_max_seq_len = cfg.rope_scaling.original_max_position_embeddings,
        device               = torch.device(DEVICE),
    )

    attn = GroupedQueryAttention(
        hidden_size  = cfg.hidden_size,
        num_heads_q  = cfg.num_attention_heads,
        num_heads_kv = cfg.num_key_value_heads,
        head_dim     = cfg.head_dim,
        rope_freqs   = freqs,
        layer_idx    = 0,
    ).to(DEVICE, DTYPE)
    attn.eval()

    # Random init weights — we only care about latency, not numerical correctness here
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(mean=0.0, std=0.02)

    # One-layer cache that the bench attention writes into
    cache = KVCache(
        n_layers    = 1,
        max_batch   = 1,
        max_seq_len = MAX_CTX,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )

    B = 1
    ctx_lens = [128, 512, 1024, 2048, 4096, 8192]

    # KV bytes touched per decode step = read K + V cached prefix
    def kv_read_bytes(ctx_len: int) -> int:
        # k_full and v_full: each (B, n_kv, ctx_len+1, head_dim) bf16
        return 2 * B * cfg.num_key_value_heads * (ctx_len + 1) * cfg.head_dim * 2

    weight_bytes = (
        cfg.num_attention_heads  * cfg.head_dim * cfg.hidden_size  # wq
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wk
        + cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size  # wv
        + cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim  # wo
    ) * 2  # bfloat16

    PEAK_BW = 960.0  # GB/s for RTX 6000 Ada

    print(f"  {'ctx_len':>8}  {'decode ms':>10}  {'prefill ms':>11}  {'speedup':>8}  {'BW GB/s':>10}  {'BW%peak':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*10}  {'-'*8}")

    for ctx_len in ctx_lens:
        # --- prefill once to populate the cache ---
        cache.reset()
        x_prompt = torch.randn(B, ctx_len, cfg.hidden_size, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            _ = attn(x_prompt, start_pos=0, kv_cache=cache)

        # --- decode step: one new token at position ctx_len ---
        x_dec = torch.randn(B, 1, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        decode_ms = bench_fn(lambda: attn(x_dec, start_pos=ctx_len, kv_cache=cache))

        # Reference: cost of one PREFILL forward at the same length (no-cache baseline)
        # i.e. what you would pay per generated token without a KV cache
        prefill_ms = bench_fn(lambda: attn(x_prompt, start_pos=0, kv_cache=None))

        bytes_moved = weight_bytes + kv_read_bytes(ctx_len)
        bw     = bandwidth_gb_s(bytes_moved, decode_ms)
        bw_pct = bw / PEAK_BW * 100
        speedup = prefill_ms / decode_ms if decode_ms > 0 else float("inf")

        print(
            f"  {ctx_len:>8}  {decode_ms:>9.4f}ms  {prefill_ms:>10.3f}ms  "
            f"{speedup:>7.1f}x  {bw:>10.1f}  {bw_pct:>7.1f}%"
        )

        record("attention_kv", "pytorch", f"decode ctx={ctx_len}", decode_ms, bw,
               extra={"ctx_len": ctx_len, "batch": B,
                      "prefill_equiv_ms": round(prefill_ms, 4),
                      "speedup_vs_prefill": round(speedup, 2)})

        # Don't carry pos forward between sweep points; cache.reset() at top of loop
        # (positional truth is held by the caller via start_pos)

    print("\n  Reading: decode ms stays near-flat for small ctx (projection-bound),")
    print("  then scales linearly once ctx_len dominates the K/V read.")
    print("  speedup = how much you save per token by caching vs re-running prefill.\n")


if __name__ == "__main__":
    cfg = ModelConfig.llama_3_2_3b()

    print(f"\n{'='*70}")
    print("  Section 11 — Attention + KV Cache Benchmark")
    print(f"{'='*70}")

    run_benchmarks(cfg)
    print_results("attention_kv")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
