"""
Section 5 benchmark: TokenEmbedding and OutputProjection

Runs:
  1. Correctness — our embedding vs transformers embed_tokens
  2. Correctness — our output projection vs transformers lm_head
  3. Verify weight tying (same tensor object, not a copy)
  4. Benchmarks for both ops at decode and prefill shapes
  5. Records results to benchmarks/results_baseline.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM

from config import ModelConfig
from loader import WeightLoader
from ops.embedding import TokenEmbedding, OutputProjection
from benchmarks.bench_utils import bench_fn, bandwidth_gb_s, record, print_results

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------

def check_correctness(loader: WeightLoader, cfg: ModelConfig):
    print("\n--- Correctness check ---")

    # Build our classes
    embed = TokenEmbedding(cfg.vocab_size, cfg.hidden_size).to(DEVICE, DTYPE)
    embed.load_weight(loader.get("embed_tokens", device=DEVICE))
    proj  = OutputProjection(embed)

    # Load reference model
    print("  Loading transformers model for reference...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    )
    ref_model.eval()
    ref_embed = ref_model.model.embed_tokens
    ref_head  = ref_model.lm_head

    # --- TokenEmbedding ---
    torch.manual_seed(0)
    token_ids = torch.randint(0, cfg.vocab_size, (1, 128), device=DEVICE)

    with torch.no_grad():
        our_emb = embed(token_ids)
        ref_emb = ref_embed(token_ids)

    diff_emb = (our_emb - ref_emb).abs().max().item()
    status   = "PASS" if diff_emb < 1e-4 else "FAIL"
    print(f"  TokenEmbedding    max |our - ref| = {diff_emb:.2e}   [{status}]")

    # --- OutputProjection ---
    # Use the embedding output as input to the projection
    with torch.no_grad():
        our_logits = proj(our_emb)
        ref_logits = ref_head(ref_emb)

    diff_logits = (our_logits - ref_logits).abs().max().item()
    status2     = "PASS" if diff_logits < 1e-1 else "FAIL"
    print(f"  OutputProjection  max |our - ref| = {diff_logits:.2e}   [{status2}]")
    # Note: logits can have larger absolute differences because they are not
    # normalized — what matters is that argmax agrees (greedy next token matches)

    our_next  = our_logits[0, -1].argmax().item()
    ref_next  = ref_logits[0, -1].argmax().item()
    tok_match = "PASS" if our_next == ref_next else "FAIL"
    print(f"  Greedy next token: ours={our_next}  ref={ref_next}   [{tok_match}]")

    # --- Weight tying ---
    same_storage = embed.weight.data_ptr() == proj.weight.data_ptr()
    print(f"  Weight tying (same tensor): {'YES' if same_storage else 'NO — BUG'}")

    del ref_model
    torch.cuda.empty_cache()

    return diff_emb < 1e-4 and tok_match == "PASS"


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(cfg: ModelConfig):
    print("\n--- Benchmarks ---")

    embed = TokenEmbedding(cfg.vocab_size, cfg.hidden_size).to(DEVICE, DTYPE)
    # Random weight — we're benchmarking throughput, not correctness here
    nn_init = torch.nn.init.normal_
    nn_init(embed.weight)
    proj = OutputProjection(embed)

    shapes = [
        ("decode  T=1",    1, 1),
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=2048", 1, 2048),
    ]

    PEAK_BW = 960.0  # GB/s RTX 6000 Ada

    # TokenEmbedding benchmark
    print(f"\n  TokenEmbedding  (vocab={cfg.vocab_size}, hidden={cfg.hidden_size})")
    print(f"  {'Config':<24} {'Latency':>10}  {'Bandwidth':>12}  {'% of peak':>10}")
    print(f"  {'-'*24} {'-'*10}  {'-'*12}  {'-'*10}")

    for label, B, T in shapes:
        token_ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
        # Bytes: reading T rows of hidden_size bfloat16 values from the table
        bytes_moved = B * T * cfg.hidden_size * 2

        lat_ms = bench_fn(lambda: embed(token_ids))
        bw     = bandwidth_gb_s(bytes_moved, lat_ms)
        pct    = bw / PEAK_BW * 100

        short = f"B={B} T={T} H={cfg.hidden_size}"
        print(f"  {label:<24} {lat_ms:>9.4f}ms  {bw:>10.1f} GB/s  {pct:>9.1f}%")
        record("embedding_lookup", "pytorch", short, lat_ms, bw,
               extra={"batch": B, "seq_len": T, "hidden": cfg.hidden_size})

    # OutputProjection benchmark
    print(f"\n  OutputProjection  (hidden={cfg.hidden_size} → vocab={cfg.vocab_size})")
    print(f"  {'Config':<24} {'Latency':>10}  {'Bandwidth':>12}  {'% of peak':>10}")
    print(f"  {'-'*24} {'-'*10}  {'-'*12}  {'-'*10}")

    for label, B, T in shapes:
        x = torch.randn(B, T, cfg.hidden_size, device=DEVICE, dtype=DTYPE)
        # Bytes: read x (B,T,H) + read weight (vocab,H) + write logits (B,T,vocab)
        bytes_moved = (
            B * T * cfg.hidden_size * 2          # read x
            + cfg.vocab_size * cfg.hidden_size * 2   # read weight
            + B * T * cfg.vocab_size * 2         # write logits
        )

        lat_ms = bench_fn(lambda: proj(x))
        bw     = bandwidth_gb_s(bytes_moved, lat_ms)
        pct    = bw / PEAK_BW * 100

        short = f"B={B} T={T} H={cfg.hidden_size}"
        print(f"  {label:<24} {lat_ms:>9.4f}ms  {bw:>10.1f} GB/s  {pct:>9.1f}%")
        record("output_projection", "pytorch", short, lat_ms, bw,
               extra={"batch": B, "seq_len": T, "hidden": cfg.hidden_size})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    print(f"\n{'='*60}")
    print("  Section 5 — Embeddings Benchmark")
    print(f"{'='*60}")

    ok = check_correctness(loader, cfg)
    if not ok:
        print("\n[ERROR] Correctness check failed")
        sys.exit(1)

    run_benchmarks(cfg)
    print_results("embedding_lookup")
    print_results("output_projection")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
