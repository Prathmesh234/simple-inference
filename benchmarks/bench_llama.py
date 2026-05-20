"""
Section 10 benchmark: LlamaModel (full prefill)

Runs:
  1. Correctness — our LlamaModel vs transformers AutoModelForCausalLM
     (greedy next-token must agree)
  2. Benchmarks: prefill latency + tokens/sec at T=128, 512, 1024
  3. Records results to benchmarks/results_baseline.json

What this measures
------------------
The full forward pass: embed → 28 × block → norm → lm_head.

Key numbers to understand:
  - Prefill throughput (tokens/sec) = T / latency_s
    This is what determines time-to-first-token.
  - 28 × block estimate from bench_block tells us how close we are to
    that lower bound (overhead from embed/norm/head should be tiny).
  - VRAM at prefill grows with T (activations for all T tokens live in memory).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from benchmarks.bench_utils import bench_fn, record, print_results

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"

PROMPT = "The capital of France is"


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------

def check_correctness(model: LlamaModel, cfg: ModelConfig):
    print("\n--- Correctness check ---")
    print("  Loading transformers reference model...")

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    )
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokens = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    print(tokens.shape)
    T = tokens.shape[1]
    print(f"  Prompt: '{PROMPT}'")
    print(f"  Tokens: {T}")

    with torch.no_grad():
        our_logits = model(tokens, start_pos=0)          # (1, T, vocab_size)
        ref_logits = ref_model(tokens).logits            # (1, T, vocab_size)

    # Check logit agreement at last position (next-token prediction)
    our_next = our_logits[0, -1].argmax().item()
    ref_next = ref_logits[0, -1].argmax().item()
    our_tok  = tokenizer.decode([our_next])
    ref_tok  = tokenizer.decode([ref_next])

    diff      = (our_logits - ref_logits).abs().max().item()
    mean_diff = (our_logits - ref_logits).abs().mean().item()
    greedy_match = our_next == ref_next
    status = "PASS" if greedy_match else "FAIL"

    print(f"\n  max  |our - ref| logit = {diff:.2e}")
    print(f"  mean |our - ref| logit = {mean_diff:.2e}")
    print(f"  our next token  : {our_next} → '{our_tok}'")
    print(f"  ref next token  : {ref_next} → '{ref_tok}'")
    print(f"  greedy match    : {greedy_match}   [{status}]")

    del ref_model
    torch.cuda.empty_cache()

    return greedy_match


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(model: LlamaModel, cfg: ModelConfig):
    print("\n--- Benchmarks ---")
    print("  Measuring prefill: embed → 28 blocks → norm → lm_head\n")

    shapes = [
        ("prefill T=128",  1, 128),
        ("prefill T=512",  1, 512),
        ("prefill T=1024", 1, 1024),
    ]

    print(f"  {'Config':<24} {'Latency':>10}  {'Tok/sec':>10}  {'VRAM MB':>9}")
    print(f"  {'-'*24} {'-'*10}  {'-'*10}  {'-'*9}")

    for label, B, T in shapes:
        token_ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)

        # Measure peak VRAM during forward pass
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize()

        lat_ms = bench_fn(lambda: model(token_ids, start_pos=0))

        peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
        toks_per_sec = (B * T) / (lat_ms * 1e-3)

        short = f"B={B} T={T}"
        print(f"  {label:<24} {lat_ms:>9.2f}ms  {toks_per_sec:>10,.0f}  {peak_vram_mb:>8.0f}")
        record("llama_prefill", "pytorch", short, lat_ms,
               extra={"batch": B, "seq_len": T,
                      "tokens_per_sec": round(toks_per_sec, 1),
                      "peak_vram_mb": round(peak_vram_mb, 1)})

    # Decode estimate: single token forward (T=1)
    print(f"\n  --- Decode estimate (T=1, no KV cache yet) ---")
    B, T = 1, 1
    token_ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
    lat_ms = bench_fn(lambda: model(token_ids, start_pos=0))
    toks_per_sec = 1.0 / (lat_ms * 1e-3)
    short = f"B={B} T={T}"
    print(f"  decode T=1                {lat_ms:>9.2f}ms  {toks_per_sec:>10.1f} tok/s")
    record("llama_prefill", "pytorch", short, lat_ms,
           extra={"batch": B, "seq_len": T,
                  "tokens_per_sec": round(toks_per_sec, 1)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    print(f"\n{'='*60}")
    print("  Section 10 — LlamaModel Full Prefill Benchmark")
    print(f"{'='*60}")

    print("\n  Loading model weights...")
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params/1e9:.3f}B")

    ok = check_correctness(model, cfg)
    if not ok:
        print("\n[ERROR] Greedy next-token mismatch — check weight loading")
        sys.exit(1)

    run_benchmarks(model, cfg)
    print_results("llama_prefill")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
