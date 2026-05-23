"""
Iteration 01 — Naive Engine (no KV cache, greedy decode)

State of the engine at this point
-----------------------------------
  ✅ LlamaModel: full 28-layer forward pass
  ✅ Prefill:    processes T prompt tokens in one shot
  ✅ Decode:     greedy argmax of the last logit row
  ❌ KV cache:   none — every decode step re-runs the full forward over
                 (prompt + all generated tokens so far)
  ❌ Sampling:   no temperature / top-k / top-p yet (Section 12)

This file is the direct apples-to-apples baseline for `02_kv_cache.py`:
SAME workloads, SAME no-truncation policy, SAME per-workload decode-token
count (default 256), SAME metrics — the only difference is the missing KV
cache. That isolates the cache's contribution.

Why this is slow
----------------
Without a cache, producing the n-th new token costs a full forward over
(T + n) tokens. Total work to produce N decode tokens is therefore
    O(sum_{n=0..N-1} (T + n))  ~=  O(N·T + N²/2)
which is quadratic in the generated length. Iter 02 reduces this to O(N·T)
prefill + O(N) per step (K/V reuse), unlocking real-time generation.

What we measure (matches iter 02 schema)
-----------------------------------------
  avg_prompt_len      : average tokenized prompt length per sequence
  decode_tokens       : new tokens generated per sequence (from JSON, default 256)
  avg_prefill_ms      : the very first forward pass (just the prompt)
  avg_decode_ms       : average per-step decode latency
                        (grows with step index → we report the mean)
  decode_tok_per_s    : batch_size / avg_decode_ms
  total_wall_ms       : end-to-end time for all N sequences
  seqs_per_sec        : N / total_wall_s
  tokens_per_sec      : (prompt + decode) tokens / total_wall_s
  peak_vram_gb        : GPU memory high-water mark during the loop

Prompts are NEVER truncated — long workloads (w7/w8/w9) stay long by design.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import itertools
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel

# ── constants ────────────────────────────────────────────────────────────────
DEVICE              = "cuda"
DTYPE               = torch.bfloat16
MODEL_ID            = "meta-llama/Llama-3.2-3B"
DEFAULT_DECODE_TOKS = 256       # fallback if a workload doesn't override
WARMUP              = 2

RESULTS_FILE   = Path(__file__).parent / "results.json"
SEQUENCES_FILE = Path(__file__).parent / "test_sequences.json"

WORKLOAD_IDS = [
    "w1_single",
    "w2_small_batch",
    "w3_micro_batches",
    "w4_medium",
    "w5_large",
    "w6_xlarge",
    "w7_4k",
    "w8_8k",
    "w9_16k",
]


# ── tokenization helpers (identical to iter 02) ──────────────────────────────

def load_sequences(wl_data: dict, n_total: int) -> list[str]:
    if "sequences" in wl_data:
        seqs = wl_data["sequences"]
    else:
        seqs = wl_data["pool"]
    return list(itertools.islice(itertools.cycle(seqs), n_total))


def tokenize_batch(texts: list[str], tokenizer) -> torch.Tensor:
    """No truncation — long prompts are intentional."""
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    return enc["input_ids"].to(DEVICE)


# ── naive (cache-less) generation ────────────────────────────────────────────

@torch.no_grad()
def prefill_and_decode_naive(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
) -> tuple[torch.Tensor, float, float]:
    """
    No KV cache. Each decode step re-runs the full forward over the
    concatenated (prompt + generated so far). Returns:

        generated:     (B, n_new_tokens) greedy tokens
        prefill_ms:    cost of the first forward (prompt-only)
        decode_ms_avg: average per-step decode latency
    """
    B, T = prompt_ids.shape

    # --- prefill: forward over the prompt alone ---------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(prompt_ids, start_pos=0)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
    seq = torch.cat([prompt_ids, next_tok], dim=1)            # (B, T+1)
    generated = [next_tok]

    # --- decode loop: re-run the full forward each time -------------------
    decode_step_times = []
    for _ in range(n_new_tokens - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # NOTE: start_pos=0 because RoPE positions are derived from the
        # input length each call; without a cache we always pass the full
        # sequence so position 0 is the actual sequence start.
        logits = model(seq, start_pos=0)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        decode_step_times.append((time.perf_counter() - t0) * 1000)
        seq = torch.cat([seq, next_tok], dim=1)
        generated.append(next_tok)

    decode_ms_avg = (
        sum(decode_step_times) / len(decode_step_times) if decode_step_times else 0.0
    )
    return torch.cat(generated, dim=1), prefill_ms, decode_ms_avg


# ── workload runner ──────────────────────────────────────────────────────────

def run_workload(
    model: LlamaModel,
    wl_meta: dict,
    all_token_batches: list[torch.Tensor],
    n_decode_tokens: int,
) -> dict:
    n_total    = wl_meta["n_total"]
    batch_size = wl_meta["batch_size"]
    n_batches  = len(all_token_batches)

    prompt_tokens = sum(b.numel() for b in all_token_batches)
    decode_tokens = n_total * n_decode_tokens

    # ── warm-up ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(WARMUP):
            _, _, _ = prefill_and_decode_naive(
                model, all_token_batches[0], min(n_decode_tokens, 4)
            )
    torch.cuda.synchronize()

    # ── timed run ────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(DEVICE)
    prefill_ms_all = []
    decode_ms_all  = []

    wall_start = time.perf_counter()
    for batch_ids in all_token_batches:
        _, p_ms, d_ms = prefill_and_decode_naive(model, batch_ids, n_decode_tokens)
        prefill_ms_all.append(p_ms)
        decode_ms_all.append(d_ms)
    wall_end = time.perf_counter()

    total_wall_ms     = (wall_end - wall_start) * 1000
    peak_vram_gb      = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    avg_prefill_ms    = sum(prefill_ms_all) / len(prefill_ms_all)
    avg_decode_ms     = sum(decode_ms_all)  / len(decode_ms_all)
    seqs_per_sec      = n_total / (total_wall_ms * 1e-3)
    total_tok_per_s   = (prompt_tokens + decode_tokens) / (total_wall_ms * 1e-3)
    decode_tok_per_s  = batch_size / (avg_decode_ms * 1e-3) if avg_decode_ms > 0 else 0.0
    avg_prompt_len    = prompt_tokens / n_total

    return {
        "n_total":          n_total,
        "batch_size":       batch_size,
        "n_batches":        n_batches,
        "avg_prompt_len":   round(avg_prompt_len, 1),
        "decode_tokens":    n_decode_tokens,
        "avg_prefill_ms":   round(avg_prefill_ms, 2),
        "avg_decode_ms":    round(avg_decode_ms, 3),
        "decode_tok_per_s": round(decode_tok_per_s, 1),
        "total_wall_ms":    round(total_wall_ms, 2),
        "seqs_per_sec":     round(seqs_per_sec, 1),
        "tokens_per_sec":   round(total_tok_per_s, 1),
        "peak_vram_gb":     round(peak_vram_gb, 2),
    }


# ── results I/O ───────────────────────────────────────────────────────────────

def save_results(all_results: list, workloads: list):
    data = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            data = json.load(f)

    data["01_naive_engine"] = {
        "description": (
            "LlamaModel WITHOUT KV cache. Prefill once, then N greedy decode "
            "steps per sequence — each step re-runs the full forward over the "
            "growing sequence. N comes from each workload, default "
            f"{DEFAULT_DECODE_TOKS}. Prompts are NOT truncated."
        ),
        "workloads": [
            {"label": wl["label"], **r}
            for wl, r in zip(workloads, all_results)
        ],
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def print_table(all_results: list, workloads: list):
    print(f"\n{'='*115}")
    print(f"  01 — Naive Engine  (NO KV cache, no truncation, greedy decode, bfloat16, RTX 6000 Ada)")
    print(f"{'='*115}")
    hdr = (f"  {'Workload':<22} {'B':>4}  {'Batches':>7}  {'PromptLen':>9}  {'NewTok':>6}  "
           f"{'Prefill ms':>11}  {'Decode ms':>10}  {'Decode tok/s':>13}  "
           f"{'Wall ms':>9}  {'VRAM GB':>8}")
    print(hdr)
    print(f"  {'-'*22} {'-'*4}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*13}  {'-'*9}  {'-'*8}")
    for wl, r in zip(workloads, all_results):
        print(
            f"  {wl['label']:<22} {r['batch_size']:>4}  {r['n_batches']:>7}  "
            f"{r['avg_prompt_len']:>9.0f}  {r['decode_tokens']:>6}  "
            f"{r['avg_prefill_ms']:>10.1f}ms  {r['avg_decode_ms']:>9.2f}ms  "
            f"{r['decode_tok_per_s']:>13.1f}  "
            f"{r['total_wall_ms']:>8.1f}ms  {r['peak_vram_gb']:>7.2f}"
        )
    print(f"{'='*115}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*115}")
    print("  Loading model and tokenizer...")
    print(f"{'='*115}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    with open(SEQUENCES_FILE) as f:
        seq_data = json.load(f)
    wl_by_id = {wl["id"]: wl for wl in seq_data["workloads"]}

    print(f"\n  Tokenizing all workloads (no truncation)...")
    workloads     = []
    tokenized_wls = []
    decode_per_wl = []

    for wid in WORKLOAD_IDS:
        wl_def = wl_by_id[wid]
        n_total    = wl_def["n_total"]
        batch_size = wl_def["batch_size"]
        n_dec      = int(wl_def.get("decode_tokens", DEFAULT_DECODE_TOKS))
        seqs       = load_sequences(wl_def, n_total)

        batches = []
        for i in range(0, n_total, batch_size):
            chunk = seqs[i : i + batch_size]
            batches.append(tokenize_batch(chunk, tokenizer))

        workloads.append(wl_def)
        tokenized_wls.append(batches)
        decode_per_wl.append(n_dec)
        print(f"    {wl_def['label']:<22}  {n_total:>5} seqs  →  {len(batches):>3} batches  "
              f"(longest prompt: {max(b.shape[1] for b in batches)} toks, "
              f"decoding {n_dec} new tokens)")

    params      = sum(p.numel() for p in model.parameters())
    vram_loaded = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"\n  Model  : {params/1e9:.3f}B params")
    print(f"  VRAM   : {vram_loaded:.2f} GB after weight load (no KV cache buffer in this engine)")

    all_results = []
    for wl_def, token_batches, n_dec in zip(workloads, tokenized_wls, decode_per_wl):
        print(f"\n  Running: {wl_def['label']}  (decode={n_dec}) ...")
        r = run_workload(model, wl_def, token_batches, n_dec)
        all_results.append(r)
        print(
            f"    prefill {r['avg_prefill_ms']:.1f}ms  |  "
            f"decode {r['avg_decode_ms']:.2f}ms/step  |  "
            f"{r['decode_tok_per_s']:.0f} decode tok/s  |  "
            f"wall {r['total_wall_ms']:.1f}ms  |  "
            f"{r['peak_vram_gb']:.2f} GB VRAM"
        )

    print_table(all_results, workloads)
    save_results(all_results, workloads)
    print(f"\n  Results saved to iterations/results.json")
