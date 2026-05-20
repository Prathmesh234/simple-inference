"""
Iteration 01 — Naive Engine (no KV cache, no sampling)

State of the engine at this point
-----------------------------------
  ✅ LlamaModel: full 28-layer forward pass
  ✅ Prefill: processes T tokens in one shot
  ❌ KV cache: not yet — every decode step re-runs the full sequence
  ❌ Sampling: not yet — we measure raw forward-pass throughput only

What we measure
---------------
For each workload we record:
  - avg_batch_ms    : average forward pass latency per micro-batch
  - total_wall_ms   : total time to process ALL N sequences
  - seqs_per_sec    : N / total_wall_s
  - tokens_per_sec  : N * avg_seq_len / total_wall_s
  - peak_vram_gb    : GPU memory high-water mark during the loop

Six workloads (identical across all iterations/ files)
---------------------------------------------------------
  #  N total   batch_size   micro-batches   note
  1      1          1            1          single-sequence baseline
  2      8          8            1          small batch, one pass
  3     24          3            8          8 micro-batches of 3
  4    128         16            8          8 micro-batches of 16
  5   1024         32           32          32 micro-batches of 32
  6   2048         32           64          64 micro-batches of 32

Sequences loaded from iterations/test_sequences.json.
Each workload uses real prompts from diverse challenging domains.
Sequences are tokenized and padded to the longest in each batch.
Pool sequences are cycled to reach n_total for large workloads.
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
DEVICE      = "cuda"
DTYPE       = torch.bfloat16
MODEL_ID    = "meta-llama/Llama-3.2-3B"
MAX_SEQ_LEN = 256    # truncate all sequences to this length
WARMUP      = 3      # warm-up passes before timing

RESULTS_FILE   = Path(__file__).parent / "results.json"
SEQUENCES_FILE = Path(__file__).parent / "test_sequences.json"


# ── workload definitions ─────────────────────────────────────────────────────
# Must match ids in test_sequences.json
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


# ── tokenization helpers ──────────────────────────────────────────────────────

def load_sequences(wl_data: dict, n_total: int) -> list[str]:
    """
    Return exactly n_total strings for this workload.
    Uses 'sequences' if present, else cycles 'pool'.
    """
    if "sequences" in wl_data:
        seqs = wl_data["sequences"]
    else:
        seqs = wl_data["pool"]
    # cycle through pool to reach n_total
    return list(itertools.islice(itertools.cycle(seqs), n_total))


def tokenize_batch(texts: list[str], tokenizer, max_len: int) -> torch.Tensor:
    """
    Tokenize a list of strings and return (B, T) padded token IDs on DEVICE.
    Pads to the longest sequence in the batch (capped at max_len).
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    return enc["input_ids"].to(DEVICE)


# ── workload runner ───────────────────────────────────────────────────────────

def run_workload(
    model: LlamaModel,
    cfg: ModelConfig,
    wl_meta: dict,
    all_token_batches: list[torch.Tensor],
) -> dict:
    """
    Run one workload. all_token_batches is a pre-tokenized list of (B, T) tensors.
    Returns a dict of timing/memory stats.
    """
    n_total    = wl_meta["n_total"]
    batch_size = wl_meta["batch_size"]
    n_batches  = len(all_token_batches)

    # Total token count (accounts for variable-length padding)
    total_tokens = sum(b.numel() for b in all_token_batches)

    # ── warm-up ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(all_token_batches[0], start_pos=0)
    torch.cuda.synchronize()

    # ── timed run ────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(DEVICE)
    batch_latencies = []

    wall_start = time.perf_counter()
    with torch.no_grad():
        for batch_ids in all_token_batches:
            t0 = time.perf_counter()
            _ = model(batch_ids, start_pos=0)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_latencies.append((t1 - t0) * 1000)
    wall_end = time.perf_counter()

    total_wall_ms  = (wall_end - wall_start) * 1000
    peak_vram_gb   = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    avg_batch_ms   = sum(batch_latencies) / len(batch_latencies)
    seqs_per_sec   = n_total / (total_wall_ms * 1e-3)
    tokens_per_sec = total_tokens / (total_wall_ms * 1e-3)
    avg_seq_len    = total_tokens / n_total

    return {
        "n_total":        n_total,
        "batch_size":     batch_size,
        "n_batches":      n_batches,
        "avg_seq_len":    round(avg_seq_len, 1),
        "avg_batch_ms":   round(avg_batch_ms, 2),
        "total_wall_ms":  round(total_wall_ms, 2),
        "seqs_per_sec":   round(seqs_per_sec, 1),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_vram_gb":   round(peak_vram_gb, 2),
    }


# ── results I/O ───────────────────────────────────────────────────────────────

def save_results(all_results: list, workloads: list):
    data = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            data = json.load(f)

    data["01_naive_engine"] = {
        "description": "LlamaModel prefill only, no KV cache, no sampling. Real prompts.",
        "max_seq_len": MAX_SEQ_LEN,
        "workloads": [
            {"label": wl["label"], **r}
            for wl, r in zip(workloads, all_results)
        ],
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def print_table(all_results: list, workloads: list):
    print(f"\n{'='*95}")
    print(f"  01 — Naive Engine  (real prompts, pad-to-longest, bfloat16, RTX 6000 Ada)")
    print(f"{'='*95}")
    hdr = (f"  {'Workload':<22} {'B':>4}  {'Batches':>7}  {'AvgLen':>7}  "
           f"{'Avg/batch':>10}  {'Wall ms':>9}  {'Seq/s':>8}  {'Tok/s':>10}  {'VRAM GB':>8}")
    print(hdr)
    print(f"  {'-'*22} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*8}")
    for wl, r in zip(workloads, all_results):
        print(
            f"  {wl['label']:<22} {r['batch_size']:>4}  {r['n_batches']:>7}  "
            f"{r['avg_seq_len']:>7.0f}  "
            f"{r['avg_batch_ms']:>9.1f}ms  {r['total_wall_ms']:>8.1f}ms  "
            f"{r['seqs_per_sec']:>8.1f}  {r['tokens_per_sec']:>10,.0f}  "
            f"{r['peak_vram_gb']:>7.2f}"
        )
    print(f"{'='*95}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*95}")
    print("  Loading model and tokenizer...")
    print(f"{'='*95}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    params      = sum(p.numel() for p in model.parameters())
    vram_loaded = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"  Model  : {params/1e9:.3f}B params")
    print(f"  VRAM   : {vram_loaded:.2f} GB after weight load")

    # Load workload definitions from JSON
    with open(SEQUENCES_FILE) as f:
        seq_data = json.load(f)

    wl_by_id = {wl["id"]: wl for wl in seq_data["workloads"]}

    print(f"\n  Tokenizing all workloads (max_len={MAX_SEQ_LEN})...")
    workloads      = []
    tokenized_wls  = []

    for wid in WORKLOAD_IDS:
        wl_def = wl_by_id[wid]
        n_total    = wl_def["n_total"]
        batch_size = wl_def["batch_size"]

        seqs = load_sequences(wl_def, n_total)

        # Tokenize into micro-batches
        batches = []
        for i in range(0, n_total, batch_size):
            chunk = seqs[i : i + batch_size]
            batches.append(tokenize_batch(chunk, tokenizer, MAX_SEQ_LEN))

        workloads.append(wl_def)
        tokenized_wls.append(batches)
        print(f"    {wl_def['label']:<22}  {n_total:>5} seqs  →  {len(batches)} batches")

    all_results = []
    for wl_def, token_batches in zip(workloads, tokenized_wls):
        print(f"\n  Running: {wl_def['label']} ...")
        r = run_workload(model, cfg, wl_def, token_batches)
        all_results.append(r)
        print(
            f"    avg batch {r['avg_batch_ms']:.1f}ms  |  "
            f"total {r['total_wall_ms']:.1f}ms  |  "
            f"{r['seqs_per_sec']:.1f} seq/s  |  "
            f"{r['tokens_per_sec']:,.0f} tok/s  |  "
            f"{r['peak_vram_gb']:.2f} GB VRAM"
        )

    print_table(all_results, workloads)
    save_results(all_results, workloads)
    print(f"\n  Results saved to iterations/results.json")

