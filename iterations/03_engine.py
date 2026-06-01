"""
Iteration 03 — Zero-to-Engine  (KV cache + sampling)

State of the engine at this point
-----------------------------------
  ✅ LlamaModel: full 28-layer forward pass
  ✅ Prefill:    processes T prompt tokens in one shot
  ✅ KV cache:   K/V from prefill are stored; decode reuses them
  ✅ Sampling:   temperature=0.7, top_k=50, top_p=0.9
  (everything else identical to 02_kv_cache.py)

What changed from 02
---------------------
The only delta: `argmax` in the decode loop is replaced by `sample()` from
`sampling.py` with:
    temperature = 0.7   (sharpens the distribution slightly vs T=1)
    top_k       = 50    (hard-cap the candidate set to top 50 tokens)
    top_p       = 0.9   (nucleus: further trim to 90% cumulative mass)

Prefill still uses argmax for the very first token so the timing numbers are
comparable to 02 (sampling cost is sub-millisecond vs a full forward pass).
Actually, for consistency the first generated token also uses sampling.

Everything else — workload definitions, KV cache sizing, timing methodology,
results schema — is byte-for-byte identical to 02_kv_cache.py.

Why sampling doesn't really affect latency
-------------------------------------------
The sample() call operates on a (B, vocab_size) tensor after the forward pass.
For vocab_size=128,256 and B≤32, it takes ~0.05 ms — well under 1% of a
typical decode step. The benchmark confirms this, but the numbers should be
indistinguishable from iteration 02.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import env_loader  # noqa: F401  reads .env so USE_TRITON / HF_TOKEN are set before imports
import time
import json
import itertools
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from sampling import sample
import ops.rmsnorm as rmsnorm_mod

# ── constants ────────────────────────────────────────────────────────────────
DEVICE              = "cuda"
DTYPE               = torch.bfloat16
MODEL_ID            = "meta-llama/Llama-3.2-3B"
DEFAULT_DECODE_TOKS = 256
WARMUP              = 2

# Sampling knobs
TEMPERATURE = 0.7
TOP_K       = 50
TOP_P       = 0.9

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


# ── tokenization helpers (identical to 01 / 02) ──────────────────────────────

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


# ── KV-cache generation with sampling ────────────────────────────────────────

@torch.no_grad()
def prefill_and_decode(
    model: LlamaModel,
    kv_cache: KVCache,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
) -> tuple[torch.Tensor, float, float]:
    """
    Prefill + N sampled decode steps.

    Sampling config: temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P.
    Everything else (timing, KV cache usage) is identical to 02_kv_cache.py.
    """
    kv_cache.reset()
    B, T = prompt_ids.shape

    # --- prefill --------------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(prompt_ids, start_pos=0, kv_cache=kv_cache)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # First new token — sampled, not greedy
    next_tok = sample(
        logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P
    ).unsqueeze(-1)  # (B, 1)
    generated = [next_tok]

    # --- decode loop ----------------------------------------------------------
    decode_step_times = []
    pos = T
    for _ in range(n_new_tokens - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits   = model(next_tok, start_pos=pos, kv_cache=kv_cache)
        next_tok = sample(
            logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P
        ).unsqueeze(-1)
        torch.cuda.synchronize()
        decode_step_times.append((time.perf_counter() - t0) * 1000)
        generated.append(next_tok)
        pos += 1

    decode_ms_avg = (
        sum(decode_step_times) / len(decode_step_times) if decode_step_times else 0.0
    )
    return torch.cat(generated, dim=1), prefill_ms, decode_ms_avg


# ── workload runner (identical to 02) ────────────────────────────────────────

def run_workload(
    model: LlamaModel,
    kv_cache: KVCache,
    wl_meta: dict,
    all_token_batches: list[torch.Tensor],
    n_decode_tokens: int,
) -> dict:
    n_total    = wl_meta["n_total"]
    batch_size = wl_meta["batch_size"]
    n_batches  = len(all_token_batches)

    prompt_tokens = sum(b.numel() for b in all_token_batches)
    decode_tokens = n_total * n_decode_tokens

    with torch.no_grad():
        for _ in range(WARMUP):
            prefill_and_decode(model, kv_cache, all_token_batches[0], min(n_decode_tokens, 4))
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    prefill_ms_all = []
    decode_ms_all  = []

    wall_start = time.perf_counter()
    for batch_ids in all_token_batches:
        _, p_ms, d_ms = prefill_and_decode(model, kv_cache, batch_ids, n_decode_tokens)
        prefill_ms_all.append(p_ms)
        decode_ms_all.append(d_ms)
    wall_end = time.perf_counter()

    total_wall_ms    = (wall_end - wall_start) * 1000
    peak_vram_gb     = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    avg_prefill_ms   = sum(prefill_ms_all) / len(prefill_ms_all)
    avg_decode_ms    = sum(decode_ms_all)  / len(decode_ms_all)
    seqs_per_sec     = n_total / (total_wall_ms * 1e-3)
    total_tok_per_s  = (prompt_tokens + decode_tokens) / (total_wall_ms * 1e-3)
    decode_tok_per_s = batch_size / (avg_decode_ms * 1e-3) if avg_decode_ms > 0 else 0.0
    avg_prompt_len   = prompt_tokens / n_total

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

    data["03_engine"] = {
        "description": (
            f"LlamaModel + KVCache + sampling "
            f"(T={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}). "
            "Prefill once, then N sampled decode steps per sequence. "
            f"N comes from each workload, default {DEFAULT_DECODE_TOKS}. "
            "Prompts are NOT truncated."
        ),
        "sampling": {"temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P},
        "workloads": [
            {"label": wl["label"], **r}
            for wl, r in zip(workloads, all_results)
        ],
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def print_table(all_results: list, workloads: list):
    print(f"\n{'='*120}")
    print(
        f"  03 — Zero-to-Engine  "
        f"(KV cache + sampling T={TEMPERATURE} top_k={TOP_K} top_p={TOP_P}, "
        f"bfloat16, RTX 6000 Ada)"
    )
    print(f"{'='*120}")
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
    print(f"{'='*120}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*110}")
    print("  Loading model and tokenizer...")
    backend = "triton (fused kernels)" if rmsnorm_mod.USE_TRITON else "pytorch (reference)"
    print(f"  Backend: {backend}    [override with USE_TRITON=true/false in .env]")
    print(f"  Sampling: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")
    print(f"{'='*110}")

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
    workloads       = []
    tokenized_wls   = []
    decode_per_wl   = []
    max_batch       = 0
    max_prompt_seen = 0

    for wid in WORKLOAD_IDS:
        wl_def = wl_by_id[wid]
        n_total    = wl_def["n_total"]
        batch_size = wl_def["batch_size"]
        n_dec      = int(wl_def.get("decode_tokens", DEFAULT_DECODE_TOKS))
        seqs       = load_sequences(wl_def, n_total)

        batches = []
        for i in range(0, n_total, batch_size):
            chunk = seqs[i : i + batch_size]
            ids   = tokenize_batch(chunk, tokenizer)
            batches.append(ids)
            max_prompt_seen = max(max_prompt_seen, ids.shape[1])

        max_batch = max(max_batch, batch_size)
        workloads.append(wl_def)
        tokenized_wls.append(batches)
        decode_per_wl.append(n_dec)
        print(f"    {wl_def['label']:<22}  {n_total:>5} seqs  →  {len(batches):>3} batches  "
              f"(longest prompt: {max(b.shape[1] for b in batches)} toks, "
              f"decoding {n_dec} new tokens)")

    max_decode_per_wl = max(decode_per_wl)
    max_seq_len       = max_prompt_seen + max_decode_per_wl

    kv_cache = KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = max_batch,
        max_seq_len = max_seq_len,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )

    params      = sum(p.numel() for p in model.parameters())
    vram_loaded = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"\n  Model    : {params/1e9:.3f}B params")
    print(f"  KV cache : {kv_cache}")
    print(f"             (sized for longest prompt {max_prompt_seen} + {max_decode_per_wl} decode tokens)")
    print(f"  VRAM     : {vram_loaded:.2f} GB")
    print(f"  Sampling : temperature={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}")

    all_results = []
    for wl_def, token_batches, n_dec in zip(workloads, tokenized_wls, decode_per_wl):
        print(f"\n  Running: {wl_def['label']}  (decode={n_dec}) ...")
        r = run_workload(model, kv_cache, wl_def, token_batches, n_dec)
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
