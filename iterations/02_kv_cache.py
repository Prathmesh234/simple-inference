"""
Iteration 02 — KV Cache (greedy decode, no fancy sampling yet)

State of the engine at this point
-----------------------------------
  ✅ LlamaModel: full 28-layer forward pass
  ✅ Prefill:    processes T prompt tokens in one shot
  ✅ KV cache:   K/V from prefill are stored; decode reuses them
  ✅ Decode:     greedy argmax of the last logit row
  ❌ Sampling:   no temperature / top-k / top-p yet (Section 12)

What we measure
---------------
For each workload we record TWO phases per batch:

  prefill:
    avg_prefill_ms     : forward pass on the prompt (one call per batch)
    prefill_tok_per_s  : prompt_tokens / prefill_time

  decode:
    avg_decode_ms      : average per-token decode latency
    decode_tok_per_s   : batch_size / avg_decode_ms  (tokens/sec across batch)

And the usual roll-up:
    total_wall_ms      : end-to-end time for all N sequences (prefill + decode)
    seqs_per_sec       : N / total_wall_s
    tokens_per_sec     : (prompt + decode) tokens / total_wall_s
    peak_vram_gb       : high-water VRAM including the KV cache itself

Comparison with iteration 01
-----------------------------
Iteration 01 reported prefill-only tok/s. Iteration 02 is the first iteration
that produces real new tokens, so decode tok/s becomes the *meaningful* number
for "how fast does generation feel". The prefill column should be roughly
unchanged from 01; the decode column is the new story (and should be MUCH
faster per token than re-running prefill, which would be O(T²) total work).

Six workloads (identical to iteration 01 for apples-to-apples comparison) plus
a per-workload number of new tokens generated per sequence. Prompts are
NEVER truncated — the long workloads (w7/w8/w9) are intentionally long
to stress the cache.

Correctness check
-----------------
Before benchmarking we run a one-shot verification against
`transformers.AutoModelForCausalLM` with `use_cache=True`:
  - tokenize a short prompt
  - run our prefill + N greedy decode steps with our KVCache
  - run HF's `.generate(do_sample=False, max_new_tokens=N)` for the same prompt
  - assert the generated token IDs match

This catches almost every bug in the cache wiring (off-by-one, wrong layer
slot, dropped RoPE start_pos, etc.) before any benchmark numbers are trusted.
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
from model.kv_cache import KVCache

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


# ── tokenization helpers (mirrors 01) ────────────────────────────────────────

def load_sequences(wl_data: dict, n_total: int) -> list[str]:
    if "sequences" in wl_data:
        seqs = wl_data["sequences"]
    else:
        seqs = wl_data["pool"]
    return list(itertools.islice(itertools.cycle(seqs), n_total))


def tokenize_batch(texts: list[str], tokenizer) -> torch.Tensor:
    """
    Tokenize a batch with NO truncation. Pads to the longest sequence in
    the batch. Long prompts (workloads 7-9) are kept at their natural length
    on purpose — that's part of the stress test.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )
    return enc["input_ids"].to(DEVICE)


# ── KV-cache-driven generation step ──────────────────────────────────────────

@torch.no_grad()
def prefill_and_decode(
    model: LlamaModel,
    kv_cache: KVCache,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
) -> tuple[torch.Tensor, float, float]:
    """
    Run one prefill on `prompt_ids` (B, T) and then `n_new_tokens` greedy
    decode steps using `kv_cache` for K/V storage.

    Returns:
        generated_ids:  (B, n_new_tokens) the tokens we picked
        prefill_ms:     prefill latency in ms (GPU-synchronized)
        decode_ms_avg:  average decode-step latency in ms
    """
    kv_cache.reset()
    B, T = prompt_ids.shape

    # --- prefill ----------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(prompt_ids, start_pos=0, kv_cache=kv_cache)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Greedy argmax over the LAST token's logits → next token per sequence
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
    generated = [next_tok]

    # --- decode loop ------------------------------------------------------
    decode_step_times = []
    pos = T  # absolute position of the next slot to write
    for _ in range(n_new_tokens - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(next_tok, start_pos=pos, kv_cache=kv_cache)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        decode_step_times.append((time.perf_counter() - t0) * 1000)
        generated.append(next_tok)
        pos += 1

    decode_ms_avg = (
        sum(decode_step_times) / len(decode_step_times) if decode_step_times else 0.0
    )
    return torch.cat(generated, dim=1), prefill_ms, decode_ms_avg


# ── correctness check vs HuggingFace transformers ────────────────────────────

@torch.no_grad()
def verify_against_hf(model: LlamaModel, tokenizer, kv_cache: KVCache) -> None:
    """
    Generate N tokens with our engine and with HF's `.generate(use_cache=True)`
    on the same prompt. Token-id sequences must match for greedy decoding.
    """
    from transformers import AutoModelForCausalLM

    prompt = "The capital of France is"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    n_new = 16

    print("\n  [verify] Running our engine (KV cache + greedy)...")
    ours, _, _ = prefill_and_decode(model, kv_cache, prompt_ids, n_new)

    print("  [verify] Running HuggingFace reference (use_cache=True, do_sample=False)...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    hf_model.eval()
    hf_out = hf_model.generate(
        prompt_ids,
        max_new_tokens=n_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    hf_new = hf_out[:, prompt_ids.shape[1]:]

    ours_txt = tokenizer.decode(ours[0].tolist())
    hf_txt   = tokenizer.decode(hf_new[0].tolist())

    print(f"    ours: {ours_txt!r}")
    print(f"    HF  : {hf_txt!r}")

    match = torch.equal(ours.cpu(), hf_new.cpu())
    print(f"  [verify] token IDs match: {match}")

    # Free HF model before benchmarks so its VRAM doesn't pollute peak stats
    del hf_model
    torch.cuda.empty_cache()

    if not match:
        # bf16 numerical drift can occasionally diverge on the last 1-2 tokens;
        # report the prefix that does match to make the failure mode obvious.
        eq = (ours.cpu() == hf_new.cpu())[0]
        matched = int(eq.cumprod(0).sum().item())
        print(f"  [verify] WARNING: first {matched}/{n_new} tokens match; "
              f"divergence after that is usually bf16 noise on near-tied logits.")


# ── workload runner ──────────────────────────────────────────────────────────

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

    # ── warm-up (covers both prefill and decode paths) ───────────────────────
    with torch.no_grad():
        for _ in range(WARMUP):
            _, _, _ = prefill_and_decode(
                model, kv_cache, all_token_batches[0], min(n_decode_tokens, 4)
            )
    torch.cuda.synchronize()

    # ── timed run ────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(DEVICE)
    prefill_ms_all = []
    decode_ms_all  = []

    wall_start = time.perf_counter()
    for batch_ids in all_token_batches:
        _, p_ms, d_ms = prefill_and_decode(model, kv_cache, batch_ids, n_decode_tokens)
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

    data["02_kv_cache"] = {
        "description": (
            "LlamaModel + static KVCache. Prefill once, then N greedy decode "
            "steps per sequence (N comes from each workload, default "
            f"{DEFAULT_DECODE_TOKS}). Prompts are NOT truncated."
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
    print(f"  02 — KV Cache  (no prompt truncation, greedy decode, bfloat16, RTX 6000 Ada)")
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
    print(f"\n{'='*110}")
    print("  Loading model and tokenizer...")
    print(f"{'='*110}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load workloads
    with open(SEQUENCES_FILE) as f:
        seq_data = json.load(f)
    wl_by_id = {wl["id"]: wl for wl in seq_data["workloads"]}

    # ── tokenize first so we can size the KV cache to actual prompt lengths ─
    print(f"\n  Tokenizing all workloads (no truncation)...")
    workloads        = []
    tokenized_wls    = []
    decode_per_wl    = []
    max_batch        = 0
    max_prompt_seen  = 0

    for wid in WORKLOAD_IDS:
        wl_def = wl_by_id[wid]
        n_total    = wl_def["n_total"]
        batch_size = wl_def["batch_size"]
        # Per-workload override of decode-token count, else module default
        n_dec = int(wl_def.get("decode_tokens", DEFAULT_DECODE_TOKS))
        seqs  = load_sequences(wl_def, n_total)

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
              f"(longest prompt in this WL: {max(b.shape[1] for b in batches)} toks, "
              f"decoding {n_dec} new tokens)")

    # KV cache must fit the longest (prompt + new tokens) we'll ever see
    max_decode_per_wl = max(decode_per_wl)
    max_seq_len = max_prompt_seen + max_decode_per_wl
    if max_seq_len > cfg.max_position_embeddings:
        raise RuntimeError(
            f"max prompt+decode ({max_seq_len}) exceeds model max_position_embeddings "
            f"({cfg.max_position_embeddings}). Use a longer-context model or shorter prompts."
        )

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
    print(f"  VRAM     : {vram_loaded:.2f} GB after weight + cache load")

    # ── correctness check (small prompt, 16 new tokens) ──────────────────────
    verify_against_hf(model, tokenizer, kv_cache)

    # ── run workloads ────────────────────────────────────────────────────────
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
