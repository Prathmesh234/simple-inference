"""
Profiling Level 1 — Basic torch.profiler

The simplest useful profiler: one context manager, one table.

Key learning from the HF blog: the first few iterations carry GPU startup
overhead — cuDNN/cuBLAS plan selection, CUDA context initialization, Triton
kernel JIT compilation. Always warm up before profiling, or you measure
initialization instead of steady-state.

What you get:
  - CPU time   : time on the host thread (Python dispatch + driver calls)
  - CUDA time  : actual GPU execution time per kernel
  - self time  : time excluding calls to child ops (identifies the true hot op)

Run:
    python -m profiling.01_basic
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.profiler import profile, ProfilerActivity

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"


def profile_prefill(model: LlamaModel, cfg: ModelConfig, seq_len: int = 128):
    """Wrap a single prefill in the minimal profiler and print the op table."""
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=DEVICE)

    # Warmup — critical.
    # Without warmup, cuBLAS algorithm selection and Triton JIT cost dominate,
    # making every matmul look ~10-100× slower than it actually is at steady state.
    print(f"  Warming up prefill (T={seq_len})...")
    with torch.no_grad():
        for _ in range(5):
            model(ids, start_pos=0)
    torch.cuda.synchronize()

    print(f"  Profiling prefill (T={seq_len})...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        with torch.no_grad():
            model(ids, start_pos=0)

    print(f"\n  --- Prefill T={seq_len}: top 20 ops by CUDA time ---")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    print(f"\n  --- Prefill T={seq_len}: top 10 ops by CPU self-time ---")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=10,
    ))


def profile_decode(model: LlamaModel, cfg: ModelConfig, t_prefill: int = 128):
    """Profile a single decode step (T=1 with KV cache) after a prefill."""
    kv = KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = 1,
        max_seq_len = t_prefill + 16,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )

    # Populate KV cache with a prefill first
    prefill_ids = torch.randint(0, cfg.vocab_size, (1, t_prefill), device=DEVICE)
    with torch.no_grad():
        model(prefill_ids, start_pos=0, kv_cache=kv)

    decode_ids = torch.randint(0, cfg.vocab_size, (1, 1), device=DEVICE)

    # Warmup decode steps
    print(f"  Warming up decode (T=1, prefill_len={t_prefill})...")
    with torch.no_grad():
        for _ in range(5):
            model(decode_ids, start_pos=t_prefill, kv_cache=kv)
    torch.cuda.synchronize()

    print(f"  Profiling decode step...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        with torch.no_grad():
            model(decode_ids, start_pos=t_prefill, kv_cache=kv)

    print(f"\n  --- Decode T=1 (prefill={t_prefill}): top 20 ops by CUDA time ---")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    # CUDA self-time separates "this op's kernel" from "ops it calls"
    # (important for understanding whether attention or MLP dominates decode)
    print(f"\n  --- Decode T=1: top 10 by self_cuda_time ---")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=10,
    ))

    del kv


def main():
    print(f"\n{'='*70}")
    print("  Profiling Level 1 — Basic torch.profiler")
    print(f"{'='*70}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model  = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    vram_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"\n  Model loaded — {vram_gb:.2f} GB VRAM\n")

    print(f"{'─'*70}")
    print("  1. Prefill")
    print(f"{'─'*70}")
    profile_prefill(model, cfg, seq_len=512)

    print(f"\n{'─'*70}")
    print("  2. Decode")
    print(f"{'─'*70}")
    profile_decode(model, cfg, t_prefill=128)

    print(f"\n{'='*70}")
    print("  What to look for:")
    print("    - aten::mm / aten::bmm  → raw matmul kernels (MLP + attention)")
    print("    - aten::scaled_dot_product_attention → fused SDPA (prefill attention)")
    print("    - High self_cpu_time vs cuda_time → CPU-bound dispatch overhead")
    print("    - Prefill: matmuls dominate (compute-bound)")
    print("    - Decode:  KV cache reads + attention dominate (memory-bound)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
