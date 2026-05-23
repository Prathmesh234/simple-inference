"""
benchmarks/run_baseline.py — Section 13

Full baseline benchmark suite. Run this once to establish the PyTorch
baseline before any Triton kernels are added. Re-run after each kernel
swap (Section 14) to measure the delta.

What we measure
---------------
1. Prefill throughput    : tokens/sec at T = 64, 128, 256, 512, 1024
2. Decode throughput     : tokens/sec with KV cache (T=1 per step, B=1 and B=8)
3. Per-op breakdown      : fraction of time in RMSNorm / Attention / MLP / RoPE
                           for one full prefill forward pass (T=512)
4. Memory                : peak VRAM at each prefill length and during decode

All results are appended to benchmarks/results_baseline.json under the key
"run_baseline", and a formatted table is printed to stdout.

Hardware target: RTX 6000 Ada (48 GB VRAM, 960 GB/s bandwidth, 1457 TFLOPS BF16)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn as nn

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from ops.rmsnorm import RMSNorm
from ops.attention import GroupedQueryAttention
from ops.mlp import SwiGLUMLP
from benchmarks.bench_utils import bench_fn, BASELINE_FILE

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"


# ── 1. Prefill throughput ────────────────────────────────────────────────────

def bench_prefill(model: LlamaModel, cfg: ModelConfig) -> list[dict]:
    print(f"\n{'─'*70}")
    print("  1. Prefill throughput  (embed → 28 blocks → norm → lm_head)")
    print(f"{'─'*70}")
    print(f"  {'Seq len':>8}  {'Latency':>10}  {'Tok/sec':>12}  {'VRAM MB':>10}")
    print(f"  {'-------':>8}  {'-------':>10}  {'-------':>12}  {'-------':>10}")

    results = []
    for T in [64, 128, 256, 512, 1024]:
        ids = torch.randint(0, cfg.vocab_size, (1, T), device=DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)

        lat_ms = bench_fn(lambda: model(ids, start_pos=0))

        peak_mb  = torch.cuda.max_memory_allocated(DEVICE) / 1e6
        tok_s    = T / (lat_ms * 1e-3)

        print(f"  {T:>8}  {lat_ms:>9.2f}ms  {tok_s:>12,.0f}  {peak_mb:>9.1f}")
        results.append({"seq_len": T, "latency_ms": round(lat_ms, 3),
                        "tokens_per_sec": round(tok_s, 1),
                        "peak_vram_mb": round(peak_mb, 1)})
    return results


# ── 2. Decode throughput ─────────────────────────────────────────────────────

def bench_decode(model: LlamaModel, cfg: ModelConfig) -> list[dict]:
    print(f"\n{'─'*70}")
    print("  2. Decode throughput  (T=1 per step, KV cache, B=1 and B=8)")
    print(f"{'─'*70}")
    print(f"  {'Batch':>6}  {'Prefill T':>10}  {'Latency/tok':>12}  {'Tok/sec':>10}  {'VRAM MB':>9}")
    print(f"  {'-----':>6}  {'---------':>10}  {'-----------':>12}  {'-------':>10}  {'-------':>9}")

    results = []
    for B in [1, 8]:
        for T_prefill in [128, 512]:
            if T_prefill * B > 4096:
                continue  # skip OOM-likely combos

            kv = KVCache(
                n_layers    = cfg.num_hidden_layers,
                max_batch   = B,
                max_seq_len = T_prefill + 64,
                n_heads_kv  = cfg.num_key_value_heads,
                head_dim    = cfg.head_dim,
                dtype       = DTYPE,
                device      = DEVICE,
            )

            # Prefill to populate cache
            prefill_ids = torch.randint(0, cfg.vocab_size, (B, T_prefill), device=DEVICE)
            with torch.no_grad():
                model(prefill_ids, start_pos=0, kv_cache=kv)

            # Decode: single token step
            decode_ids = torch.randint(0, cfg.vocab_size, (B, 1), device=DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            lat_ms = bench_fn(lambda: model(decode_ids, start_pos=T_prefill, kv_cache=kv))

            peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
            tok_s   = B / (lat_ms * 1e-3)

            print(f"  {B:>6}  {T_prefill:>10}  {lat_ms:>11.3f}ms  {tok_s:>10.1f}  {peak_mb:>8.1f}")
            results.append({"batch": B, "prefill_tokens": T_prefill,
                            "latency_ms": round(lat_ms, 3),
                            "tokens_per_sec": round(tok_s, 1),
                            "peak_vram_mb": round(peak_mb, 1)})
            del kv
    return results


# ── 3. Per-op breakdown ──────────────────────────────────────────────────────

class _Timer:
    """Accumulates wall-clock time for a named op via PyTorch forward hooks."""
    def __init__(self):
        self.times: dict[str, list[float]] = defaultdict(list)
        self._handles = []

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            torch.cuda.synchronize()
            self.times[name].append(self._stop())
        return hook

    def _make_pre_hook(self, name: str):
        def pre_hook(module, inp):
            torch.cuda.synchronize()
            self._start()
        return pre_hook

    def _start(self):
        self._t0 = torch.cuda.Event(enable_timing=True)
        self._t1 = torch.cuda.Event(enable_timing=True)
        self._t0.record()

    def _stop(self) -> float:
        self._t1.record()
        torch.cuda.synchronize()
        return self._t0.elapsed_time(self._t1)

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            tag = None
            if isinstance(module, RMSNorm):
                tag = "rmsnorm"
            elif isinstance(module, GroupedQueryAttention):
                tag = "attention"
            elif isinstance(module, SwiGLUMLP):
                tag = "mlp"

            if tag is not None:
                h1 = module.register_forward_pre_hook(self._make_pre_hook(tag))
                h2 = module.register_forward_hook(self._make_hook(tag))
                self._handles.extend([h1, h2])

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def summary(self) -> dict[str, float]:
        return {k: sum(v) for k, v in self.times.items()}


def bench_op_breakdown(model: LlamaModel, cfg: ModelConfig) -> dict:
    print(f"\n{'─'*70}")
    print("  3. Per-op breakdown  (T=512, B=1, one forward pass)")
    print(f"{'─'*70}")

    T   = 512
    ids = torch.randint(0, cfg.vocab_size, (1, T), device=DEVICE)

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            model(ids, start_pos=0)
    torch.cuda.synchronize()

    # Timed run with hooks
    timer = _Timer()
    timer.register(model)

    with torch.no_grad():
        model(ids, start_pos=0)

    timer.remove()
    op_times = timer.summary()

    # Full forward time for reference
    lat_total = bench_fn(lambda: model(ids, start_pos=0))

    total_accounted = sum(op_times.values())
    print(f"\n  Full forward (bench_fn median) : {lat_total:.2f} ms")
    print(f"  Hook-measured total            : {total_accounted:.2f} ms")
    print(f"\n  {'Op':<14}  {'Time ms':>9}  {'% of forward':>14}")
    print(f"  {'--':<14}  {'-------':>9}  {'------------':>14}")

    result = {}
    for op in ["rmsnorm", "attention", "mlp"]:
        ms  = op_times.get(op, 0.0)
        pct = 100 * ms / lat_total if lat_total > 0 else 0.0
        print(f"  {op:<14}  {ms:>9.2f}  {pct:>13.1f}%")
        result[op] = {"total_ms": round(ms, 3), "pct_of_forward": round(pct, 1)}

    other_ms  = max(0.0, lat_total - total_accounted)
    other_pct = 100 * other_ms / lat_total if lat_total > 0 else 0.0
    print(f"  {'other':<14}  {other_ms:>9.2f}  {other_pct:>13.1f}%")
    result["other"]        = {"total_ms": round(other_ms, 3), "pct_of_forward": round(other_pct, 1)}
    result["full_forward"] = {"total_ms": round(lat_total, 3), "pct_of_forward": 100.0}

    print(f"\n  Note: 'other' = embed + final norm + lm_head + CUDA kernel launch overhead")
    return result


# ── 4. Save + print ──────────────────────────────────────────────────────────

def save_results(prefill: list, decode: list, breakdown: dict):
    data = {}
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            data = json.load(f)

    data["run_baseline"] = {
        "description": "Section 13 full baseline: prefill throughput, decode throughput, per-op breakdown.",
        "hardware":    "RTX 6000 Ada (48 GB, 960 GB/s BW, 1457 TFLOPS BF16)",
        "dtype":       "bfloat16",
        "prefill":     prefill,
        "decode":      decode,
        "op_breakdown_T512": breakdown,
    }

    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved to {BASELINE_FILE}")


def print_summary(prefill: list, decode: list, breakdown: dict):
    print(f"\n{'='*70}")
    print("  BASELINE SUMMARY — Llama 3.2-3B, bfloat16, RTX 6000 Ada")
    print(f"{'='*70}")

    print(f"\n  Prefill throughput:")
    for r in prefill:
        print(f"    T={r['seq_len']:<5}  {r['latency_ms']:>7.2f} ms  "
              f"{r['tokens_per_sec']:>10,.0f} tok/s  "
              f"{r['peak_vram_mb']:>7.0f} MB VRAM")

    print(f"\n  Decode throughput (with KV cache):")
    for r in decode:
        print(f"    B={r['batch']} prefill={r['prefill_tokens']:<5}  "
              f"{r['latency_ms']:>7.3f} ms/tok  "
              f"{r['tokens_per_sec']:>8.1f} tok/s")

    print(f"\n  Per-op time (T=512, one forward pass):")
    for op, v in breakdown.items():
        bar = "█" * int(v["pct_of_forward"] / 3)
        print(f"    {op:<14}  {v['total_ms']:>7.2f} ms  {v['pct_of_forward']:>5.1f}%  {bar}")

    print(f"{'='*70}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("  Section 13 — Full Baseline Benchmark Suite")
    print(f"{'='*70}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    vram_loaded  = torch.cuda.memory_allocated(DEVICE) / 1e9
    print(f"\n  Model  : {total_params/1e9:.3f}B params")
    print(f"  VRAM   : {vram_loaded:.2f} GB after weight load")

    with torch.no_grad():
        prefill_results   = bench_prefill(model, cfg)
        decode_results    = bench_decode(model, cfg)
        breakdown_results = bench_op_breakdown(model, cfg)

    print_summary(prefill_results, decode_results, breakdown_results)
    save_results(prefill_results, decode_results, breakdown_results)
