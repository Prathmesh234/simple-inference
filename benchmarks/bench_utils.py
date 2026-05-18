"""
Shared benchmark utilities.

Every op section records its numbers here so Phase 2 (Triton kernels)
has a clean baseline to compare against.

Results file: benchmarks/results_baseline.json
Schema:
{
  "rmsnorm": {
    "pytorch": [
      {"label": "B=1 T=128 H=3072", "latency_ms": 0.042, "bandwidth_gb_s": 441.2},
      ...
    ]
  },
  "rope": { ... },
  ...
}

When Triton kernels are added, results_triton.json will mirror this schema
so you can diff the two files directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import triton

RESULTS_DIR = Path(__file__).parent
BASELINE_FILE = RESULTS_DIR / "results_baseline.json"
TRITON_FILE   = RESULTS_DIR / "results_triton.json"


# ---------------------------------------------------------------------------
# Core timing primitive
# ---------------------------------------------------------------------------

def bench_fn(fn: Callable, warmup: int = 25, rep: int = 100) -> float:
    """
    Return median latency in milliseconds for a zero-argument callable.

    Uses triton.testing.do_bench which:
      - runs `warmup` iterations (discarded)
      - runs `rep` timed iterations
      - returns the median (robust to GPU clock jitter)

    The callable must capture all its arguments via closure.
    """
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def bandwidth_gb_s(bytes_moved: int, latency_ms: float) -> float:
    """Convert bytes moved + latency to GB/s."""
    return bytes_moved / (latency_ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def _load(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def record(
    op_name: str,
    backend: str,
    label: str,
    latency_ms: float,
    bandwidth_gb_s_val: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    results_file: Path = BASELINE_FILE,
):
    """
    Append one benchmark result.

    op_name     : e.g. "rmsnorm", "attention"
    backend     : "pytorch" | "triton" | "torch_compile"
    label       : human-readable config, e.g. "B=1 T=128 H=3072"
    latency_ms  : median latency from bench_fn()
    bandwidth_gb_s_val : optional, computed by caller who knows bytes_moved
    extra       : any other fields to store (e.g. {"seq_len": 128})
    """
    data = _load(results_file)
    data.setdefault(op_name, {}).setdefault(backend, [])

    entry: Dict[str, Any] = {"label": label, "latency_ms": round(latency_ms, 4)}
    if bandwidth_gb_s_val is not None:
        entry["bandwidth_gb_s"] = round(bandwidth_gb_s_val, 2)
    if extra:
        entry.update(extra)

    # Replace existing entry with same label, or append
    existing = data[op_name][backend]
    for i, e in enumerate(existing):
        if e["label"] == label:
            existing[i] = entry
            break
    else:
        existing.append(entry)

    _save(results_file, data)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(op_name: str, results_file: Path = BASELINE_FILE):
    data = _load(results_file)
    if op_name not in data:
        print(f"No results for '{op_name}' in {results_file}")
        return

    print(f"\n{'='*70}")
    print(f"  Benchmark: {op_name}")
    print(f"{'='*70}")
    print(f"  {'Backend':<16} {'Config':<28} {'Latency':>10}  {'Bandwidth':>12}")
    print(f"  {'-'*16} {'-'*28} {'-'*10}  {'-'*12}")

    for backend, entries in data[op_name].items():
        for e in entries:
            lat  = f"{e['latency_ms']:.4f} ms"
            bw   = f"{e['bandwidth_gb_s']:.1f} GB/s" if "bandwidth_gb_s" in e else "—"
            print(f"  {backend:<16} {e['label']:<28} {lat:>10}  {bw:>12}")

    print(f"{'='*70}")


def compare_backends(op_name: str, results_file: Path = BASELINE_FILE):
    """Print a speedup table: each triton/compile entry vs its pytorch baseline."""
    data = _load(results_file)
    if op_name not in data:
        return

    baseline = {e["label"]: e["latency_ms"] for e in data[op_name].get("pytorch", [])}
    if not baseline:
        return

    print(f"\n  Speedup vs PyTorch baseline ({op_name}):")
    print(f"  {'Backend':<16} {'Config':<28} {'Speedup':>8}")
    print(f"  {'-'*16} {'-'*28} {'-'*8}")
    for backend, entries in data[op_name].items():
        if backend == "pytorch":
            continue
        for e in entries:
            if e["label"] in baseline:
                speedup = baseline[e["label"]] / e["latency_ms"]
                print(f"  {backend:<16} {e['label']:<28} {speedup:>7.2f}x")
