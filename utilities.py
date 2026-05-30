"""
utilities.py — shared run-metrics helpers.

One place for logging/printing per-run inference metrics so generate.py (and
any other entry point) record numbers in the same format. Each generation run
gets its own numbered file under `metrics/` (run_000.json, run_001.json, ...),
mirroring how the engine iterations keep results so runs can be diffed.
"""

from __future__ import annotations

from typing import Optional
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

# Per-run metrics are written here, one numbered file per generation run.
METRICS_DIR = Path(__file__).parent / "metrics"


def next_run_path(metrics_dir: Path = METRICS_DIR) -> Path:
    """
    Return the next free metrics path: metrics/run_000.json, run_001.json, ...

    The index is one past the highest run_NNN.json already present, so each run
    gets its own file and nothing is ever overwritten.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(metrics_dir.glob("run_*.json"))
    next_idx = 0
    if existing:
        last_stem = existing[-1].stem          # "run_007"
        try:
            next_idx = int(last_stem.split("_")[1]) + 1
        except (IndexError, ValueError):
            next_idx = len(existing)
    return metrics_dir / f"run_{next_idx:03d}.json"


def write_run_metrics(
    prompt: str,
    generated_text: str,
    stats: dict,
    *,
    backend: str,
    model_id: str = "",
    sampling: Optional[dict] = None,
    metrics_dir: Path = METRICS_DIR,
) -> Path:
    """
    Write one run's metrics to the next numbered file under `metrics/` and
    return that path. Only the numbers are recorded — prompt and generated text
    are intentionally omitted; this file is for latency/throughput, not content.

    `stats` is the dict returned by `generate.generate_with_stats`.
    `prompt` / `generated_text` are accepted for call compatibility but not stored.
    """
    record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id":   model_id,
        "backend":    backend,
        "device":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "sampling":   sampling or {},
        "metrics":    stats,
    }
    path = next_run_path(metrics_dir)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def print_metrics_table(stats: dict, backend: str) -> None:
    """Human-readable one-run summary. TTFT and TPOT are the headline numbers."""
    print(f"\n{'='*72}")
    print(f"  Generation metrics   (backend: {backend})")
    print(f"{'='*72}")
    rows = [
        ("TTFT ms (time to 1st tok)", f"{stats.get('ttft_ms', 0.0):.2f}"),
        ("TPOT ms (per output tok)",  f"{stats.get('tpot_ms', 0.0):.3f}"),
        ("decode tok/s (1000/TPOT)",  f"{stats.get('decode_tok_s', 0.0):.1f}"),
        ("prefill ms",                f"{stats.get('prefill_ms', 0.0):.2f}"),
        ("prompt tokens",             f"{stats.get('prompt_tokens', 0)}"),
        ("new tokens",                f"{stats.get('new_tokens', 0)}"),
        ("total tok/s",               f"{stats.get('total_tok_s', 0.0):.1f}"),
        ("total wall ms",             f"{stats.get('total_wall_ms', 0.0):.2f}"),
        ("peak VRAM GB",              f"{stats.get('peak_vram_gb', 0.0):.2f}"),
    ]
    for label, val in rows:
        print(f"  {label:<26}: {val:>12}")
    print(f"{'='*72}")
