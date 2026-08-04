"""Combine megabenchmark state results into JSON, CSV, Markdown, and console tables."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CSV_FIELDS = (
    "state_id",
    "state_label",
    "workload_id",
    "workload_label",
    "request_count",
    "prompt_tokens_total",
    "generated_tokens",
    "wall_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "inter_token_p50_ms",
    "inter_token_p95_ms",
    "output_tokens_per_second",
    "steady_decode_tokens_per_second",
    "request_per_second",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "kv_storage_gb",
    "decode_backend",
    "cuda_graphs_active",
    "output_sha256",
    "parity_with_reference",
    "common_output_prefix_fraction",
)

REFERENCE_STATE_BY_WORKLOAD = {
    "single_request": "00_naive_no_kv_pytorch",
    "batch8": "01_static_kv_pytorch",
    "decode_heavy": "01_static_kv_pytorch",
    "continuous_mixed": "05_contiguous_continuous_batching",
    "long_prefill": "05_contiguous_continuous_batching",
    "capacity_mixed": "05_contiguous_continuous_batching",
}


def common_prefix_fraction(reference: list[list[int]], candidate: list[list[int]]) -> float:
    matched = 0
    total = 0
    for expected, actual in zip(reference, candidate):
        total += len(expected)
        for left, right in zip(expected, actual):
            if left != right:
                break
            matched += 1
    return matched / total if total else 1.0


def load_state_results(results_dir: Path) -> list[dict[str, Any]]:
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    results = []
    for state in manifest["states"]:
        path = results_dir / "states" / f"{state['id']}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        results.append(data)
    return results


def flatten(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: dict[str, list[list[int]]] = {}
    for state in states:
        if state.get("status") != "ok":
            continue
        for workload in state.get("workloads", []):
            reference_state = REFERENCE_STATE_BY_WORKLOAD.get(workload["id"])
            if state["state"]["id"] == reference_state:
                references[workload["id"]] = [
                    request["output_token_ids"]
                    for request in workload["requests"]
                ]

    rows = []
    for state in states:
        if state.get("status") != "ok":
            continue
        for workload in state.get("workloads", []):
            outputs = [
                request["output_token_ids"] for request in workload["requests"]
            ]
            reference = references.get(workload["id"])
            exact = outputs == reference if reference is not None else None
            engine = workload.get("engine", {})
            kv = engine.get("kv", {})
            memory = workload["memory"]
            rows.append(
                {
                    "state_id": state["state"]["id"],
                    "state_label": state["state"]["label"],
                    "workload_id": workload["id"],
                    "workload_label": workload["label"],
                    "request_count": workload["request_count"],
                    "prompt_tokens_total": workload["prompt_tokens"]["total"],
                    "generated_tokens": workload["generated_tokens"],
                    "wall_ms": workload["wall_ms"],
                    "ttft_p50_ms": workload["ttft_ms"]["p50"],
                    "ttft_p95_ms": workload["ttft_ms"]["p95"],
                    "inter_token_p50_ms": workload["inter_token_ms"]["p50"],
                    "inter_token_p95_ms": workload["inter_token_ms"]["p95"],
                    "output_tokens_per_second": workload["output_tokens_per_second"],
                    "steady_decode_tokens_per_second": workload[
                        "steady_decode_tokens_per_second"
                    ],
                    "request_per_second": workload["request_per_second"],
                    "peak_allocated_gb": memory["max_allocated_bytes"] / 1e9,
                    "peak_reserved_gb": memory["max_reserved_bytes"] / 1e9,
                    "kv_storage_gb": kv.get("storage_bytes", 0) / 1e9,
                    "decode_backend": engine.get("decode_backend"),
                    "cuda_graphs_active": engine.get("cuda_graphs_active"),
                    "output_sha256": workload["output_sha256"],
                    "parity_with_reference": exact,
                    "common_output_prefix_fraction": (
                        common_prefix_fraction(reference, outputs)
                        if reference is not None
                        else None
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(states: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    successful = [state for state in states if state.get("status") == "ok"]
    failed = [state for state in states if state.get("status") != "ok"]
    lines = [
        "# Inference Engine Megabenchmark",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    if successful:
        environment = successful[0]["environment"]
        gpu = environment.get("gpu", {})
        lines.extend(
            [
                "## Environment",
                "",
                f"- GPU: {gpu.get('name', 'unknown')} ({gpu.get('total_memory_bytes', 0) / 1e9:.2f} GB)",
                f"- Compute capability: {gpu.get('compute_capability', 'unknown')}",
                f"- PyTorch: {environment.get('torch')} / CUDA: {environment.get('cuda_runtime')}",
                f"- Git commit: `{environment.get('git_commit')}` (dirty: {environment.get('git_dirty')})",
                "",
            ]
        )

    lines.extend(
        [
            "## State Matrix",
            "",
            "| State | Optimizations | Flags | Status |",
            "|---|---|---|---|",
        ]
    )
    for state in states:
        spec = state.get("state", {})
        flags = ", ".join(
            f"{key}={value}" for key, value in spec.get("env", {}).items()
        )
        optimizations = ", ".join(spec.get("optimizations", [])) or "none"
        lines.append(
            f"| `{spec.get('id', 'unknown')}` {spec.get('label', '')} | "
            f"{optimizations} | `{flags}` | {state.get('status')} |"
        )

    lines.extend(
        [
            "",
            "## Workload Results",
            "",
            "| State | Workload | Wall ms | TTFT p50 ms | TPOT p50 ms | Output tok/s | Decode tok/s | Peak GB | KV GB | Exact parity | Prefix parity |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['state_id']}` | {row['workload_label']} | "
            f"{fmt(row['wall_ms'])} | {fmt(row['ttft_p50_ms'])} | "
            f"{fmt(row['inter_token_p50_ms'])} | "
            f"{fmt(row['output_tokens_per_second'], 1)} | "
            f"{fmt(row['steady_decode_tokens_per_second'], 1)} | "
            f"{fmt(row['peak_allocated_gb'])} | {fmt(row['kv_storage_gb'])} | "
            f"{fmt(row['parity_with_reference'])} | "
            f"{fmt(100 * row['common_output_prefix_fraction'], 1) + '%' if row['common_output_prefix_fraction'] is not None else '-'} |"
        )

    by_workload: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_workload.setdefault(row["workload_id"], []).append(row)
    lines.extend(["", "## Incremental Speedups", ""])
    for workload_id, workload_rows in by_workload.items():
        lines.extend(
            [
                f"### {workload_rows[0]['workload_label']}",
                "",
                "| State | Speedup vs first | Speedup vs previous |",
                "|---|---:|---:|",
            ]
        )
        baseline = workload_rows[0]["wall_ms"]
        previous = baseline
        for row in workload_rows:
            wall = row["wall_ms"]
            lines.append(
                f"| `{row['state_id']}` | {baseline / wall:.2f}x | "
                f"{previous / wall:.2f}x |"
            )
            previous = wall
        lines.append("")

    prefix_states = [
        state for state in successful if state.get("prefix_cache") is not None
    ]
    if prefix_states:
        lines.extend(
            [
                "## Prefix Cache Hit, Eviction, and Recompute",
                "",
                "| State | Shared tokens | Requests | Hit p50 ms | Miss p50 ms | Recompute / hit | Hit match | Miss match | Evicted match | Evictions | Output parity |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for state in prefix_states:
            prefix = state["prefix_cache"]
            stats = prefix.get("cache_stats") or {}
            lines.append(
                f"| `{state['state']['id']}` | {prefix['config']['shared_tokens']} | "
                f"{prefix['config']['concurrent_requests']} | "
                f"{fmt(prefix['hit_latency_ms']['p50'])} | "
                f"{fmt(prefix['miss_latency_ms']['p50'])} | "
                f"{fmt(prefix['recompute_over_hit'])}x | "
                f"{prefix['matched_tokens_hit'][0]} | "
                f"{prefix['matched_tokens_miss'][0]} | "
                f"{prefix['evicted_prefix_match_tokens']} | "
                f"{stats.get('evictions', 0)} | "
                f"{fmt(prefix['output_parity'])} |"
            )
        lines.append("")

    if failed:
        lines.extend(["## Failed States", ""])
        for state in failed:
            lines.append(
                f"- `{state.get('state', {}).get('id', 'unknown')}`: "
                f"{state.get('error', 'see state log')}"
            )
        lines.append("")
    return "\n".join(lines)


def print_console(rows: list[dict[str, Any]], states: list[dict[str, Any]]) -> None:
    print("\nMEGABENCHMARK SUMMARY")
    print(
        f"{'STATE':<42} {'WORKLOAD':<24} {'WALL ms':>10} "
        f"{'TTFT':>9} {'TPOT':>9} {'TOK/s':>10} {'GB':>7} {'PARITY':>8}"
    )
    print("-" * 125)
    for row in rows:
        print(
            f"{row['state_id']:<42} {row['workload_id']:<24} "
            f"{row['wall_ms']:>10.2f} {fmt(row['ttft_p50_ms']):>9} "
            f"{fmt(row['inter_token_p50_ms']):>9} "
            f"{row['output_tokens_per_second']:>10.1f} "
            f"{row['peak_allocated_gb']:>7.2f} "
            f"{fmt(row['parity_with_reference']):>8}"
        )
    failed = [state for state in states if state.get("status") != "ok"]
    if failed:
        print(f"\nFailed states: {', '.join(state['state']['id'] for state in failed)}")


def generate_reports(results_dir: Path) -> dict[str, Path]:
    states = load_state_results(results_dir)
    rows = flatten(states)
    combined = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "states": states,
        "rows": rows,
    }
    combined_path = results_dir / "combined.json"
    csv_path = results_dir / "summary.csv"
    markdown_path = results_dir / "report.md"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n")
    write_csv(csv_path, rows)
    markdown_path.write_text(markdown_report(states, rows) + "\n")
    print_console(rows, states)
    return {
        "combined": combined_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    paths = generate_reports(args.results_dir)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
