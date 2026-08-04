# Inference Engine Megabenchmark

This suite measures the existing optimization stack in isolated subprocesses:

`no KV cache -> static KV -> Triton -> autotuning -> fused transpose ->
continuous batching -> CUDA graphs -> paged KV -> direct paged Triton ->
radix prefix caching`

Every state explicitly sets all relevant environment flags before importing the
model or kernels and receives separate Triton and TorchInductor cache
directories. This avoids cross-state contamination from import-time
configuration, compiler caches, CUDA graphs, or model compilation.

## Run

```bash
uv run python -m benchmarks.megabenchmark.run --profile full
```

Use the quick profile to validate the environment first:

```bash
uv run python -m benchmarks.megabenchmark.run --profile quick
```

Useful controls:

```bash
# Show every state ID.
uv run python -m benchmarks.megabenchmark.run --list-states

# Preview the selected matrix without loading the model.
uv run python -m benchmarks.megabenchmark.run --profile full --dry-run

# Run selected states only.
uv run python -m benchmarks.megabenchmark.run \
  --profile full \
  --states 08_paged_direct_triton_eager,09_paged_direct_triton_cuda_graphs,10_prefix_cache

# Include the torch.compile alternative in addition to the core progression.
uv run python -m benchmarks.megabenchmark.run --profile full --states all
```

## Artifacts

Each run creates a timestamped directory under `benchmarks/megabenchmark/results/`
containing:

- `states/*.json`: complete per-state metrics, request outputs, and step traces
- `logs/*.log`: full subprocess output
- `combined.json`: all raw state results and flattened rows
- `summary.csv`: one row per state/workload
- `report.md`: tables and incremental speedups ready for `plan.md`
- `manifest.json`: exact state, flag, model, and workload configuration

An output directory must be empty. This prevents stale state files from a
different profile or model run from entering the aggregate report.

Metrics include model/engine initialization, TTFT, end-to-end latency,
inter-token latency, prefill- and decode-bearing step throughput, request and
token throughput, allocated/reserved VRAM, KV storage, allocator state, graph
backend, output hashes, exact parity, and common output-prefix parity. The
prefix-cache state uses at least 16 concurrent users, a 2,048-token shared
system prompt, and 512-token user-specific tails. It measures seeded hits,
forced LRU eviction, post-eviction recomputation, matched tokens, evictions,
and hit/miss output parity.
