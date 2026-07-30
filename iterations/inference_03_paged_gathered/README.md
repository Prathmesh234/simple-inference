# 03 — Gathered PagedAttention

Fixed per-request KV rows are replaced by a physical page pool and request block
tables. Pages are allocated lazily, gathered back into a dense temporary, and
passed to SDPA. This is the easiest paged implementation to inspect.

Read `block_allocator.py`, then `paged_kv_cache.py`, then the two attention
helpers in `engine.py`.

```bash
uv run python -m unittest iterations.inference_03_paged_gathered.test_engine -v
uv run python -m iterations.inference_03_paged_gathered.benchmark
```
