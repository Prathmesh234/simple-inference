# 01 — Contiguous eager serving

The first continuous-batching engine. Each running request owns one complete
`max_seq_len` row in the contiguous KV cache. Prefill and decode both execute
eagerly.

The serving files are preserved from commit `d8d0303`.

Start with `engine.py`, then read `scheduler.py` to see how requests enter and
leave the changing decode batch.

```bash
uv run python -m unittest iterations.inference_01_contiguous_eager.test_engine -v
uv run python -m iterations.inference_01_contiguous_eager.benchmark
```
