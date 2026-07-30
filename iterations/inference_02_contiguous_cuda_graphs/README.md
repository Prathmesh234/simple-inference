# 02 — Contiguous KV cache with CUDA graphs

This keeps the fixed KV-cache rows from iteration 01 and captures decode into
power-of-two batch buckets. Fixed input buffers are updated in place before
replay; prefill remains eager.

The serving files are preserved from commit `3a6a324`.

```bash
uv run python -m unittest iterations.inference_02_contiguous_cuda_graphs.test_engine -v
uv run python -m iterations.inference_02_contiguous_cuda_graphs.benchmark
```
