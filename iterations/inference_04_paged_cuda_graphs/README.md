# 04 — PagedAttention with CUDA-graph decode

This is the current serving implementation. Page allocation stays outside graph
capture. Fixed token, position, RoPE, and block-table tensors are updated in
place before replay, while the paged Triton kernel follows block-table entries
directly instead of gathering K/V.

Prefill remains eager because prompt shapes vary.

```bash
uv run python -m unittest iterations.inference_04_paged_cuda_graphs.test_engine -v
uv run python -m iterations.inference_04_paged_cuda_graphs.benchmark
```
