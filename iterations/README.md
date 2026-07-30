# Inference serving iterations

`serving/` is the current implementation. These folders preserve the serving
layer at each major milestone while continuing to share `model/`, `ops/`, and
`kernels/`.

| Iteration | KV layout | Decode path |
|---|---|---|
| `inference_01_contiguous_eager` | Fixed row per request | Eager |
| `inference_02_contiguous_cuda_graphs` | Fixed row per request | Bucketed CUDA graphs |
| `inference_03_paged_gathered` | Physical blocks + block tables | Gather pages, then SDPA |
| `inference_04_paged_cuda_graphs` | Physical blocks + fixed GPU metadata | Direct paged CUDA-graph decode |

Run any snapshot from the repository root:

```bash
uv run python -m unittest iterations.inference_01_contiguous_eager.test_engine -v
uv run python -m iterations.inference_01_contiguous_eager.benchmark
```

Replace `01_contiguous_eager` with the milestone you want to inspect.
