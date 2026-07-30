"""Small end-to-end throughput benchmark for iteration 02."""

from __future__ import annotations

import time

import torch

from .test_engine import build_engine


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_graphs = device.type == "cuda"
    engine = build_engine(device, use_cuda_graphs=use_graphs, warmup=True)
    graphs_active = engine.graph_decoder is not None
    requests = [
        engine.add_request([1, 4 + index % 8, 6], max_new_tokens=8)
        for index in range(8)
    ]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    engine.run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens = sum(len(request.generated) for request in requests)
    mode = "CUDA graphs" if graphs_active else "eager fallback"
    print(f"contiguous {mode}: {tokens / elapsed:.1f} tok/s ({tokens} tokens, {elapsed:.3f}s)")


if __name__ == "__main__":
    main()
