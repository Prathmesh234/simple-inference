"""Small end-to-end throughput benchmark for iteration 01."""

from __future__ import annotations

import time

import torch

from .test_engine import build_engine


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = build_engine(device, warmup=True)
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
    print(f"contiguous eager: {tokens / elapsed:.1f} tok/s ({tokens} tokens, {elapsed:.3f}s)")


if __name__ == "__main__":
    main()
