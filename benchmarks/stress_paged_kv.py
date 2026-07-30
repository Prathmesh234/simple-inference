"""
Stress test: one contiguous KV cache instance vs one paged KV cache instance.

Both instances receive the SAME physical K/V byte budget.

Contiguous:
    every admitted request permanently owns `max_seq_len` token slots.

Paged:
    one shared block pool is allocated once; each request reserves only
    `ceil((prompt_len + max_new_tokens) / block_size)` pages.

The default budget is exactly the old serving configuration:

    8 requests * 4096 tokens/request = 32,768 physical token slots

On a CUDA machine, pass `--allocate` to instantiate the real tensors one cache
at a time. Without CUDA, the script still executes the exact admission math and
allocator lifecycle without allocating the multi-GB pool.

Run:
    uv run python -m benchmarks.stress_paged_kv
    uv run python -m benchmarks.stress_paged_kv --allocate
"""

from __future__ import annotations

import argparse
import gc
import math
import random
from dataclasses import dataclass

import torch

from model.kv_cache import KVCache
from serving.block_allocator import BlockAllocator
from serving.paged_kv_cache import PagedKVCache


LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16
DTYPE_BYTES = 2


@dataclass(frozen=True)
class RequestShape:
    prompt_len: int
    max_new_tokens: int

    @property
    def max_length(self) -> int:
        return self.prompt_len + self.max_new_tokens


def bytes_per_token_slot() -> int:
    """K+V bytes for one logical token position across all model layers."""
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES


def make_workload(
    count: int,
    prompt_min: int,
    prompt_max: int,
    max_new_tokens: int,
    seed: int,
) -> list[RequestShape]:
    rng = random.Random(seed)
    return [
        RequestShape(
            prompt_len=rng.randint(prompt_min, prompt_max),
            max_new_tokens=max_new_tokens,
        )
        for _ in range(count)
    ]


def stress_contiguous(
    requests: list[RequestShape],
    token_slot_budget: int,
    max_seq_len: int,
) -> tuple[int, str]:
    """
    A contiguous cache can admit only one request per complete max-length row.

    Request lengths do not matter: every request consumes `max_seq_len` slots.
    """
    max_rows = token_slot_budget // max_seq_len
    admitted = min(max_rows, len(requests))
    reason = (
        "all fixed KV rows are occupied"
        if admitted < len(requests)
        else "workload exhausted before cache"
    )
    return admitted, reason


def stress_paged(
    requests: list[RequestShape],
    token_slot_budget: int,
    block_size: int,
) -> tuple[int, int, str]:
    """FCFS admission into one fixed physical page pool."""
    num_blocks = token_slot_budget // block_size
    allocator = BlockAllocator(num_blocks)
    admitted = 0

    for request_id, request in enumerate(requests, start=1):
        blocks = math.ceil(request.max_length / block_size)
        if not allocator.can_reserve(request_id, blocks):
            return admitted, allocator.reserved_blocks, "insufficient free page reservation"
        allocator.reserve(request_id, blocks)
        admitted += 1

    return admitted, allocator.reserved_blocks, "workload exhausted before cache"


def allocate_contiguous(max_rows: int, max_seq_len: int, device: str) -> int:
    """Instantiate one real contiguous pool; returns allocated bytes."""
    cache = KVCache(
        n_layers=LAYERS,
        max_batch=max_rows,
        max_seq_len=max_seq_len,
        n_heads_kv=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=device,
    )
    allocated = cache.bytes()
    del cache
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return allocated


def allocate_paged(num_blocks: int, block_size: int, device: str) -> int:
    """Instantiate one real paged pool; returns allocated bytes."""
    cache = PagedKVCache(
        n_layers=LAYERS,
        num_blocks=num_blocks,
        block_size=block_size,
        n_heads_kv=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=device,
    )
    allocated = cache.bytes()
    del cache
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return allocated


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--static-rows", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prompt-min", type=int, default=64)
    parser.add_argument("--prompt-max", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--trials", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allocate",
        action="store_true",
        help="allocate each real multi-GB cache instance (requires CUDA)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.max_seq_len <= 0 or args.static_rows <= 0:
        raise ValueError("block_size, max_seq_len, and static_rows must be positive")
    if args.prompt_min <= 0 or args.prompt_max < args.prompt_min:
        raise ValueError("invalid prompt length range")
    if args.prompt_max + args.max_new_tokens > args.max_seq_len:
        raise ValueError("largest request exceeds max_seq_len")

    token_slot_budget = args.static_rows * args.max_seq_len
    byte_budget = token_slot_budget * bytes_per_token_slot()
    num_blocks = token_slot_budget // args.block_size
    static_capacity, static_reason = stress_contiguous(
        make_workload(
            args.requests,
            args.prompt_min,
            args.prompt_max,
            args.max_new_tokens,
            args.seed,
        ),
        token_slot_budget,
        args.max_seq_len,
    )

    paged_capacities = []
    paged_reserved = []
    for trial in range(args.trials):
        workload = make_workload(
            args.requests,
            args.prompt_min,
            args.prompt_max,
            args.max_new_tokens,
            args.seed + trial,
        )
        admitted, reserved, _ = stress_paged(
            workload, token_slot_budget, args.block_size
        )
        paged_capacities.append(admitted)
        paged_reserved.append(reserved)

    mean_paged = sum(paged_capacities) / len(paged_capacities)
    print("=" * 78)
    print("ONE-INSTANCE KV-CACHE STRESS TEST")
    print("=" * 78)
    print(f"model KV geometry : {LAYERS} layers, {KV_HEADS} KV heads, D={HEAD_DIM}, BF16")
    print(f"physical budget   : {token_slot_budget:,} token slots = {byte_budget / 1e9:.3f} GB")
    print(f"request workload  : prompt U[{args.prompt_min},{args.prompt_max}] + "
          f"{args.max_new_tokens} max decode tokens")
    print(f"paged block size  : {args.block_size} tokens ({num_blocks:,} physical blocks)\n")

    print("CONTIGUOUS INSTANCE")
    print(f"  admitted before full : {static_capacity}")
    print(f"  failure reason       : {static_reason}")
    print(f"  allocation/request   : {args.max_seq_len:,} token slots")

    print("\nPAGED INSTANCE")
    print(f"  mean admitted        : {mean_paged:.1f}")
    print(f"  p05 / p50 / p95      : "
          f"{percentile(paged_capacities, 0.05)} / "
          f"{percentile(paged_capacities, 0.50)} / "
          f"{percentile(paged_capacities, 0.95)}")
    print(f"  max observed         : {max(paged_capacities)}")
    print(f"  mean improvement     : {mean_paged / static_capacity:.2f}x")
    print(f"  mean page utilization: "
          f"{sum(paged_reserved) / len(paged_reserved) / num_blocks:.1%}")
    print("  failure reason       : insufficient free page reservation")

    if args.allocate:
        if not torch.cuda.is_available():
            raise RuntimeError("--allocate requires a CUDA device")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if byte_budget > free_bytes:
            raise MemoryError(
                f"stress pool needs {byte_budget / 1e9:.3f} GB but only "
                f"{free_bytes / 1e9:.3f} GB CUDA memory is free"
            )
        print("\nREAL CUDA ALLOCATION (one instance at a time)")
        contiguous_bytes = allocate_contiguous(
            args.static_rows, args.max_seq_len, "cuda"
        )
        print(f"  contiguous pool      : {contiguous_bytes / 1e9:.3f} GB allocated")
        paged_bytes = allocate_paged(num_blocks, args.block_size, "cuda")
        print(f"  paged pool           : {paged_bytes / 1e9:.3f} GB allocated")
        print("  equal-byte check     : "
              f"{'PASS' if contiguous_bytes == paged_bytes else 'FAIL'}")
    else:
        mode = "CUDA available; rerun with --allocate" if torch.cuda.is_available() else "no CUDA device"
        print(f"\nREAL ALLOCATION SKIPPED: {mode}")


if __name__ == "__main__":
    main()
