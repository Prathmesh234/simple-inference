"""
Iteration 09 — PagedAttention (original gathered reference path)

This iteration intentionally captures the FIRST PagedAttention implementation,
before the direct page-table Triton kernel and CUDA-graph integration:

    BlockAllocator
        -> per-request logical-to-physical block table
        -> PagedKVCache writes K/V into fixed-size physical pages
        -> gather pages back into logical sequence order
        -> masked PyTorch SDPA

Why preserve this version?
--------------------------
It makes the paging mechanism visible and independently testable. Production
PagedAttention removes the gather by reading block tables inside the attention
kernel, but the allocator and logical-position mapping remain the same.

What this script shows
----------------------
1. Two requests allocate pages in an interleaved order, proving one request's
   logical sequence does not need physically contiguous memory.
2. Gathered paged attention matches contiguous SDPA numerically.
3. Llama-3.2-3B KV-capacity improvement for short/mixed-length requests.
4. The latency cost of materializing gathered K/V before SDPA.

No model weights are required.

Run:
    USE_TRITON=false uv run python iterations/09_paged_attention.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from benchmarks.bench_paged_attention import (
    print_attention_results,
    print_capacity_results,
)
from serving.paged_kv_cache import PagedKVCache


def block_table_demo() -> None:
    """
    Force interleaved physical allocation:

        request 101 logical page 0 -> physical page 0
        request 202 logical page 0 -> physical page 1
        request 101 logical page 1 -> physical page 2

    Request 101 therefore owns logical pages `[0, 1]` through the non-contiguous
    physical table `[0, 2]`.
    """
    cache = PagedKVCache(
        n_layers=1,
        num_blocks=4,
        block_size=4,
        n_heads_kv=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    cache.reserve_request(request_id=101, max_tokens=8)
    cache.reserve_request(request_id=202, max_tokens=4)

    first_page = torch.tensor(
        [[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]]]
    )
    other_request = torch.tensor(
        [[[100.0, 110.0], [101.0, 111.0], [102.0, 112.0], [103.0, 113.0]]]
    )
    second_page = torch.tensor(
        [[[4.0, 14.0], [5.0, 15.0], [6.0, 16.0], [7.0, 17.0]]]
    )

    cache.append(0, 101, 0, first_page, first_page)
    cache.append(0, 202, 0, other_request, other_request)
    cache.append(0, 101, 4, second_page, second_page)

    gathered_k, _ = cache.gather(layer_idx=0, request_id=101)
    expected = torch.cat((first_page, second_page), dim=1)
    torch.testing.assert_close(gathered_k, expected)

    print("=" * 72)
    print("ITERATION 09 — PAGED ATTENTION: BLOCK-TABLE WALKTHROUGH")
    print("=" * 72)
    print(f"block size                : {cache.block_size} tokens")
    print(f"request 101 block table   : {cache.block_tables[101]}")
    print(f"request 202 block table   : {cache.block_tables[202]}")
    print("\nrequest 101 logical mapping:")
    for position in range(8):
        logical_block = position // cache.block_size
        block_offset = position % cache.block_size
        physical_block = cache.block_tables[101][logical_block]
        print(
            f"  token {position}: logical block {logical_block}, "
            f"offset {block_offset} -> physical block {physical_block}"
        )
    print("\ngathered request-101 K:")
    print(gathered_k)
    print("gather correctness        : PASS")


def main() -> None:
    block_table_demo()
    print("\n" + "=" * 72)
    print_capacity_results()
    print_attention_results()


if __name__ == "__main__":
    main()
