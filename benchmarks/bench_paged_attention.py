"""
Section 16 benchmark: paged KV capacity and gathered-attention overhead.

This benchmark intentionally needs no model weights:

1. Capacity simulation uses Llama-3.2-3B's real KV geometry to compare fixed
   max-length rows against 16-token pages for short and mixed-length requests.
2. An attention microbenchmark compares contiguous SDPA with the first
   correctness-oriented paged path (page-table gather + masked SDPA).

Run:
    USE_TRITON=false uv run python -m benchmarks.bench_paged_attention
"""

from __future__ import annotations

import math
import random
import statistics
import time

import torch
import torch.nn.functional as F

from serving.paged_kv_cache import PagedKVCache, paged_attention_forward


LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2
BLOCK_SIZE = 16
STATIC_MAX_RUNNING = 8
STATIC_MAX_SEQ_LEN = 4096
POOL_BLOCKS = STATIC_MAX_RUNNING * math.ceil(STATIC_MAX_SEQ_LEN / BLOCK_SIZE)


def kv_bytes(token_slots: int) -> int:
    """K+V bytes across every model layer for `token_slots` logical positions."""
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES * token_slots


def paged_concurrency(max_lengths: list[int], pool_blocks: int = POOL_BLOCKS) -> int:
    """FCFS requests that can reserve pages before the physical pool is full."""
    used = 0
    admitted = 0
    for length in max_lengths:
        needed = math.ceil(length / BLOCK_SIZE)
        if used + needed > pool_blocks:
            break
        used += needed
        admitted += 1
    return admitted


def print_capacity_results() -> None:
    token_slots = STATIC_MAX_RUNNING * STATIC_MAX_SEQ_LEN
    static_gb = kv_bytes(token_slots) / 1e9
    print("KV CAPACITY — Llama-3.2-3B BF16")
    print(f"static baseline : {STATIC_MAX_RUNNING} rows x {STATIC_MAX_SEQ_LEN} tokens")
    print(f"physical pool   : {POOL_BLOCKS} blocks x {BLOCK_SIZE} = {token_slots:,} token slots")
    print(f"pool footprint  : {static_gb:.3f} GB\n")

    fixed_lengths = (384, 800, 1280)
    print(f"{'request max':>12} | {'static conc.':>12} | {'paged conc.':>11} | {'gain':>7}")
    print("-" * 54)
    for length in fixed_lengths:
        paged = POOL_BLOCKS // math.ceil(length / BLOCK_SIZE)
        print(
            f"{length:>12,} | {STATIC_MAX_RUNNING:>12} | {paged:>11} | "
            f"{paged / STATIC_MAX_RUNNING:>6.2f}x"
        )

    rng = random.Random(0)
    trials = []
    for _ in range(10_000):
        # Section-16 workload: prompt U[64,1024] plus 256 decode tokens.
        lengths = [rng.randint(64, 1024) + 256 for _ in range(256)]
        trials.append(paged_concurrency(lengths))
    mean = statistics.mean(trials)
    p05 = sorted(trials)[int(0.05 * len(trials))]
    p95 = sorted(trials)[int(0.95 * len(trials))]
    print(
        "\nmixed workload (prompt U[64,1024] + 256 decode): "
        f"mean={mean:.1f}, p05={p05}, p95={p95} concurrent "
        f"({mean / STATIC_MAX_RUNNING:.2f}x vs static)"
    )

    # Same eight 800-token requests, but size the physical pool to their actual
    # declared maxima rather than eight worst-case 4096-token rows.
    right_sized_slots = STATIC_MAX_RUNNING * math.ceil(800 / BLOCK_SIZE) * BLOCK_SIZE
    right_sized_gb = kv_bytes(right_sized_slots) / 1e9
    saving = 1.0 - right_sized_gb / static_gb
    print(
        f"8 requests capped at 800 tokens need {right_sized_gb:.3f} GB of pages "
        f"vs {static_gb:.3f} GB static ({saving:.1%} less KV memory)."
    )


def bench(fn, warmup: int = 10, repetitions: int = 50, rounds: int = 7) -> float:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(rounds):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000 / repetitions)
    return statistics.median(timings)


def print_attention_results() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    batch = 8
    query_heads = 8
    kv_heads = 2
    head_dim = 64
    lengths = [128, 160, 192, 224, 256, 288, 320, 352]
    max_length = max(lengths)

    torch.manual_seed(0)
    q = torch.randn(batch, query_heads, 1, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_heads, max_length, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    positions = torch.tensor([length - 1 for length in lengths], device=device)
    key_lengths = torch.tensor(lengths, device=device)
    cols = torch.arange(max_length, device=device)
    mask = torch.zeros(batch, 1, 1, max_length, dtype=dtype, device=device)
    mask.masked_fill_(cols.view(1, 1, 1, -1) >= key_lengths.view(-1, 1, 1, 1), float("-inf"))

    cache = PagedKVCache(
        n_layers=1,
        num_blocks=sum(math.ceil(length / BLOCK_SIZE) for length in lengths),
        block_size=BLOCK_SIZE,
        n_heads_kv=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    for request_id, length in enumerate(lengths):
        cache.reserve_request(request_id, length)
        cache.append(0, request_id, 0, k[request_id, :, :length], v[request_id, :, :length])

    def contiguous_attention():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    def gathered_paged_attention():
        gathered_k, gathered_v = cache.gather_batch(0, list(range(batch)), lengths)
        return paged_attention_forward(
            q,
            gathered_k,
            gathered_v,
            positions.view(batch, 1),
            key_lengths,
            num_kv_groups=query_heads // kv_heads,
        )

    expected = contiguous_attention()
    actual = gathered_paged_attention()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    contiguous_ms = bench(contiguous_attention)
    paged_ms = bench(gathered_paged_attention)
    print(f"\nGATHERED ATTENTION OVERHEAD — {device.type.upper()}")
    print(f"contiguous SDPA       : {contiguous_ms:.3f} ms")
    print(f"page gather + SDPA    : {paged_ms:.3f} ms")
    print(f"current latency ratio : {paged_ms / contiguous_ms:.2f}x")


def main() -> None:
    print_capacity_results()
    print_attention_results()


if __name__ == "__main__":
    main()
