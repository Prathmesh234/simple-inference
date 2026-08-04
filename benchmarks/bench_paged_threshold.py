"""
Real-GPU threshold benchmark: contiguous KV rows vs paged KV blocks.

This script loads the actual Llama-3.2-3B model, then compares ONE cache
instance at a time under the same physical KV byte budget:

1. Find how many full `max_seq_len` contiguous rows fit beside model weights.
2. Allocate one paged pool with exactly the same number of token slots.
3. Feed mixed-length requests until each cache cannot admit another request.
4. Optionally run prefill + decode steps through the real InferenceEngine at
   each measured threshold.

The contiguous failure is structural: every request consumes one complete
`max_seq_len` row. The paged failure happens only when the shared physical page
budget cannot reserve another request's declared maximum length.

Examples:
    # Safe threshold measurement; retains 1 GB free as headroom.
    uv run python -m benchmarks.bench_paged_threshold

    # Probe as close as possible to CUDA OOM.
    uv run python -m benchmarks.bench_paged_threshold --headroom-gb 0

    # Execute threshold batches while reserving activation workspace.
    uv run python -m benchmarks.bench_paged_threshold \
        --headroom-gb 6 --execute-threshold

Notes:
    - Run with USE_CUDA_GRAPHS=false to isolate paging from graph replay.
    - `--execute-threshold` may hit an activation-memory OOM even when the KV
      cache fits; that means the prefill batch is too large, not that paging
      failed. The script reports the distinction.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import env_loader  # noqa: F401
import torch

from config import ModelConfig
from iterations.inference_01_contiguous_eager.engine import (
    InferenceEngine as ContiguousInferenceEngine,
)
from loader import WeightLoader
from model.kv_cache import KVCache
from model.llama import LlamaModel
from serving.block_allocator import BlockAllocator
from serving.engine import InferenceEngine
from serving.paged_kv_cache import PagedKVCache


MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda"
DTYPE = torch.bfloat16
RESULTS_FILE = Path(__file__).with_name("paged_threshold_results.json")


@dataclass(frozen=True)
class RequestShape:
    prompt_len: int
    max_new_tokens: int

    @property
    def max_length(self) -> int:
        return self.prompt_len + self.max_new_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prompt-min", type=int, default=64)
    parser.add_argument("--prompt-max", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--request-tokens",
        type=int,
        default=None,
        help=(
            "set every request's total prompt+decode length exactly; prompt "
            "length becomes request_tokens - max_new_tokens"
        ),
    )
    parser.add_argument("--request-pool", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--headroom-gb",
        type=float,
        default=1.0,
        help="CUDA memory left unused while finding the contiguous threshold",
    )
    parser.add_argument(
        "--execute-threshold",
        action="store_true",
        help="run real prefill/decode at both measured thresholds",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=2,
        help="decode iterations after prefill when --execute-threshold is set",
    )
    return parser.parse_args()


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_workload(args: argparse.Namespace, cfg: ModelConfig) -> list[RequestShape]:
    rng = random.Random(args.seed)
    if args.request_tokens is not None:
        prompt_len = args.request_tokens - args.max_new_tokens
        if prompt_len < 1:
            raise ValueError("--request-tokens must exceed --max-new-tokens")
        requests = [
            RequestShape(
                prompt_len=prompt_len,
                max_new_tokens=args.max_new_tokens,
            )
            for _ in range(args.request_pool)
        ]
    else:
        requests = [
            RequestShape(
                prompt_len=rng.randint(args.prompt_min, args.prompt_max),
                max_new_tokens=args.max_new_tokens,
            )
            for _ in range(args.request_pool)
        ]
    if max(request.max_length for request in requests) > args.max_seq_len:
        raise ValueError("generated request exceeds --max-seq-len")
    if args.max_seq_len > cfg.max_position_embeddings:
        raise ValueError("--max-seq-len exceeds the model position limit")
    return requests


def make_tokens(length: int, cfg: ModelConfig, salt: int) -> list[int]:
    """Deterministic valid token IDs; semantics do not affect KV allocation."""
    body = [
        1 + ((salt * 131 + position * 17) % (cfg.vocab_size - 1))
        for position in range(max(0, length - 1))
    ]
    return [cfg.bos_token_id] + body


def try_contiguous_rows(
    cfg: ModelConfig,
    rows: int,
    max_seq_len: int,
) -> tuple[bool, int]:
    cache = None
    try:
        cache = KVCache(
            n_layers=cfg.num_hidden_layers,
            max_batch=rows,
            max_seq_len=max_seq_len,
            n_heads_kv=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            dtype=DTYPE,
            device=DEVICE,
        )
        torch.cuda.synchronize()
        return True, cache.bytes()
    except torch.OutOfMemoryError:
        return False, 0
    finally:
        del cache
        cleanup_cuda()


def find_contiguous_threshold(
    cfg: ModelConfig,
    max_seq_len: int,
    headroom_bytes: int,
) -> tuple[int, int]:
    """
    Use current free VRAM to bound the search, then validate with real tensors.

    Binary search is intentionally over complete KV rows, exactly matching the
    non-paged engine's allocation unit.
    """
    free_bytes, _ = torch.cuda.mem_get_info()
    row_bytes = cfg.kv_cache_bytes(1, max_seq_len)
    usable = max(0, free_bytes - headroom_bytes)
    upper = usable // row_bytes
    if upper < 1:
        raise MemoryError(
            f"one contiguous KV row needs {row_bytes / 1e9:.3f} GB, but only "
            f"{usable / 1e9:.3f} GB is available after headroom"
        )

    low, high = 0, int(upper)
    best_bytes = 0
    while low < high:
        candidate = (low + high + 1) // 2
        ok, allocated = try_contiguous_rows(cfg, candidate, max_seq_len)
        if ok:
            low = candidate
            best_bytes = allocated
        else:
            high = candidate - 1
    if best_bytes == 0:
        ok, best_bytes = try_contiguous_rows(cfg, low, max_seq_len)
        if not ok:
            raise RuntimeError("contiguous threshold search found no allocatable row count")
    return low, best_bytes


def find_paged_threshold(
    requests: list[RequestShape],
    num_blocks: int,
    block_size: int,
) -> tuple[int, int]:
    allocator = BlockAllocator(num_blocks)
    for index, request in enumerate(requests, start=1):
        needed = math.ceil(request.max_length / block_size)
        if not allocator.can_reserve(index, needed):
            return index - 1, allocator.reserved_blocks
        allocator.reserve(index, needed)
    return len(requests), allocator.reserved_blocks


def validate_equal_paged_allocation(
    cfg: ModelConfig,
    num_blocks: int,
    block_size: int,
) -> int:
    cache = None
    try:
        cache = PagedKVCache(
            n_layers=cfg.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_heads_kv=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            dtype=DTYPE,
            device=DEVICE,
        )
        torch.cuda.synchronize()
        return cache.bytes()
    finally:
        del cache
        cleanup_cuda()


@torch.no_grad()
def execute_engine(
    model: LlamaModel,
    cfg: ModelConfig,
    requests: list[RequestShape],
    *,
    paged: bool,
    max_running: int,
    max_seq_len: int,
    block_size: int,
    num_kv_blocks: int | None,
    decode_steps: int,
) -> dict:
    """
    Run one prefill plus a few decode iterations at the measured threshold.

    All requests are admitted together so this tests the actual peak batch.
    An activation OOM is reported separately from the KV admission threshold.
    """
    cleanup_cuda()
    engine = None
    phase = "engine_initialization"
    try:
        common = dict(
            model=model,
            max_running=max_running,
            max_seq_len=max_seq_len,
            token_budget=sum(request.prompt_len for request in requests),
            temperature=0.0,
            warmup=False,
        )
        if paged:
            engine = InferenceEngine(
                **common,
                block_size=block_size,
                num_kv_blocks=num_kv_blocks,
                use_cuda_graphs=False,
            )
        else:
            engine = ContiguousInferenceEngine(**common)
        for index, request in enumerate(requests):
            engine.add_request(
                make_tokens(request.prompt_len, cfg, index),
                max_new_tokens=request.max_new_tokens,
            )

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        phase = "prefill"
        engine.step()  # admit + prefill + first token
        for step in range(decode_steps):
            phase = f"decode_step_{step + 1}"
            engine.step()
        torch.cuda.synchronize()
        finished = sum(
            request.state.value == "finished"
            for request in engine.scheduler.running
        )
        return {
            "status": "pass",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "submitted": len(requests),
            "running": len(engine.scheduler.running),
            "waiting": len(engine.scheduler.waiting),
            "finished_still_pending_eviction": finished,
        }
    except torch.OutOfMemoryError as error:
        return {
            "status": "execution_oom",
            "phase": phase,
            "submitted": len(requests),
            "error": str(error).splitlines()[0],
        }
    finally:
        del engine
        cleanup_cuda()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to this Python process. Run this script from "
            "the SSH shell/environment where nvidia-smi and torch.cuda work."
        )
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        raise ValueError("--block-size must be a positive power of two")
    if args.execute_threshold and args.headroom_gb < 1.0:
        print(
            "[warning] --execute-threshold with less than 1 GB headroom is "
            "expected to OOM after the KV cache fits because prefill/decode "
            "still need activation workspace. Use --headroom-gb 6 for the "
            "first executable-threshold run."
        )

    print(f"Loading {args.model_id}...")
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(args.model_id)
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()
    cleanup_cuda()

    free_after_model, total_vram = torch.cuda.mem_get_info()
    model_allocated = torch.cuda.memory_allocated()
    workload = make_workload(args, cfg)
    headroom_bytes = int(args.headroom_gb * 1e9)

    contiguous_requests, contiguous_bytes = find_contiguous_threshold(
        cfg, args.max_seq_len, headroom_bytes
    )
    block_bytes = cfg.kv_cache_bytes(1, args.block_size)
    if contiguous_bytes % block_bytes != 0:
        raise ValueError(
            "the contiguous KV budget is not divisible by one paged block; "
            "choose a block size that divides max_seq_len"
        )
    num_blocks = contiguous_bytes // block_bytes
    paged_bytes = validate_equal_paged_allocation(
        cfg, num_blocks, args.block_size
    )
    if paged_bytes != contiguous_bytes:
        raise AssertionError(
            f"unequal cache budgets: contiguous={contiguous_bytes}, paged={paged_bytes}"
        )
    paged_requests, used_blocks = find_paged_threshold(
        workload, num_blocks, args.block_size
    )

    result = {
        "model_id": args.model_id,
        "gpu": torch.cuda.get_device_name(),
        "total_vram_gb": round(total_vram / 1e9, 3),
        "model_allocated_gb": round(model_allocated / 1e9, 3),
        "free_after_model_gb": round(free_after_model / 1e9, 3),
        "headroom_gb": args.headroom_gb,
        "max_seq_len": args.max_seq_len,
        "block_size": args.block_size,
        "kv_budget_gb": round(contiguous_bytes / 1e9, 3),
        "workload": {
            "prompt_min": args.prompt_min,
            "prompt_max": args.prompt_max,
            "max_new_tokens": args.max_new_tokens,
            "request_tokens": args.request_tokens,
            "seed": args.seed,
        },
        "contiguous_threshold": contiguous_requests,
        "paged_threshold": paged_requests,
        "improvement": round(paged_requests / contiguous_requests, 3),
        "paged_blocks_used": used_blocks,
        "paged_blocks_total": num_blocks,
        "paged_utilization": round(used_blocks / num_blocks, 4),
    }

    print("\n" + "=" * 88)
    print("ACTUAL LLAMA-3.2-3B KV THRESHOLD")
    print("=" * 88)
    print(f"GPU                     : {result['gpu']}")
    print(f"VRAM total              : {result['total_vram_gb']:.3f} GB")
    print(f"model allocated         : {result['model_allocated_gb']:.3f} GB")
    print(f"free after model        : {result['free_after_model_gb']:.3f} GB")
    print(f"equal KV budget         : {result['kv_budget_gb']:.3f} GB")
    if args.request_tokens is not None:
        print(
            f"request shape           : {args.request_tokens} total tokens "
            f"({args.request_tokens - args.max_new_tokens} prompt + "
            f"{args.max_new_tokens} decode)"
        )
    else:
        print(f"request shape           : prompt U[{args.prompt_min},{args.prompt_max}] "
              f"+ {args.max_new_tokens} decode")
    print(f"contiguous threshold    : {contiguous_requests} requests")
    print(f"paged threshold         : {paged_requests} requests")
    print(f"concurrency improvement : {result['improvement']:.2f}x")
    print(f"paged block utilization : {result['paged_utilization']:.1%}")
    boundary = "CUDA allocation" if args.headroom_gb == 0 else "selected safe VRAM budget"
    print(f"threshold boundary      : {boundary}")
    print(
        f"next contiguous request : WAITS — no full {args.max_seq_len}-token row remains"
    )
    print(
        "next paged request      : WAITS — insufficient blocks for its declared maximum"
    )

    if args.execute_threshold:
        print("\nExecuting contiguous threshold batch...")
        result["contiguous_execution"] = execute_engine(
            model,
            cfg,
            workload[:contiguous_requests],
            paged=False,
            max_running=contiguous_requests,
            max_seq_len=args.max_seq_len,
            block_size=args.block_size,
            num_kv_blocks=None,
            decode_steps=args.decode_steps,
        )
        print(result["contiguous_execution"])

        print("\nExecuting paged threshold batch...")
        result["paged_execution"] = execute_engine(
            model,
            cfg,
            workload[:paged_requests],
            paged=True,
            max_running=paged_requests,
            max_seq_len=args.max_seq_len,
            block_size=args.block_size,
            num_kv_blocks=num_blocks,
            decode_steps=args.decode_steps,
        )
        print(result["paged_execution"])

    RESULTS_FILE.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
