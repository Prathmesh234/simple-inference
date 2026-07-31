"""Real-model TTFT benchmark for long-context paged radix prefix caching."""

from __future__ import annotations

import argparse
import statistics
import time

import env_loader  # noqa: F401
import torch

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from serving.engine import InferenceEngine
from tokenizer import Tokenizer


MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda"
DTYPE = torch.bfloat16
COMMON_TEXT = (
    "You are a careful technical assistant analyzing a GPU inference engine. "
    "Explain assumptions clearly, use concrete measurements, and distinguish "
    "prefill latency, decode latency, memory capacity, and request throughput. "
)
SEED_TAIL = (
    "The first workload studies memory allocation, scheduling, and page reuse. "
)
TARGET_TAIL = (
    "The second workload studies attention kernels, CUDA graphs, and latency. "
)
COLD_TEXT = (
    "A completely separate English document discusses distributed databases, "
    "replication, consensus, storage engines, failure recovery, and networking. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
    )
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def build_model(model_id: str) -> LlamaModel:
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(model_id)
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    return model.eval()


def repeat_tokens(tokenizer: Tokenizer, text: str, count: int) -> list[int]:
    if count <= 0:
        return []
    body = tokenizer.encode(text, add_bos=False)
    if not body:
        raise RuntimeError("benchmark text produced no tokens")
    copies = (count + len(body) - 1) // len(body)
    return (body * copies)[:count]


def build_prompt_pair(
    tokenizer: Tokenizer,
    *,
    context_len: int,
    shared_tokens: int,
) -> tuple[list[int] | None, list[int]]:
    """Build a seed/target pair with an exact block-aligned shared prefix."""
    if not 0 <= shared_tokens < context_len:
        raise ValueError(
            f"shared_tokens must be in [0, {context_len}), got {shared_tokens}"
        )
    if shared_tokens == 0:
        target = [tokenizer.bos_id] + repeat_tokens(
            tokenizer, COLD_TEXT, context_len - 1
        )
        return None, target

    common = [tokenizer.bos_id] + repeat_tokens(
        tokenizer, COMMON_TEXT, shared_tokens - 1
    )
    tail_len = context_len - shared_tokens
    seed = common + repeat_tokens(tokenizer, SEED_TAIL, tail_len)
    target = common + repeat_tokens(tokenizer, TARGET_TAIL, tail_len)
    return seed, target


def build_engine(
    model: LlamaModel,
    *,
    max_seq_len: int,
    block_size: int,
    use_prefix_cache: bool,
) -> InferenceEngine:
    return InferenceEngine(
        model=model,
        max_running=1,
        max_seq_len=max_seq_len,
        block_size=block_size,
        eos_id=-1,
        temperature=0.0,
        warmup=False,
        use_cuda_graphs=False,
        use_paged_attention=True,
        use_prefix_cache=use_prefix_cache,
    )


def prefill_ms(
    engine: InferenceEngine,
    prompt: list[int],
) -> tuple[float, int, int]:
    request = engine.add_request(prompt, max_new_tokens=1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.step()
    torch.cuda.synchronize()
    if not request.reached_limit():
        raise RuntimeError("one-token benchmark request did not finish in prefill")
    return (
        (time.perf_counter() - start) * 1000,
        request.cached_prefix_len,
        request.generated[0],
    )


def benchmark_case(
    uncached: InferenceEngine,
    cached: InferenceEngine,
    *,
    seed: list[int] | None,
    target: list[int],
    repetitions: int,
) -> tuple[float, float, int]:
    uncached_times: list[float] = []
    cached_times: list[float] = []
    matched_tokens: list[int] = []

    for _ in range(repetitions):
        uncached.reset()
        cached.reset()
        if seed is not None:
            prefill_ms(cached, seed)

        uncached_ms, _, expected_token = prefill_ms(uncached, target)
        cached_ms, matched, actual_token = prefill_ms(cached, target)
        if actual_token != expected_token:
            raise AssertionError(
                "prefix-cached prefill changed the greedy next token: "
                f"{actual_token} != {expected_token}"
            )
        uncached_times.append(uncached_ms)
        cached_times.append(cached_ms)
        matched_tokens.append(matched)

    return (
        statistics.median(uncached_times),
        statistics.median(cached_times),
        int(statistics.median(matched_tokens)),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.context_lengths) < 2:
        raise ValueError("context lengths must be at least 2")
    if args.block_size <= 0 or args.tail_tokens < 1 or args.repetitions < 1:
        raise ValueError("block-size, tail-tokens, and repetitions must be positive")

    tokenizer = Tokenizer.from_pretrained(args.model_id)
    model = build_model(args.model_id)
    max_seq_len = max(args.context_lengths) + 1
    uncached = build_engine(
        model,
        max_seq_len=max_seq_len,
        block_size=args.block_size,
        use_prefix_cache=False,
    )
    cached = build_engine(
        model,
        max_seq_len=max_seq_len,
        block_size=args.block_size,
        use_prefix_cache=True,
    )

    # Compile/warm kernels before measured cases; reset removes warmup KV state.
    warmup = build_prompt_pair(
        tokenizer,
        context_len=min(args.context_lengths),
        shared_tokens=0,
    )[1]
    prefill_ms(uncached, warmup)
    prefill_ms(cached, warmup)

    header = (
        f"{'context':>7} | {'case':>9} | {'matched':>8} | {'hit %':>6} | "
        f"{'uncached':>10} | {'cached':>10} | {'speedup':>7}"
    )
    print("\n" + header)
    print("-" * len(header))

    for context_len in args.context_lengths:
        half_shared = (context_len // 2 // args.block_size) * args.block_size
        full_shared = (
            (context_len - args.tail_tokens) // args.block_size
        ) * args.block_size
        cases = (
            ("miss", 0),
            ("half-hit", half_shared),
            ("full-hit", full_shared),
        )
        for label, shared_tokens in cases:
            seed, target = build_prompt_pair(
                tokenizer,
                context_len=context_len,
                shared_tokens=shared_tokens,
            )
            uncached_ms, cached_ms, matched = benchmark_case(
                uncached,
                cached,
                seed=seed,
                target=target,
                repetitions=args.repetitions,
            )
            print(
                f"{context_len:>7} | {label:>9} | {matched:>8} | "
                f"{100 * matched / context_len:>5.1f}% | "
                f"{uncached_ms:>8.3f} ms | {cached_ms:>8.3f} ms | "
                f"{uncached_ms / cached_ms:>6.2f}x"
            )


if __name__ == "__main__":
    main()
