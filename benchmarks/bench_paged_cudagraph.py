"""
Real-model paged decode benchmark: gather, direct eager, and graph replay.

Both regimes use the same Llama-3.2-3B weights, prompts, paged KV geometry,
block size, context length, and batch sizes:

  gathered eager: gather physical pages into dense K/V, then call PyTorch SDPA
  direct eager:   launch the direct Triton paged kernel normally
  direct graph:   replay the exact same direct Triton path as one CUDA graph

The primary CUDA-graph speedup is direct-eager / direct-graph, so it does not
mix graph replay with the separate benefit of eliminating the gather.

Run:
    uv run python -m benchmarks.bench_paged_cudagraph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import env_loader  # noqa: F401
import torch
import triton

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from serving.engine import InferenceEngine
from serving.request import Request, RequestState
from tokenizer import Tokenizer


MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda"
DTYPE = torch.bfloat16
RESULTS_FILE = Path(__file__).with_name("paged_cudagraph_results.json")
LOGIT_RTOL = 0.05
LOGIT_ATOL = 0.15
ENGLISH_PROMPTS = (
    (
        "A small inference engine processes requests in two stages. During "
        "prefill it reads the prompt and writes keys and values into the cache. "
        "During decode it generates one token at a time while reusing that cache."
    ),
    (
        "CUDA graphs reduce launch overhead by recording a fixed sequence of GPU "
        "operations once and replaying it later. Static buffers keep their memory "
        "addresses while token, position, and page-table values change in place."
    ),
    (
        "Paged attention divides the key-value cache into fixed-size physical "
        "blocks. Each request owns a block table that maps logical token positions "
        "to those blocks, allowing short requests to avoid reserving long rows."
    ),
    (
        "Continuous batching rebuilds the active batch after every decode step. "
        "Finished requests release their capacity immediately, and waiting requests "
        "can begin without waiting for every sequence in the previous batch."
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--decode-room", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--results-file", type=Path, default=RESULTS_FILE)
    return parser.parse_args()


def build_model(model_id: str) -> LlamaModel:
    print(f"Loading {model_id}...")
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(model_id)
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    return model.eval()


def build_english_prompts(
    tokenizer: Tokenizer,
    batch_size: int,
    context_len: int,
) -> list[list[int]]:
    """Create distinct tokenized English prompts with exactly `context_len` tokens."""
    prompts: list[list[int]] = []
    for row in range(batch_size):
        text = ENGLISH_PROMPTS[row % len(ENGLISH_PROMPTS)]
        body = tokenizer.encode(text, add_bos=False)
        if not body:
            raise RuntimeError("English benchmark prompt produced no tokens")
        needed = context_len - 1
        repeated = (body * ((needed + len(body) - 1) // len(body)))[:needed]
        prompts.append([tokenizer.bos_id, *repeated])
    return prompts


def build_engine(
    model: LlamaModel,
    *,
    max_running: int,
    max_seq_len: int,
    block_size: int,
    use_cuda_graphs: bool,
) -> InferenceEngine:
    return InferenceEngine(
        model=model,
        max_running=max_running,
        max_seq_len=max_seq_len,
        block_size=block_size,
        temperature=0.0,
        warmup=False,
        use_cuda_graphs=use_cuda_graphs,
        use_paged_attention=True,
    )


def make_steady_requests(
    engine: InferenceEngine,
    prompts: list[list[int]],
) -> list[Request]:
    engine.reset()
    if not prompts:
        raise ValueError("prompts must not be empty")
    context_len = len(prompts[0])
    if any(len(prompt) != context_len for prompt in prompts):
        raise ValueError("all benchmark prompts must have the same context length")
    max_new_tokens = engine.max_seq_len - context_len
    for prompt in prompts:
        engine.add_request(prompt, max_new_tokens=max_new_tokens)

    engine.step()
    running = [
        request
        for request in engine.scheduler.running
        if request.state is RequestState.DECODE
    ]
    if len(running) != len(prompts):
        raise RuntimeError(
            f"expected {len(prompts)} decode requests, found {len(running)}"
        )
    return running


@torch.no_grad()
def eager_logits(engine: InferenceEngine, requests: list[Request]) -> torch.Tensor:
    return engine._decode_logits_eager(requests)


@torch.no_grad()
def graph_logits(engine: InferenceEngine, requests: list[Request]) -> torch.Tensor:
    decoder = engine.graph_decoder
    if decoder is None:
        raise RuntimeError("paged CUDA-graph decoder is not active")
    logits = decoder.logits(
        [request.id for request in requests],
        [request.pos for request in requests],
        [request.last_token for request in requests],
    )
    if logits is None:
        raise RuntimeError("no CUDA-graph bucket can hold the running batch")
    return logits


@torch.no_grad()
def direct_eager_logits(
    engine: InferenceEngine,
    requests: list[Request],
) -> torch.Tensor:
    decoder = engine.graph_decoder
    if decoder is None:
        raise RuntimeError("paged CUDA-graph decoder is not active")
    running = len(requests)
    bucket = decoder.bucket_for(running)
    if bucket is None:
        raise RuntimeError("no direct-paged bucket can hold the running batch")
    if bucket not in decoder.graphs:
        decoder.capture(bucket)
    buffers = decoder.bufs[bucket]
    decoder._fill_for_decode(
        buffers,
        bucket,
        [request.id for request in requests],
        [request.pos for request in requests],
        [request.last_token for request in requests],
    )
    return decoder._run_layers(buffers, bucket)[:running].clone()


def benchmark_ms(fn, *, warmup: int, rep: int) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")
    if args.context_len < 1 or args.decode_room < 2:
        raise ValueError("context-len must be positive and decode-room must be >= 2")

    max_seq_len = args.context_len + args.decode_room
    max_running = max(args.batch_sizes)
    tokenizer = Tokenizer.from_pretrained(args.model_id)
    model = build_model(args.model_id)

    print("Building paged eager engine...")
    eager = build_engine(
        model,
        max_running=max_running,
        max_seq_len=max_seq_len,
        block_size=args.block_size,
        use_cuda_graphs=False,
    )
    print("Building paged CUDA-graph engine...")
    graph = build_engine(
        model,
        max_running=max_running,
        max_seq_len=max_seq_len,
        block_size=args.block_size,
        use_cuda_graphs=True,
    )

    print(
        f"\ncontext={args.context_len}, max_seq={max_seq_len}, "
        f"block={args.block_size}, batches={args.batch_sizes}"
    )
    header = (
        f"{'batch':>6} | {'gather ms':>9} | {'direct ms':>9} | {'graph ms':>9} | "
        f"{'graph win':>9} | {'graph tok/s':>12} | {'max |diff|':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    rows: list[dict[str, float | int]] = []
    for batch_size in args.batch_sizes:
        prompts = build_english_prompts(tokenizer, batch_size, args.context_len)
        eager_requests = make_steady_requests(eager, prompts)
        graph_requests = make_steady_requests(graph, prompts)

        expected = eager_logits(eager, eager_requests)
        direct = direct_eager_logits(graph, graph_requests)
        actual = graph_logits(graph, graph_requests)
        direct_diff = float((expected.float() - direct.float()).abs().max().item())
        graph_diff = float((expected.float() - actual.float()).abs().max().item())
        max_abs_diff = max(direct_diff, graph_diff)
        direct_top1_agreement = float(
            (expected.argmax(dim=-1) == direct.argmax(dim=-1)).float().mean().item()
        )
        graph_top1_agreement = float(
            (expected.argmax(dim=-1) == actual.argmax(dim=-1)).float().mean().item()
        )
        top1_agreement = min(direct_top1_agreement, graph_top1_agreement)
        torch.testing.assert_close(
            direct.float(),
            expected.float(),
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        )
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        )
        if top1_agreement < 1.0:
            raise AssertionError(
                "direct/graph decode changed at least one greedy next token"
            )

        gathered_ms = benchmark_ms(
            lambda: eager_logits(eager, eager_requests),
            warmup=args.warmup,
            rep=args.rep,
        )
        direct_ms = benchmark_ms(
            lambda: direct_eager_logits(graph, graph_requests),
            warmup=args.warmup,
            rep=args.rep,
        )
        graph_ms = benchmark_ms(
            lambda: graph_logits(graph, graph_requests),
            warmup=args.warmup,
            rep=args.rep,
        )
        graph_speedup = direct_ms / graph_ms
        end_to_end_speedup = gathered_ms / graph_ms
        gathered_tps = batch_size / (gathered_ms * 1e-3)
        direct_tps = batch_size / (direct_ms * 1e-3)
        graph_tps = batch_size / (graph_ms * 1e-3)
        rows.append(
            {
                "batch_size": batch_size,
                "gathered_eager_ms": gathered_ms,
                "direct_eager_ms": direct_ms,
                "graph_ms": graph_ms,
                "cuda_graph_speedup": graph_speedup,
                "gathered_to_graph_speedup": end_to_end_speedup,
                "gathered_eager_tokens_per_second": gathered_tps,
                "direct_eager_tokens_per_second": direct_tps,
                "graph_tokens_per_second": graph_tps,
                "max_abs_logit_diff": max_abs_diff,
                "top1_agreement": top1_agreement,
            }
        )
        print(
            f"{batch_size:>6} | {gathered_ms:>9.3f} | {direct_ms:>9.3f} | "
            f"{graph_ms:>9.3f} | {graph_speedup:>8.2f}x | "
            f"{graph_tps:>12.1f} | {max_abs_diff:>10.4f}"
        )

    result = {
        "gpu": torch.cuda.get_device_name(),
        "model_id": args.model_id,
        "context_len": args.context_len,
        "max_seq_len": max_seq_len,
        "block_size": args.block_size,
        "prompt_source": "tokenized English benchmark prompts",
        "rows": rows,
    }
    args.results_file.write_text(json.dumps(result, indent=2) + "\n")

    best = max(rows, key=lambda row: float(row["cuda_graph_speedup"]))
    mean_speedup = (
        sum(float(row["cuda_graph_speedup"]) for row in rows) / len(rows)
    )
    print(f"\nMean CUDA-graph-only speedup: {mean_speedup:.2f}x")
    print(
        f"Best CUDA-graph-only speedup: {float(best['cuda_graph_speedup']):.2f}x "
        f"at batch {int(best['batch_size'])}"
    )
    print(f"Results saved to {args.results_file.resolve()}")


if __name__ == "__main__":
    main()
