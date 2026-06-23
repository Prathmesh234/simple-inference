"""
Section 19 benchmark: batched decode — eager ragged SDPA vs CUDA graphs.

What this measures
------------------
The continuous-batching engine's DECODE step is memory-bound and (per the
profiler) CPU-launch-bound: ~300 tiny kernels per token across 28 layers. A
CUDA graph collapses that launch stream into one replay. This benchmark
quantifies the win at several batch sizes by timing the engine's own
`_decode_batch` over a steady-state batch of requests.

Two regimes, identical work
---------------------------
For each batch size B we build B requests already advanced to a fixed context
length (prefilled), then time ONE decode step repeatedly:

  - eager  : InferenceEngine(use_cuda_graphs=False) → ragged SDPA path
  - graph  : InferenceEngine(use_cuda_graphs=True)  → replay the captured
             per-batch-size CUDA graph (batch already equals a bucket, so no
             padding)

Both run the same 28-layer forward and the same sampling; only the decode
kernel-launch path differs. We report median per-step latency (ms), the
implied decode throughput (tokens/s = B / step_latency), and the speedup.

Run:
    XDG_CONFIG_HOME=~/.cache/xdgconfig UV_CACHE_DIR=~/.cache/uv \
    PATH=~/.local/bin:$PATH uv run python -m benchmarks.bench_engine_cudagraph
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import env_loader  # noqa: F401  loads .env (HF_TOKEN)
import torch
import triton

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from serving.engine import InferenceEngine
from serving.request import Request, RequestState

MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Steady-state context length each request sits at when we time decode. The
# decode cost is dominated by the per-layer launches, not this length, but we
# keep it realistic so the masked full-cache read does meaningful work.
CONTEXT_LEN = 512
# Batch sizes to sweep — all powers of two so the graph path needs no padding
# (each B is exactly a captured bucket).
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
MAX_SEQ = CONTEXT_LEN + 64


def build_model() -> LlamaModel:
    print("Loading model...")
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()
    return model


def make_steady_requests(engine: InferenceEngine, B: int, context_len: int) -> list[Request]:
    """
    Build B requests sitting in DECODE state at absolute position `context_len`.

    We drive the engine's real prefill so each request's KV cache is genuinely
    populated [0, context_len), then leave them in steady state ready to decode.
    Returns the running requests (already holding slots).
    """
    engine.reset()
    vocab = engine.cfg.vocab_size
    prompt = [engine.cfg.bos_token_id] + [(i * 7 + 1) % vocab for i in range(context_len - 1)]
    # Budget just has to fit the cache; we reset pos before every timed step so a
    # request never actually advances past `context_len`.
    budget = engine.max_seq_len - context_len
    for _ in range(B):
        engine.add_request(list(prompt), max_new_tokens=budget)
    # Run steps until all B are admitted and prefilled (now in DECODE state).
    for _ in range(B + 4):
        engine.step()
        running = [r for r in engine.scheduler.running if r.state is RequestState.DECODE]
        if len(running) == B:
            break
    running = [r for r in engine.scheduler.running if r.state is RequestState.DECODE]
    assert len(running) == B, f"expected {B} decode requests, got {len(running)}"
    return running


@torch.no_grad()
def time_decode(engine: InferenceEngine, reqs: list[Request], rep: int = 100) -> float:
    """Median ms for one engine decode step over `reqs` (no request ever finishes)."""
    # Snapshot positions so each timed step re-decodes from the same state and
    # nobody hits its (huge) token budget or overflows the cache.
    base_pos = [r.pos for r in reqs]

    def one_step():
        for r, p in zip(reqs, base_pos):
            r.pos = p
            r.eos_hit = False
        engine._decode_batch(reqs)

    return triton.testing.do_bench(one_step, warmup=25, rep=rep)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    model = build_model()

    print(f"\nDecode benchmark: eager vs CUDA graph")
    print(f"context_len={CONTEXT_LEN}, max_seq={MAX_SEQ}, batch sizes={BATCH_SIZES}\n")

    # One engine per regime; reuse across batch sizes (max_running = biggest B).
    max_running = max(BATCH_SIZES)
    print("Building eager engine...")
    eager = InferenceEngine(
        model=model, max_running=max_running, max_seq_len=MAX_SEQ,
        temperature=0.0, warmup=True, use_cuda_graphs=False,
    )
    print("Building CUDA-graph engine (capturing all buckets)...")
    graph = InferenceEngine(
        model=model, max_running=max_running, max_seq_len=MAX_SEQ,
        temperature=0.0, warmup=True, use_cuda_graphs=True,
    )

    header = f"{'batch':>6} | {'eager ms':>9} | {'graph ms':>9} | {'speedup':>8} | {'eager tok/s':>12} | {'graph tok/s':>12}"
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for B in BATCH_SIZES:
        eager_reqs = make_steady_requests(eager, B, CONTEXT_LEN)
        eager_ms = time_decode(eager, eager_reqs)

        graph_reqs = make_steady_requests(graph, B, CONTEXT_LEN)
        graph_ms = time_decode(graph, graph_reqs)

        speedup = eager_ms / graph_ms
        eager_tps = B / (eager_ms * 1e-3)
        graph_tps = B / (graph_ms * 1e-3)
        rows.append((B, eager_ms, graph_ms, speedup, eager_tps, graph_tps))
        print(f"{B:>6} | {eager_ms:>9.3f} | {graph_ms:>9.3f} | {speedup:>7.2f}x | "
              f"{eager_tps:>12.1f} | {graph_tps:>12.1f}")

    print("\nSummary")
    print("-------")
    best = max(rows, key=lambda r: r[3])
    print(f"  Best speedup: {best[3]:.2f}x at batch size {best[0]}")
    print(f"  (eager {best[1]:.3f} ms/step → graph {best[2]:.3f} ms/step)")
    avg_speedup = sum(r[3] for r in rows) / len(rows)
    print(f"  Mean speedup across batch sizes: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
