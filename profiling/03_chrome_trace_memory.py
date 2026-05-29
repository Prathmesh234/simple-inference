"""
Profiling Level 3 — Chrome trace + memory profiling + memory snapshot

Key learnings from the HF blog:

1. export_chrome_trace(path)
   Writes a JSON trace viewable in chrome://tracing or ui.perfetto.dev.
   Shows: CPU thread timeline, CUDA stream timeline, kernel names, durations.
   More detail than TensorBoard's Trace view — kernel names come directly from
   the CUDA driver, not PyTorch's op registry.

2. profile_memory=True
   Tracks every tensor allocation and free during the profiled region.
   key_averages() then exposes "self_cpu_memory_usage" and "cuda_memory_usage"
   per op — tells you *which op allocates the most memory*, not just which is slow.
   Critical for diagnosing OOM during long prefills or large-batch decode.

3. with_stack=True
   Captures the Python call stack at the point each op is dispatched.
   Makes the Chrome trace navigable: click any kernel and see the Python line
   that triggered it.
   Expensive (~30% overhead) — use only when you need stack attribution,
   not in production benchmarks.

4. Memory snapshot API (separate from profiler)
   torch.cuda.memory._record_memory_history() captures a full allocation
   history with per-tensor stack traces. Richer than profile_memory=True:
   shows which *tensor* was allocated, its shape, dtype, and full call stack.
   Visualize the .pickle at: https://pytorch.org/memory_viz
   Use this when you need to understand peak VRAM composition
   (weights vs activations vs KV cache vs temporaries).

Run:
    python -m profiling.03_chrome_trace_memory
    # Open profiling/traces/*.json in chrome://tracing
    # Upload profiling/traces/memory_snapshot.pickle to pytorch.org/memory_viz
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache

DEVICE    = "cuda"
DTYPE     = torch.bfloat16
MODEL_ID  = "meta-llama/Llama-3.2-3B"
TRACE_DIR = Path(__file__).parent / "traces"


def _warmup(model: LlamaModel, cfg: ModelConfig, seq_len: int = 128, steps: int = 5):
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=DEVICE)
    with torch.no_grad():
        for _ in range(steps):
            model(ids, start_pos=0)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 1. Prefill Chrome trace + memory breakdown
# ---------------------------------------------------------------------------

def profile_prefill_chrome(model: LlamaModel, cfg: ModelConfig, seq_len: int = 512):
    """
    Full-options profiler on a prefill → Chrome trace + memory table.

    with_stack=True is ON here so you can navigate from any CUDA kernel
    back to the Python line that dispatched it.
    """
    ids  = torch.randint(0, cfg.vocab_size, (1, seq_len), device=DEVICE)
    path = TRACE_DIR / f"prefill_T{seq_len}.json"

    _warmup(model, cfg, seq_len=seq_len)

    print(f"  Profiling prefill T={seq_len}  →  {path}")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,    # track cuda/cpu allocations per op
        with_stack=True,        # Python callstack per dispatch
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            with record_function(f"prefill_T{seq_len}"):
                model(ids, start_pos=0)

    prof.export_chrome_trace(str(path))
    print(f"  Chrome trace written — open in chrome://tracing or ui.perfetto.dev\n")

    # Memory-sorted table: which ops allocate the most GPU memory?
    print("  --- Top 10 ops by CUDA memory allocation ---")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    # Self-time table: strips out child op cost, isolates each kernel's own cost
    print("\n  --- Top 10 ops by self_cuda_time ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # FLOPs table: only ops that support FLOP counting show non-zero values
    print("\n  --- Top 10 ops by FLOP count ---")
    print(prof.key_averages().table(sort_by="flops", row_limit=10))


# ---------------------------------------------------------------------------
# 2. Decode Chrome trace (very different profile from prefill)
# ---------------------------------------------------------------------------

def profile_decode_chrome(model: LlamaModel, cfg: ModelConfig, t_prefill: int = 128):
    """
    Chrome trace for a single decode step.

    Decode is memory-bound; expect to see short CUDA kernels with large gaps
    (memory latency) rather than the long matmul tiles visible in prefill.
    """
    kv = KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = 1,
        max_seq_len = t_prefill + 16,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )
    prefill_ids = torch.randint(0, cfg.vocab_size, (1, t_prefill), device=DEVICE)
    with torch.no_grad():
        model(prefill_ids, start_pos=0, kv_cache=kv)

    decode_ids = torch.randint(0, cfg.vocab_size, (1, 1), device=DEVICE)
    with torch.no_grad():
        for _ in range(5):
            model(decode_ids, start_pos=t_prefill, kv_cache=kv)
    torch.cuda.synchronize()

    path = TRACE_DIR / "decode_T1.json"
    print(f"  Profiling decode T=1  →  {path}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            with record_function("decode_step"):
                model(decode_ids, start_pos=t_prefill, kv_cache=kv)

    prof.export_chrome_trace(str(path))
    print("  Chrome trace written\n")

    print("  --- Decode T=1: top 10 by self_cuda_time ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    del kv


# ---------------------------------------------------------------------------
# 3. Memory snapshot — full allocation history
# ---------------------------------------------------------------------------

def memory_snapshot(model: LlamaModel, cfg: ModelConfig):
    """
    Capture a complete tensor-level allocation history.

    The snapshot is richer than profile_memory=True — it records:
      - every tensor ever allocated (shape, dtype, device)
      - the Python stack at allocation time
      - whether the tensor was freed or is still live

    How to read it:
      1. Upload profiling/traces/memory_snapshot.pickle to pytorch.org/memory_viz
      2. The flame graph shows allocation sources by call stack
      3. The timeline shows peak and how memory was consumed over time

    When to use this vs profile_memory=True:
      - profile_memory=True → "which *op* allocated the most?"
      - memory_snapshot     → "which *tensor* is using 2 GB, and where was it created?"
    """
    path = TRACE_DIR / "memory_snapshot.pickle"
    print(f"  Memory snapshot  →  {path}")

    ids = torch.randint(0, cfg.vocab_size, (1, 512), device=DEVICE)
    _warmup(model, cfg, seq_len=512, steps=3)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.memory._record_memory_history(max_entries=100_000)

    with torch.no_grad():
        with record_function("snapshot_prefill_T512"):
            model(ids, start_pos=0)
    torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot(str(path))
    torch.cuda.memory._record_memory_history(enabled=None)   # stop recording

    peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
    reserved_mb = torch.cuda.max_memory_reserved(DEVICE) / 1e6
    print(f"  Peak allocated : {peak_mb:.0f} MB")
    print(f"  Peak reserved  : {reserved_mb:.0f} MB  (reserved > allocated = fragmentation)")
    print("  Visualize at   : https://pytorch.org/memory_viz\n")


def main():
    print(f"\n{'='*70}")
    print("  Profiling Level 3 — Chrome trace + memory profiling")
    print(f"{'='*70}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model  = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}")
    print("  1. Prefill (T=512) — Chrome trace + memory")
    print(f"{'─'*70}")
    profile_prefill_chrome(model, cfg, seq_len=512)

    print(f"{'─'*70}")
    print("  2. Decode (T=1) — Chrome trace")
    print(f"{'─'*70}")
    profile_decode_chrome(model, cfg, t_prefill=128)

    print(f"{'─'*70}")
    print("  3. Memory snapshot")
    print(f"{'─'*70}")
    memory_snapshot(model, cfg)

    print(f"{'='*70}")
    print(f"  Traces written to: {TRACE_DIR}/")
    print("  Chrome trace:    open *.json in chrome://tracing")
    print("  Memory snapshot: upload memory_snapshot.pickle to pytorch.org/memory_viz")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
