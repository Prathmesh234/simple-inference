"""
Profiling Level 4 — NVTX annotations + Nsight Systems (nsys)

Nsight Systems is NVIDIA's system-wide GPU profiler. It captures:
  - Every CUDA kernel launch (name, duration, SM occupancy)
  - CPU thread activity and CUDA API calls
  - Memory transfers (H2D, D2H, D2D)
  - NVLink and PCIe traffic
  - NVTX user-defined ranges and markers

NVTX (NVIDIA Tools Extension) is how you annotate your Python code
so that nsys shows meaningful region names instead of raw kernel addresses.

The recommended workflow (torch.profiler first, then nsys):
  1. torch.profiler → identifies the hot op (e.g. "attention takes 60% of decode")
  2. nsys          → shows GPU utilization, SM occupancy, memory bandwidth
                     for that op at the hardware level
  3. ncu (Nsight Compute) → drills into a single kernel's register usage,
     cache hit rate, warp efficiency — only after you know which kernel to target

How to run this script under nsys:
    nsys profile \\
        --trace=cuda,nvtx,osrt \\
        --capture-range=cudaProfilerApi \\
        --output=profiling/nsys_output/run \\
        python -m profiling.04_nsys

    # --capture-range=cudaProfilerApi:
    #   nsys only captures between cudaProfilerStart() and cudaProfilerStop().
    #   Without this flag nsys captures the entire process including model load,
    #   which produces a huge trace and buries the inference region.

    # --trace=cuda,nvtx,osrt:
    #   cuda  = kernel launches and memory copies
    #   nvtx  = user-defined ranges (our annotations)
    #   osrt  = OS runtime (thread scheduling, sleep, syscalls)

    # Open the resulting .nsys-rep in Nsight Systems GUI.

When run without nsys:
    python -m profiling.04_nsys
    (NVTX calls become no-ops; torch.profiler section still works)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import contextmanager
from pathlib import Path
import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity, record_function

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache

DEVICE     = "cuda"
DTYPE      = torch.bfloat16
MODEL_ID   = "meta-llama/Llama-3.2-3B"
TRACE_DIR  = Path(__file__).parent / "traces"
NSYS_DIR   = Path(__file__).parent / "nsys_output"


# ---------------------------------------------------------------------------
# NVTX helpers
# ---------------------------------------------------------------------------

@contextmanager
def nvtx_range(name: str):
    """
    Push/pop an NVTX range.

    In the nsys timeline this appears as a colored bar whose width = duration.
    Nesting is supported — deeper ranges appear as inner bars.
    The push/pop pair is exception-safe here via try/finally.
    """
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def nvtx_mark(msg: str):
    """
    Instant marker in the nsys timeline — a thin vertical line.
    Useful for "this is where the warmup ended" or "EOS token emitted".
    """
    nvtx.mark(msg)


# ---------------------------------------------------------------------------
# Annotated inference functions
# ---------------------------------------------------------------------------

def annotated_prefill(model: LlamaModel, cfg: ModelConfig):
    """
    Prefill with NVTX ranges around warmup and the measured region.

    The hierarchy:
      inference_session
        warmup                         ← discarded by nsys (before ProfilerStart)
      [cudaProfilerStart]
        profiled_region
          prefill_T256

    Using --capture-range=cudaProfilerApi in nsys means the timeline only
    shows the region between cudaProfilerStart/Stop — the warmup is invisible.
    This is the critical practice from the HF blog: don't let initialization
    pollute your hardware trace.
    """
    ids_warmup  = torch.randint(0, cfg.vocab_size, (1, 64),  device=DEVICE)
    ids_prefill = torch.randint(0, cfg.vocab_size, (1, 256), device=DEVICE)

    # Warmup outside cudaProfilerStart so nsys never sees it
    with nvtx_range("warmup"):
        with torch.no_grad():
            for _ in range(5):
                model(ids_warmup, start_pos=0)
        torch.cuda.synchronize()
    nvtx_mark("warmup_complete")

    # Tell nsys to start capturing — does nothing when running without nsys
    torch.cuda.cudart().cudaProfilerStart()

    with nvtx_range("profiled_region"):
        with torch.no_grad():
            with nvtx_range("prefill_T256"):
                out = model(ids_prefill, start_pos=0)
        torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()
    return out


def annotated_decode_loop(model: LlamaModel, cfg: ModelConfig, n_steps: int = 10):
    """
    Decode loop with per-step NVTX ranges.

    In the nsys timeline each decode_step_NN is a separate bar. Comparing
    their widths shows if decode latency grows as the KV cache fills.

    n_steps decode steps after a 128-token prefill.
    """
    T_prefill = 128
    kv = KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = 1,
        max_seq_len = T_prefill + n_steps + 4,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )

    # Prefill to populate cache — outside profiler capture range
    prefill_ids = torch.randint(0, cfg.vocab_size, (1, T_prefill), device=DEVICE)
    with torch.no_grad():
        with nvtx_range("prefill_cache_init"):
            logits = model(prefill_ids, start_pos=0, kv_cache=kv)
    torch.cuda.synchronize()

    # Warmup decode
    decode_ids = torch.randint(0, cfg.vocab_size, (1, 1), device=DEVICE)
    with torch.no_grad():
        for _ in range(3):
            model(decode_ids, start_pos=T_prefill, kv_cache=kv)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()

    with nvtx_range("decode_loop"):
        next_token = logits[:, -1:, :].argmax(dim=-1)
        with torch.no_grad():
            for step in range(n_steps):
                with nvtx_range(f"decode_step_{step:02d}"):
                    logits     = model(next_token, start_pos=T_prefill + step, kv_cache=kv)
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                # Synchronize inside the NVTX range so the range width reflects
                # true GPU time, not just CPU dispatch time.
                torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()
    del kv


def combined_torch_and_nvtx(model: LlamaModel, cfg: ModelConfig):
    """
    Use NVTX AND torch.profiler in the same run.

    NVTX ranges you push here appear both:
    - in the nsys timeline (when running under nsys)
    - inside the Chrome trace from torch.profiler

    This is the recommended dual-use pattern:
      - torch.profiler gives you key_averages() without launching nsys
      - nsys gives SM occupancy and memory bandwidth for the same trace

    Run this function under nsys to get both at once.
    """
    ids = torch.randint(0, cfg.vocab_size, (1, 128), device=DEVICE)

    with torch.no_grad():
        for _ in range(5):
            model(ids, start_pos=0)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with nvtx_range("torch_profiler_session"):
            with torch.no_grad():
                # NVTX and record_function are independent — use both for
                # maximum visibility across all tools.
                with nvtx_range("prefill_T128"):
                    with record_function("prefill_T128"):
                        model(ids, start_pos=0)

    torch.cuda.cudart().cudaProfilerStop()

    print("\n  --- Combined torch.profiler + NVTX: key averages ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = TRACE_DIR / "combined_nvtx_torch.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\n  Chrome trace: {trace_path}")


# ---------------------------------------------------------------------------
# Layer-level NVTX: annotate individual transformer blocks
# ---------------------------------------------------------------------------

def per_layer_nvtx(model: LlamaModel, cfg: ModelConfig):
    """
    Wrap each transformer block in its own NVTX range.

    In the nsys timeline you see 28 sequential bars labeled layer_00 … layer_27.
    This directly shows how attention vs MLP time splits per layer, and whether
    all layers take the same time (they should — if they don't, it's a
    scheduling artifact or a kernel launch stall worth investigating).

    Approach: register forward hooks on each TransformerBlock rather than
    modifying model code. Hooks fire at the boundary of each module's forward.
    """
    from model.block import TransformerBlock

    handles = []

    def pre_hook(layer_idx):
        def _hook(module, inp):
            nvtx.range_push(f"layer_{layer_idx:02d}")
        return _hook

    def post_hook(layer_idx):
        def _hook(module, inp, out):
            nvtx.range_pop()
        return _hook

    for i, block in enumerate(model.layers):
        if isinstance(block, TransformerBlock):
            handles.append(block.register_forward_pre_hook(pre_hook(i)))
            handles.append(block.register_forward_hook(post_hook(i)))

    ids = torch.randint(0, cfg.vocab_size, (1, 256), device=DEVICE)

    # Warmup before capture
    with torch.no_grad():
        for _ in range(3):
            model(ids, start_pos=0)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    with nvtx_range("per_layer_prefill_T256"):
        with torch.no_grad():
            model(ids, start_pos=0)
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    for h in handles:
        h.remove()


def main():
    print(f"\n{'='*70}")
    print("  Profiling Level 4 — NVTX + Nsight Systems")
    print(f"{'='*70}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model  = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    NSYS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}")
    print("  1. Annotated prefill (NVTX warmup guard + cudaProfilerStart/Stop)")
    print(f"{'─'*70}")
    annotated_prefill(model, cfg)
    print("     NVTX ranges emitted. Run under nsys to see the timeline.")

    print(f"\n{'─'*70}")
    print("  2. Annotated decode loop (10 steps, per-step NVTX ranges)")
    print(f"{'─'*70}")
    annotated_decode_loop(model, cfg, n_steps=10)
    print("     NVTX ranges emitted.")

    print(f"\n{'─'*70}")
    print("  3. Combined torch.profiler + NVTX")
    print(f"{'─'*70}")
    combined_torch_and_nvtx(model, cfg)

    print(f"\n{'─'*70}")
    print("  4. Per-layer NVTX (forward hook per TransformerBlock)")
    print(f"{'─'*70}")
    per_layer_nvtx(model, cfg)
    print("     28 layer NVTX ranges emitted.")

    print(f"\n{'='*70}")
    print("  nsys commands:")
    print()
    print("  # Capture only the profiled region (recommended):")
    print("  nsys profile \\")
    print("      --trace=cuda,nvtx,osrt \\")
    print("      --capture-range=cudaProfilerApi \\")
    print("      --output=profiling/nsys_output/run \\")
    print("      python -m profiling.04_nsys")
    print()
    print("  # Full process trace (includes model load — large file):")
    print("  nsys profile --trace=cuda,nvtx python -m profiling.04_nsys")
    print()
    print("  # ncu: single-kernel deep-dive (pick a kernel from nsys first):")
    print("  ncu --set full -o profiling/nsys_output/kernel python -m profiling.04_nsys")
    print()
    print("  Open .nsys-rep in Nsight Systems GUI.")
    print("  Open .ncu-rep in Nsight Compute GUI.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
