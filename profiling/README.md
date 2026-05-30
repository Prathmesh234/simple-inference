# Attention Kernel Profiling — deep-dive setup

Everything here targets **one** kernel: the FlashAttention-2 Triton kernel in
`kernels/attention_kernel.py`. This is the Section 13.5 profiling phase: the
benchmarks tell you *how fast*, profiling tells you *where the time goes and why*.

This folder was set up ahead of a hands-on session. The torch-only scripts run
today with zero extra installs; nsys/ncu/matplotlib are optional and documented
below.

## The golden rule: warmup
The first kernel calls carry one-time costs that inflate latency **10–100×**:
- Triton **JIT compilation** of each new (shape, dtype) specialization
- The **autotune sweep** (16 configs) on the first call per `(D, CAUSAL)` key
- cuBLAS/cuDNN plan selection for the SDPA reference

Every script calls `warmup()` before the measured/captured region. Never trust
a number taken from a cold kernel.

## Two regimes (always profile them separately)
| Regime  | Shape                | Looks like              | Expected bound |
|---------|----------------------|-------------------------|----------------|
| prefill | `Tq = Tk = T`, causal| big square score tiles  | compute-bound as T grows |
| decode  | `Tq = 1`, `Tk` grows | one query, long KV read | always memory-bound |

That difference is the whole story behind KV-cache layout, CUDA graphs, and
quantization later.

## How to run (env)
System python has no torch/triton — go through `uv`. Every script is launched
the same way:

```bash
PATH="$HOME/.local/bin:$PATH" \
XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
UV_CACHE_DIR="$HOME/.cache/uv" \
uv run python profiling/<script>.py
```

(The `XDG_CONFIG_HOME` / `UV_CACHE_DIR` vars are only a uv launcher workaround
on this box — `~/.config` is root-owned. They do **not** affect the kernels.)

## Scripts (run in this order)

| # | File | Tool | What it answers | Needs |
|---|------|------|-----------------|-------|
| 1 | `profile_attention_torch.py` | `torch.profiler` | Which op dominates? `cuda_time` vs `self_cuda_time`. | built-in |
| 2/3 | `profile_attention_trace.py` | `torch.profiler` schedule + chrome trace + memory | Steady-state op table; GPU timeline; per-op allocations. | built-in |
| – | `roofline_attention.py` | `triton.do_bench` | Compute- vs memory-bound per shape; achieved TFLOPS/GB/s vs peak. | built-in (PNG needs matplotlib) |
| 4 | `profile_attention_nsys.py` | NVTX + cudaProfilerApi | The script you launch under nsys/ncu. | runs bare too |
| – | `run_nsys.sh` | Nsight Systems | Hardware timeline, kernel gaps, occupancy. | `nsys` |
| – | `run_ncu.sh` | Nsight Compute | Single-kernel registers/cache/warp/DRAM. | `ncu` |

`profile_utils.py` holds shared constants, input builders, roofline math,
warmup, and the NVTX / cudaProfiler context managers.

Outputs (traces, PNGs, reports) land in `profiling/out/`.

## Optional tool installs (for tomorrow)
Not installed on this box — install only what you reach:

```bash
# Roofline PNG (otherwise a text table is printed):
uv pip install matplotlib

# TensorBoard timeline view (optional, Step 2):
uv pip install tensorboard torch-tb-profiler

# Nsight Systems / Compute CLIs (system packages, not pip):
#   https://developer.nvidia.com/nsight-systems
#   https://developer.nvidia.com/nsight-compute
# ncu needs GPU perf-counter permission (root, or NVreg_RestrictProfiling=0).
```

## Tool decision guide
| Question | Tool |
|---|---|
| Which op takes the most time? | `torch.profiler` key_averages |
| Compute- or memory-bound? | `roofline_attention.py` / nsys occupancy + gaps |
| Which tensor eats VRAM? | `profile_memory=True` → memory_viz snapshot |
| Is Python dispatch the bottleneck? | nsys CPU thread vs GPU stream |
| How many FLOPs per op? | `torch.profiler` `with_flops=True` |
| Why is this kernel slow internally? | `ncu` registers / cache / warp metrics |

## A note carried over from the autotune fix
`_flash_fwd` is autotuned with `key=["D", "CAUSAL"]` — deliberately **not**
`Tk`. Including `Tk` (which grows every decode token) forced a full re-tune per
token (~1.7s each). Final config selection / persistence is intentionally
deferred to the CUDA-graphs phase, where fixed block sizes are needed for graph
capture. Profile first, then pin configs.
