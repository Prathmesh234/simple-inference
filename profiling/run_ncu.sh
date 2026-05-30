#!/usr/bin/env bash
# Nsight Compute single-kernel deep dive on the FlashAttention Triton kernel.
#
# Requires `ncu` on PATH (NVIDIA Nsight Compute). Use this AFTER nsys has told
# you which kernel is the bottleneck — ncu is slow (it replays each kernel many
# times) so you target one kernel, not a whole run.
#
# --kernel-name regex restricts profiling to our Triton kernel (_flash_fwd...);
# --launch-count limits how many launches are captured. `--set full` collects
# the complete metric set (registers, cache hit rates, warp efficiency, memory
# throughput) — the "why is this kernel slow internally" answer.
#
# NOTE: ncu usually needs elevated GPU perf-counter permissions (run as root or
# set NVIDIA's `--allow-non-admin` / driver option `NVreg_RestrictProfiling=0`).
#
# Output: profiling/out/attn_ncu.ncu-rep  (open in the Nsight Compute GUI).
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"
export XDG_CONFIG_HOME="$HOME/.cache/xdgconfig"
export UV_CACHE_DIR="$HOME/.cache/uv"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not found on PATH. Install Nsight Compute first." >&2
  exit 1
fi

OUT=profiling/out/attn_ncu
ncu \
  --set full \
  --kernel-name "regex:_flash_fwd" \
  --launch-count 4 \
  --target-processes all \
  --force-overwrite \
  --export "$OUT" \
  uv run python profiling/profile_attention_nsys.py

echo "ncu report: ${OUT}.ncu-rep"
echo "Look for: achieved occupancy, registers/thread (spills?), L2 hit rate,"
echo "          and DRAM throughput vs the 960 GB/s ceiling."
