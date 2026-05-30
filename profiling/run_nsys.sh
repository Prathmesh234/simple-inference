#!/usr/bin/env bash
# Nsight Systems capture of the attention kernel (hardware timeline).
#
# Requires `nsys` on PATH. If missing, install the NVIDIA Nsight Systems CLI:
#   https://developer.nvidia.com/nsight-systems   (or your distro's package).
#
# --capture-range=cudaProfilerApi pairs with cudaProfilerStart/Stop in
# profile_attention_nsys.py so ONLY the measured region is traced (model load
# and warmup are excluded -> small, readable trace).
#
# Output: profiling/out/attn_nsys.nsys-rep  (open in the Nsight Systems GUI).
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$PATH"
export XDG_CONFIG_HOME="$HOME/.cache/xdgconfig"
export UV_CACHE_DIR="$HOME/.cache/uv"

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found on PATH. Install Nsight Systems first." >&2
  exit 1
fi

OUT=profiling/out/attn_nsys
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$OUT" \
  uv run python profiling/profile_attention_nsys.py

echo "nsys report: ${OUT}.nsys-rep"
echo "Look for: kernel-to-kernel gaps in the decode NVTX range (memory stalls),"
echo "          dense back-to-back kernels in the prefill range (compute-bound)."
