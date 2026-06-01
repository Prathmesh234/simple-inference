"""
Generic Nsight Systems / Compute target — NVTX-annotated run of ANY registered
kernel by name. Launch this UNDER nsys or ncu (see run_nsys.sh / run_ncu.sh).

Like the old per-kernel nsys script but kernel-agnostic: it warms up OUTSIDE the
captured region and wraps each prefill/decode regime in an NVTX range, so the
nsys/ncu timeline shows e.g. "rmsnorm_prefill_step0" next to the raw Triton
kernel name. All kernel shapes/callables come from profile_utils.KERNELS.

NVTX ranges and cudaProfilerStart are no-ops when no profiler is attached, so a
bare run is harmless and just validates the script works.

List the kernels:
    uv run python profiling/profile-kernels/nsys-profiler/profile_kernel_nsys.py --list

Bare run (sanity check, no profiler), default attention:
    uv run python profiling/profile-kernels/nsys-profiler/profile_kernel_nsys.py rmsnorm

Under nsys (set KERNEL to pick which one — see run_nsys.sh):
    KERNEL=rmsnorm bash profiling/profile-kernels/nsys-profiler/run_nsys.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This script lives in profile-kernels/nsys-profiler/; profile_utils.py is one
# directory up. Make it importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profile_utils import kernel_names, run_nsys_capture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Allow either a positional name or the KERNEL env var (used by run_nsys.sh).
    p.add_argument("kernel", nargs="?",
                   default=os.environ.get("KERNEL", "attention"),
                   help=f"kernel to capture. choices: {', '.join(kernel_names())}")
    p.add_argument("--list", action="store_true", help="list profilable kernels and exit")
    args = p.parse_args()

    if args.list:
        print("Profilable kernels:")
        for name in kernel_names():
            print(f"  - {name}")
        return

    run_nsys_capture(args.kernel)


if __name__ == "__main__":
    main()
