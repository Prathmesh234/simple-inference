"""
Generic torch.profiler driver — profile ANY registered kernel by name.

All the per-kernel knowledge (input shapes, the callable, prefill/decode
regimes) lives in profile_utils.KERNELS. This script is just the CLI: pick a
kernel and it prints the key_averages tables (sorted by cuda_time_total and
self_cuda_time_total) and saves them to profiling/profile-kernels/torch-profiler/out/profiler_<name>_1.txt.

List the kernels you can profile:
    uv run python profiling/profile-kernels/torch-profiler/profile_kernel_torch.py --list

Profile one (default: attention):
    PATH="$HOME/.local/bin:$PATH" XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
    UV_CACHE_DIR="$HOME/.cache/uv" \
    uv run python profiling/profile-kernels/torch-profiler/profile_kernel_torch.py rmsnorm

Profile several in one go:
    uv run python profiling/profile-kernels/torch-profiler/profile_kernel_torch.py rmsnorm rope swiglu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This script lives in profile-kernels/torch-profiler/; profile_utils.py is one
# directory up. Make it importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profile_utils import kernel_names, run_torch_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kernels", nargs="*", default=["attention"],
                   help=f"kernel name(s) to profile. choices: {', '.join(kernel_names())}")
    p.add_argument("--list", action="store_true", help="list profilable kernels and exit")
    args = p.parse_args()

    if args.list:
        print("Profilable kernels:")
        for name in kernel_names():
            print(f"  - {name}")
        return

    for name in args.kernels:
        run_torch_profile(name)


if __name__ == "__main__":
    main()
