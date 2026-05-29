"""
Profiling toolkit — torch.profiler + Nsight Systems (nsys)

Progression (simple → complex):
  01_basic.py               — warmup + minimal profiler, key_averages table
  02_schedule_tensorboard.py — schedule API, record_function, TensorBoard
  03_chrome_trace_memory.py  — Chrome trace, memory profiling, memory snapshot
  04_nsys.py                 — NVTX annotations, cudaProfilerStart/Stop,
                               combined torch.profiler + nsys workflow

Run any level standalone:
    python -m profiling.01_basic
    python -m profiling.02_schedule_tensorboard
    python -m profiling.03_chrome_trace_memory
    python -m profiling.04_nsys

For nsys (level 4):
    nsys profile --trace=cuda,nvtx,osrt \\
                 --capture-range=cudaProfilerApi \\
                 --output=profiling/nsys_output/run \\
                 python -m profiling.04_nsys
"""
