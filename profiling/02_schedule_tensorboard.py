"""
Profiling Level 2 — Schedule API + record_function + TensorBoard

Key learnings from the HF blog:

1. schedule(wait, warmup, active, repeat)
   The profiler fires in cycles. Each cycle:
     - wait   N steps: profiler is OFF (no overhead, no data)
     - warmup N steps: profiler is ON, data is thrown away (fills GPU caches)
     - active N steps: profiler is ON, data is kept → this is your trace
   repeat=0 means run cycles forever until you call profiler.stop().

   Why bother? Running the profiler on step 0 catches cuDNN plan selection,
   Triton JIT compile, and CUDA context init — all one-time costs that inflate
   every op's apparent latency. Schedule skips those automatically.

2. on_trace_ready=tensorboard_trace_handler(dir)
   After each active cycle, the profiler writes a .pt.trace.json to dir.
   Launch TensorBoard to view the timeline, flame graph, and memory curves:
       tensorboard --logdir=profiling/tb_logs

3. record_function("name") — named spans
   Any code inside record_function() appears as a labeled region in the
   TensorBoard trace and Chrome trace. Use it to annotate prefill vs decode,
   individual layers, or any logical grouping you care about.
   It also shows up in key_averages() so you can measure the span's total time.

4. record_shapes=True
   Without this, key_averages(group_by_input_shape=True) gives empty buckets.
   Turn it on when you want to see how matmul performance varies with (B, T, H).

5. with_flops=True
   The profiler estimates FLOPs for matmul and conv ops. Combined with
   timing, you get achieved TFLOPS per op — useful for roofline analysis.

Run:
    python -m profiling.02_schedule_tensorboard
    tensorboard --logdir=profiling/tb_logs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
    record_function,
)

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache

DEVICE  = "cuda"
DTYPE   = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"
LOG_DIR  = Path(__file__).parent / "tb_logs"


def make_kv_cache(cfg: ModelConfig, t_prefill: int) -> KVCache:
    return KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = 1,
        max_seq_len = t_prefill + 64,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )


def simulate_generation(model: LlamaModel, cfg: ModelConfig, prof):
    """
    One prefill + 9 decode steps. Call prof.step() after each logical step
    so the scheduler knows which phase we're in.

    The schedule below (wait=1, warmup=1, active=3) means:
      - step 0 (wait):   discarded — catches model's first-pass overhead
      - step 1 (warmup): discarded — GPU caches are warm after this
      - steps 2-4 (active): kept — these appear in TensorBoard

    Steps 5-9 fall outside the active window (repeat=1 means one cycle only).
    """
    T_prefill = 256
    N_decode  = 9

    ids = torch.randint(0, cfg.vocab_size, (1, T_prefill), device=DEVICE)
    kv  = make_kv_cache(cfg, T_prefill)

    with torch.no_grad():
        # Step 0 — wait
        with record_function("prefill"):
            logits = model(ids, start_pos=0, kv_cache=kv)
        prof.step()  # scheduler sees step 0 → wait phase, no trace

        # Steps 1-9 — decode
        next_token = logits[:, -1:, :].argmax(dim=-1)
        for i in range(N_decode):
            with record_function(f"decode_step_{i:02d}"):
                logits     = model(next_token, start_pos=T_prefill + i, kv_cache=kv)
                next_token = logits[:, -1:, :].argmax(dim=-1)
            prof.step()  # scheduler advances: warmup at step1, active at steps 2-4

    del kv


def profile_with_schedule(model: LlamaModel, cfg: ModelConfig):
    """
    Full scheduled profiling run. TensorBoard trace written after the active
    window closes.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # schedule: skip 1, warmup 1, capture 3, repeat once
    my_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=tensorboard_trace_handler(str(LOG_DIR)),
        record_shapes=True,
        with_flops=True,
    ) as prof:
        simulate_generation(model, cfg, prof)

    return prof


def profile_grouped_by_shape(model: LlamaModel, cfg: ModelConfig):
    """
    Compare the same op at prefill vs decode shape using group_by_input_shape.
    This shows concretely why prefill (large T) has different cost than decode (T=1).
    """
    kv = make_kv_cache(cfg, t_prefill=256)

    # Warmup
    ids_pre = torch.randint(0, cfg.vocab_size, (1, 256), device=DEVICE)
    ids_dec = torch.randint(0, cfg.vocab_size, (1, 1),   device=DEVICE)
    with torch.no_grad():
        for _ in range(3):
            model(ids_pre, start_pos=0)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            with record_function("prefill_T256"):
                model(ids_pre, start_pos=0, kv_cache=kv)
            with record_function("decode_T1"):
                model(ids_dec, start_pos=256, kv_cache=kv)

    print("\n  --- key_averages(group_by_input_shape=True): top 15 by CUDA time ---")
    print("  (shows how the same op e.g. aten::mm looks different at prefill vs decode shape)")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))

    del kv
    return prof


def main():
    print(f"\n{'='*70}")
    print("  Profiling Level 2 — Schedule + TensorBoard + record_function")
    print(f"{'='*70}")

    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model  = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    print(f"\n{'─'*70}")
    print("  1. Scheduled profiling (wait=1, warmup=1, active=3)")
    print(f"{'─'*70}")
    prof = profile_with_schedule(model, cfg)

    print(f"\n  TensorBoard trace → {LOG_DIR}/")
    print("  Launch: tensorboard --logdir=profiling/tb_logs\n")

    # Print summary from the captured active steps
    print("  --- Summary from scheduled active steps (top 15 by CUDA time) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print(f"\n{'─'*70}")
    print("  2. Group-by-input-shape: prefill vs decode")
    print(f"{'─'*70}")
    profile_grouped_by_shape(model, cfg)

    print(f"\n{'='*70}")
    print("  What to look for in TensorBoard:")
    print("    Operator view → sort by CUDA Total to find the hot kernel")
    print("    Trace view    → timeline of CPU dispatch + GPU kernel execution")
    print("    Memory view   → peak allocation per operator")
    print("    Group by shape → see the same matmul at different (B,T,H)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
