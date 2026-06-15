"""
profile_engine_torch.py — torch.profiler over the EXACT iteration-03 engine.

Where profile-kernels/ isolates a single Triton kernel, this profiles the whole
generation path end to end: token embedding, every transformer block (attention
+ RMSNorm + RoPE + SwiGLU MLP), the LM head, and sampling.

ENGINE PARITY
-------------
`prefill_and_decode` below is a byte-for-byte copy of the generation engine in
iterations/03_engine.py (our first engine iteration) — same constants, same
model / tokenizer / KV-cache setup, same sampling, same "prefill once then
decode exactly N tokens (no EOS early-stop)" loop, same `next_tok` reuse. The
profiled code path is therefore identical to 03; the profiler captures it via
the standard `profile(...)` context, not by altering the engine.

It splits the timeline into the two regimes that matter for serving latency:
  - PREFILL : one forward over the whole prompt (sets Time-To-First-Token)
  - DECODE  : the per-token autoregressive steps (sets Time-Per-Output-Token)

It sweeps four prompt "flavors" from profile-engine/prompt.json (short,
medium_short, medium_long, long) so you can see how prefill cost grows with
prompt length while per-token decode stays roughly flat.

For each flavor it writes, into profile-engine/torch-profiler/out/:
  - profiler_engine_<flavor>.txt        key_averages table (CPU+CUDA)
  - engine_<flavor>_trace.json          Chrome/Perfetto timeline
and prints prefill_ms / decode_ms-per-step measured the same way as 03.

Requires the gated Llama-3.2-3B weights, so set HF_TOKEN (see .env.example).

Run (all flavors):
    PATH="$HOME/.local/bin:$PATH" XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
    UV_CACHE_DIR="$HOME/.cache/uv" \
    uv run python profiling/profile-engine/torch-profiler/profile_engine_torch.py

One flavor:
    uv run python profiling/profile-engine/torch-profiler/profile_engine_torch.py --flavors short
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# This file lives at <repo>/profiling/profile-engine/torch-profiler/, so the
# repo root is three parents up. Make the engine modules importable when run as
# a script.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_loader  # noqa: F401,E402  reads .env so USE_TRITON / HF_TOKEN are set before imports
import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from config import ModelConfig  # noqa: E402
from loader import WeightLoader  # noqa: E402
from model.llama import LlamaModel  # noqa: E402
import model.llama as llama_mod  # noqa: E402
from model.kv_cache import KVCache  # noqa: E402
from sampling import sample  # noqa: E402
import ops.rmsnorm as rmsnorm_mod  # noqa: E402

# ── constants (identical to iterations/03_engine.py) ─────────────────────────
DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.2-3B"
WARMUP   = 2

# Sampling knobs (identical to 03)
TEMPERATURE = 0.7
TOP_K       = 50
TOP_P       = 0.9

PROMPT_FILE = THIS_DIR.parent / "prompt.json"
OUT_DIR = THIS_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TensorBoard logdir lives in the sibling reserved folder. The torch_tb_profiler
# plugin reads the *.pt.trace.json files written here via tensorboard_trace_handler.
TB_DIR = THIS_DIR.parent / "tensorboard-profiler" / "logdir"

ACTIVITIES = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
FLAVOR_ORDER = ["short", "medium_short", "medium_long", "long"]


# ── KV-cache generation with sampling ────────────────────────────────────────
# Byte-for-byte copy of iterations/03_engine.py::prefill_and_decode so the
# profiled path is the real engine, not a re-interpretation of it.

@torch.no_grad()
def prefill_and_decode(
    model: LlamaModel,
    kv_cache: KVCache,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
) -> tuple[torch.Tensor, float, float]:
    """
    Prefill + N sampled decode steps.

    Sampling config: temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P.
    Everything else (timing, KV cache usage) is identical to 02_kv_cache.py.
    """
    kv_cache.reset()
    B, T = prompt_ids.shape

    # --- prefill --------------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(prompt_ids, start_pos=0, kv_cache=kv_cache)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # First new token — sampled, not greedy
    next_tok = sample(
        logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P
    ).unsqueeze(-1)  # (B, 1)
    generated = [next_tok]

    # --- decode loop ----------------------------------------------------------
    decode_step_times = []
    pos = T
    for _ in range(n_new_tokens - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits   = model.decode_step(next_tok, pos, kv_cache)
        next_tok = sample(
            logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P
        ).unsqueeze(-1)
        torch.cuda.synchronize()
        decode_step_times.append((time.perf_counter() - t0) * 1000)
        generated.append(next_tok)
        pos += 1

    decode_ms_avg = (
        sum(decode_step_times) / len(decode_step_times) if decode_step_times else 0.0
    )
    return torch.cat(generated, dim=1), prefill_ms, decode_ms_avg


# ── setup (mirrors iterations/03_engine.py __main__) ─────────────────────────

def load_prompts() -> dict:
    with open(PROMPT_FILE) as f:
        return json.load(f)["prompts"]


def tokenize_one(text: str, tokenizer) -> torch.Tensor:
    """No truncation — long prompts are intentional (matches 03's tokenize_batch)."""
    enc = tokenizer([text], return_tensors="pt", padding=True)
    return enc["input_ids"].to(DEVICE)


def tokenize_batch(texts: list[str], tokenizer) -> torch.Tensor:
    """Tokenize DISTINCT prompts into one (B, T) batch.

    Decoder-only batched generation uses LEFT padding so real content ends at the
    last column: logits[:, -1, :] is then the true final token for every row and
    all rows decode in lockstep at a shared start_pos. (The engine carries a single
    global position offset and no padding mask, so left padding is the only layout
    that keeps the sampled-from column valid across rows.)
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev_side
    return enc["input_ids"].to(DEVICE)


def build_model() -> tuple[LlamaModel, ModelConfig]:
    # Identical model construction to 03_engine.py __main__.
    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()
    if model.maybe_compile():
        print(f"  torch.compile: enabled (mode={llama_mod.COMPILE_MODE})")
    return model, cfg


def build_runs(prompts: dict, flavors: list[str], batch_override, tokenizer) -> list[tuple]:
    """Expand flavors into concrete (run_name, prompt_ids, n_new_tokens) runs.

    A flavor carrying a 'B' list (with a 'texts' pool of distinct prompts) expands
    into one run per batch size, each filling the batch with the first B DISTINCT
    prompts (left-padded). Without B it stays a single batch-1 run from 'text'.
    `batch_override` (CLI --batches) replaces the flavor's own B list when given.
    """
    runs = []
    for name in flavors:
        spec  = prompts[name]
        n_dec = int(spec.get("max_new_tokens", 64))
        batches = batch_override if batch_override is not None else spec.get("B")
        if batches:
            texts = spec.get("texts") or [spec["text"]]
            for B in batches:
                if B > len(texts):
                    raise SystemExit(
                        f"flavor '{name}' batch {B} > {len(texts)} distinct prompts "
                        f"in 'texts' — add more prompts or lower the batch size.")
                ids = tokenize_batch(texts[:B], tokenizer)
                runs.append((f"{name}_b{B}", ids, n_dec))
        else:
            ids = tokenize_one(spec["text"], tokenizer)
            runs.append((name, ids, n_dec))
    return runs


def make_kv_cache(cfg: ModelConfig, runs: list[tuple]) -> tuple[KVCache, int]:
    # Size the cache for the worst case across every planned run (widest batch and
    # longest prompt+decode), exactly like 03 sizes from its workloads.
    max_batch   = max(ids.shape[0] for _, ids, _ in runs)
    max_seq_len = max(ids.shape[1] + n_dec for _, ids, n_dec in runs)
    kv_cache = KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = max_batch,
        max_seq_len = max_seq_len,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )
    return kv_cache, max_seq_len


# ── profiling driver ─────────────────────────────────────────────────────────

def _render_report(name: str, backend: str, batch: int, prompt_tokens: int,
                   n_new_tokens: int, prefill_ms: float, decode_ms: float,
                   peak_vram_gb: float, table: str) -> str:
    """Assemble a clean, self-describing report: banner, summary block, op table.
    Same banner style as the kernel reports in profile-kernels/torch-profiler/out."""
    bar = "=" * 78
    rule = "-" * 78
    # decode throughput across the whole batch — the number that climbs as more
    # rows pack into one step and the GPU goes from idle to saturated.
    decode_tps = (batch * 1000.0 / decode_ms) if decode_ms > 0 else 0.0
    return (
        f"{bar}\n"
        f"  ENGINE PROFILE (iteration-03 parity)  —  flavor='{name}'\n"
        f"{bar}\n"
        f"  backend       : {backend}\n"
        f"  batch size    : {batch}\n"
        f"  prompt tokens : {prompt_tokens} (per row, left-padded)\n"
        f"  new tokens    : {n_new_tokens}\n"
        f"  prefill       : {prefill_ms:.2f} ms\n"
        f"  decode/step   : {decode_ms:.3f} ms  ({decode_tps:.1f} tok/s over the batch)\n"
        f"  peak VRAM     : {peak_vram_gb:.2f} GB\n"
        f"  sampling      : temperature={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}\n"
        f"{rule}\n"
        f"  Op table sorted by cuda_time_total (top 100)\n"
        f"{rule}\n"
        f"{table}\n"
    )


def profile_flavor(name: str, prompt_ids: torch.Tensor, n_new_tokens: int,
                   model: LlamaModel, kv_cache: KVCache, backend: str,
                   tensorboard: bool = True):
    # Warm up this prompt shape OUTSIDE the profiler (matches 03's WARMUP=2):
    # absorbs the one-time Triton JIT + autotune sweep so the captured run is
    # steady-state.
    #
    # CUDA-graph note: when USE_CUDA_GRAPHS is on, the decode graph must be
    # captured HERE, in warmup, not inside the profiled region — capture is
    # heavy one-time work (buffer alloc + internal warmup + record) that would
    # otherwise pollute the op table. CUDAGraphDecoder already warms up
    # internally, so we don't add our own decode warmup loop; we just:
    #   1. reset any graph from a previous flavor (batch size differs per flavor)
    #   2. capture explicitly via warmup_decode_graph (no-op unless graphs on)
    #   3. freeze, so an accidental capture inside `profile(...)` raises instead
    #      of silently landing in the trace.
    B = prompt_ids.shape[0]
    model.reset_graph()
    for _ in range(WARMUP):
        prefill_and_decode(model, kv_cache, prompt_ids, min(n_new_tokens, 4))
    model.warmup_decode_graph(kv_cache, B, prompt_ids.shape[1])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    model.freeze_graph(True)
    try:
        with profile(activities=ACTIVITIES, record_shapes=True,
                     profile_memory=True, with_flops=True) as prof:
            _, prefill_ms, decode_ms = prefill_and_decode(
                model, kv_cache, prompt_ids, n_new_tokens)
    finally:
        model.freeze_graph(False)

    peak_vram_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=100)

    report = _render_report(name, backend, prompt_ids.shape[0], prompt_ids.shape[1],
                            n_new_tokens, prefill_ms, decode_ms, peak_vram_gb, table)

    # One tidy .txt report + one Chrome/Perfetto trace per flavor, both in out/.
    report_path = OUT_DIR / f"profiler_engine_{name}.txt"
    report_path.write_text(report)
    trace_path = OUT_DIR / f"engine_{name}_trace.json"

    # Kineto only lets the trace be serialized ONCE, so we export a single time and
    # reuse the bytes for both sinks. When TensorBoard output is requested we route
    # the one export through tensorboard_trace_handler (correct *.pt.trace.json name
    # the torch_tb_profiler plugin expects) and then copy it to out/ as the Chrome
    # trace; otherwise we export the Chrome trace directly.
    if tensorboard:
        tb_logdir = TB_DIR / name
        tb_logdir.mkdir(parents=True, exist_ok=True)
        tensorboard_trace_handler(str(tb_logdir))(prof)
        tb_trace = max(tb_logdir.glob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
        shutil.copyfile(tb_trace, trace_path)
    else:
        prof.export_chrome_trace(str(trace_path))

    print(f"\n{report}")
    print(f"  report : {report_path}")
    print(f"  trace  : {trace_path}  (Perfetto: serve_trace.py or drag into ui.perfetto.dev)")
    if tensorboard:
        print(f"  tboard : {tb_logdir}  (tensorboard --logdir {TB_DIR})")


def main():
    p = argparse.ArgumentParser(
        description="torch.profiler over the iteration-03 inference engine")
    p.add_argument("--flavors", nargs="*", default=FLAVOR_ORDER,
                   help=f"prompt flavors to profile. choices: {', '.join(FLAVOR_ORDER)}")
    p.add_argument("--batches", nargs="*", type=int, default=None,
                   help="batch sizes to run (overrides the flavor's 'B' in prompt.json). "
                        "Each batch is filled with that many DISTINCT prompts from 'texts'.")
    p.add_argument("--no-tensorboard", dest="tensorboard", action="store_false",
                   help="skip writing the TensorBoard logdir (on by default)")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required to profile the inference engine"

    backend = "triton (fused kernels)" if rmsnorm_mod.USE_TRITON else "pytorch (reference)"
    if llama_mod.USE_CUDA_GRAPHS:
        backend += " + cuda-graph decode"
    print(f"  Backend : {backend}    [override with USE_TRITON / USE_CUDA_GRAPHS in .env]")
    print(f"  Sampling: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")

    prompts = load_prompts()
    for name in args.flavors:
        if name not in prompts:
            raise SystemExit(f"unknown flavor '{name}'. choices: {', '.join(prompts)}")

    print("  Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model, cfg = build_model()
    runs = build_runs(prompts, args.flavors, args.batches, tokenizer)
    kv_cache, max_seq_len = make_kv_cache(cfg, runs)
    print(f"  KV cache: {kv_cache}  (max_seq_len={max_seq_len})")

    for run_name, prompt_ids, n_dec in runs:
        B, T = prompt_ids.shape
        print(f"\n########## run: {run_name}  "
              f"(batch {B}, prompt {T} toks, decode {n_dec}) ##########")
        profile_flavor(run_name, prompt_ids, n_dec, model, kv_cache, backend,
                       tensorboard=args.tensorboard)


if __name__ == "__main__":
    main()
