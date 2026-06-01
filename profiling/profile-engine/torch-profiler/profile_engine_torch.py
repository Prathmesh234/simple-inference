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
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from config import ModelConfig  # noqa: E402
from loader import WeightLoader  # noqa: E402
from model.llama import LlamaModel  # noqa: E402
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
        logits   = model(next_tok, start_pos=pos, kv_cache=kv_cache)
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


def build_engine(prompts: dict, tokenizer) -> tuple[LlamaModel, KVCache, dict, int]:
    # Identical model construction to 03_engine.py __main__.
    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)

    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    # Tokenize every flavor once so we can size the KV cache for the worst case
    # (longest prompt + its decode budget), exactly like 03 sizes from workloads.
    tokenized = {}
    max_prompt_seen   = 0
    max_decode_per_wl = 0
    for name, spec in prompts.items():
        ids   = tokenize_one(spec["text"], tokenizer)
        n_dec = int(spec.get("max_new_tokens", 64))
        tokenized[name] = (ids, n_dec)
        max_prompt_seen   = max(max_prompt_seen, ids.shape[1])
        max_decode_per_wl = max(max_decode_per_wl, n_dec)

    max_seq_len = max_prompt_seen + max_decode_per_wl
    kv_cache = KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = 1,
        max_seq_len = max_seq_len,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )
    return model, kv_cache, tokenized, max_seq_len


# ── profiling driver ─────────────────────────────────────────────────────────

def _render_report(name: str, backend: str, prompt_tokens: int, n_new_tokens: int,
                   prefill_ms: float, decode_ms: float, peak_vram_gb: float,
                   table: str) -> str:
    """Assemble a clean, self-describing report: banner, summary block, op table.
    Same banner style as the kernel reports in profile-kernels/torch-profiler/out."""
    bar = "=" * 78
    rule = "-" * 78
    return (
        f"{bar}\n"
        f"  ENGINE PROFILE (iteration-03 parity)  —  flavor='{name}'\n"
        f"{bar}\n"
        f"  backend       : {backend}\n"
        f"  prompt tokens : {prompt_tokens}\n"
        f"  new tokens    : {n_new_tokens}\n"
        f"  prefill       : {prefill_ms:.2f} ms\n"
        f"  decode/step   : {decode_ms:.3f} ms\n"
        f"  peak VRAM     : {peak_vram_gb:.2f} GB\n"
        f"  sampling      : temperature={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}\n"
        f"{rule}\n"
        f"  Op table sorted by cuda_time_total (top 25)\n"
        f"{rule}\n"
        f"{table}\n"
    )


def profile_flavor(name: str, prompt_ids: torch.Tensor, n_new_tokens: int,
                   model: LlamaModel, kv_cache: KVCache, backend: str):
    # Warm up this prompt shape OUTSIDE the profiler (matches 03's WARMUP=2):
    # absorbs the one-time Triton JIT + autotune sweep so the captured run is
    # steady-state.
    for _ in range(WARMUP):
        prefill_and_decode(model, kv_cache, prompt_ids, min(n_new_tokens, 4))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    with profile(activities=ACTIVITIES, record_shapes=True,
                 profile_memory=True, with_flops=True) as prof:
        _, prefill_ms, decode_ms = prefill_and_decode(
            model, kv_cache, prompt_ids, n_new_tokens)

    peak_vram_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)

    report = _render_report(name, backend, prompt_ids.shape[1], n_new_tokens,
                            prefill_ms, decode_ms, peak_vram_gb, table)

    # One tidy .txt report + one Chrome/Perfetto trace per flavor, both in out/.
    report_path = OUT_DIR / f"profiler_engine_{name}.txt"
    report_path.write_text(report)
    trace_path = OUT_DIR / f"engine_{name}_trace.json"
    prof.export_chrome_trace(str(trace_path))

    print(f"\n{report}")
    print(f"  report : {report_path}")
    print(f"  trace  : {trace_path}  (open at chrome://tracing or ui.perfetto.dev)")


def main():
    p = argparse.ArgumentParser(
        description="torch.profiler over the iteration-03 inference engine")
    p.add_argument("--flavors", nargs="*", default=FLAVOR_ORDER,
                   help=f"prompt flavors to profile. choices: {', '.join(FLAVOR_ORDER)}")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required to profile the inference engine"

    backend = "triton (fused kernels)" if rmsnorm_mod.USE_TRITON else "pytorch (reference)"
    print(f"  Backend : {backend}    [override with USE_TRITON=true/false in .env]")
    print(f"  Sampling: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")

    prompts = load_prompts()
    for name in args.flavors:
        if name not in prompts:
            raise SystemExit(f"unknown flavor '{name}'. choices: {', '.join(prompts)}")

    print("  Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model, kv_cache, tokenized, max_seq_len = build_engine(prompts, tokenizer)
    print(f"  KV cache: {kv_cache}  (max_seq_len={max_seq_len})")

    for name in args.flavors:
        prompt_ids, n_dec = tokenized[name]
        print(f"\n########## flavor: {name}  "
              f"(prompt {prompt_ids.shape[1]} toks, decode {n_dec}) ##########")
        profile_flavor(name, prompt_ids, n_dec, model, kv_cache, backend)


if __name__ == "__main__":
    main()
