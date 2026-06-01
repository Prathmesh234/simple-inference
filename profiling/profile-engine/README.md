# Engine Profiling — full inference path

Where `profile-kernels/` isolates a single Triton kernel, this folder profiles
the **whole generation engine** end to end: token embedding, all 28 transformer
blocks (attention + RMSNorm + RoPE + SwiGLU MLP), the LM head, and sampling.

## Engine parity

`torch-profiler/profile_engine_torch.py` runs a `prefill_and_decode` function
that is a **byte-for-byte copy** of the engine in `iterations/03_engine.py` (our
first engine iteration) — same constants, model/tokenizer/KV-cache setup,
sampling (`temperature=0.7, top_k=50, top_p=0.9`), and the "prefill once, then
decode exactly N tokens (no EOS early-stop)" loop. The profiler wraps the
unchanged engine; it does not re-interpret it.

## Two regimes

- **prefill** — one forward over the whole prompt → Time-To-First-Token
- **decode** — the per-token autoregressive steps → Time-Per-Output-Token

Four prompt **flavors** from `prompt.json` (`short`, `medium_short`,
`medium_long`, `long`) show prefill cost growing with prompt length while
per-token decode stays roughly flat.

## Folder layout

```
profile-engine/
  prompt.json             4 prompt flavors (shared input, at root)
  torch-profiler/         profile_engine_torch.py, out/
  nsys-profiler/          (reserved)
  tensorboard-profiler/   (reserved)
```

## How to run

Requires CUDA **and** the gated Llama-3.2-3B weights, so `HF_TOKEN` must be set
(in `.env`, see `.env.example`).

```bash
# all four flavors
PATH="$HOME/.local/bin:$PATH" \
XDG_CONFIG_HOME="$HOME/.cache/xdgconfig" \
UV_CACHE_DIR="$HOME/.cache/uv" \
uv run python profiling/profile-engine/torch-profiler/profile_engine_torch.py

# one (or several) flavor(s)
uv run python profiling/profile-engine/torch-profiler/profile_engine_torch.py --flavors short long
```

The backend (Triton fused kernels vs PyTorch reference) follows `USE_TRITON` in
`.env`.

## Output convention (clean format)

All engine output goes to `torch-profiler/out/` — one tidy report per flavor and
one timeline trace. Reports use the same `===` banner + summary + table format
as the kernel reports:

| File | Contents |
|------|----------|
| `out/profiler_engine_<flavor>.txt` | banner + summary block (prompt/new tokens, prefill ms, decode ms/step, peak VRAM, sampling) + `key_averages` op table sorted by `cuda_time_total` |
| `out/engine_<flavor>_trace.json` | Chrome/Perfetto timeline (open at `chrome://tracing` or `ui.perfetto.dev`) |

`nsys-profiler/out/` and `tensorboard-profiler/` are reserved for the same
out/-folder convention when those drivers are added.
