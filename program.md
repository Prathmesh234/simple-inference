# simple-inference

This is an experiment to have an LLM autonomously optimize an inference engine.

The starting point is `iterations/03_engine.py` — a working pure-PyTorch
engine for `meta-llama/Llama-3.2-11B-Vision` with a prefill + decode loop,
KV cache, and temperature/top-k/top-p sampling. **That file is the base
engine you start from. Always optimize. Never stand still — every loop
iteration must attempt a new optimization, measure it, and either keep or
revert.**

You decide which optimizations to try — KV-cache layout, fused Triton
kernels, quantization, CUDA graphs, speculative decoding, paged attention,
weight repacking, batching tricks, whatever you can justify from profile
data. No optimization is pre-prescribed. Build the engine up from
`iterations/03_engine.py` and make it as fast as it will go.

## Target model

**`meta-llama/Llama-3.2-11B-Vision`** (multimodal Llama 3.2). For
token-throughput optimization, only the text decoder matters; the vision
encoder is exercised only when an image is in the prompt. HuggingFace
spec:

### Text decoder (`MllamaTextConfig`)

| Property                | Value                                     |
|-------------------------|-------------------------------------------|
| `hidden_size`           | 4096                                      |
| `intermediate_size`     | 14336                                     |
| `num_hidden_layers`     | 40                                        |
| `num_attention_heads`   | 32                                        |
| `num_key_value_heads`   | 8  (GQA, 4:1 ratio)                       |
| `head_dim`              | 128                                       |
| `vocab_size`            | 128256                                    |
| `max_position_embeddings` | 131072                                  |
| `rope_theta`            | 500000.0                                  |
| `rms_norm_eps`          | 1e-5                                      |
| `hidden_act`            | silu (SwiGLU MLP)                         |
| `tie_word_embeddings`   | False                                     |
| `cross_attention_layers`| [3, 8, 13, 18, 23, 28, 33, 38]            |
| Norm                    | RMSNorm                                   |
| Position encoding       | RoPE                                      |

The 8 cross-attention layers attend to vision tokens when present. For
text-only prompts they pass through residually — they still cost
parameters and a forward pass, so don't skip them in the engine.

### Vision encoder (`MllamaVisionConfig`)

| Property                | Value          |
|-------------------------|----------------|
| `hidden_size`           | 1280           |
| `num_hidden_layers`     | 32             |
| `num_global_layers`     | 8              |
| `intermediate_size`     | 5120           |
| `num_attention_heads`   | 16             |
| `num_channels`          | 3              |
| `patch_size`            | 14             |
| `image_size`            | 560            |
| `vision_output_dim`     | 7680           |
| `max_num_tiles`         | 4              |

### Totals

- ~10.7 B parameters (≈ 8 B text decoder + ≈ 2.7 B vision/cross-attn + adapters)
- bfloat16 weights ≈ 21 GB on disk
- KV cache (text decoder, 1 sample, 8192 ctx) ≈ 5.4 GB in bf16

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `inference/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b inference/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these for full context:
   - `iterations/03_engine.py` — **the base engine you start from.** Pure PyTorch prefill + decode + KV cache + sampling. Every optimization is a delta against this.
   - `config.py` — `ModelConfig`. Initialize it for `Llama-3.2-11B-Vision` using the spec above, then frozen.
   - `loader.py` — pulls weights from HuggingFace. Frozen.
   - `tokenizer.py` — wraps the HF fast tokenizer. Frozen.
   - `env_loader.py` — loads `HF_TOKEN`. Frozen.
   - `generate.py` — top-level streaming generate. **Editable.**
   - `sampling.py` — temperature/top-k/top-p sampler. **Editable.**
   - `utilities.py` — metric helpers. **Editable.**
   - `model/` — `LlamaModel`, `TransformerBlock`, `KVCache`. **Editable.**
   - `ops/` — RMSNorm, attention (GQA), MLP (SwiGLU), RoPE, embedding. **Editable.**
   - `kernels/` — existing Triton kernels (attention, rmsnorm, rope, swiglu). **Editable** (replace, add, or remove).
   - `benchmarks/run_baseline.py`, `benchmarks/bench_utils.py` — the measurement harness. **Frozen.** This is your ground truth.
   - `profiling/` — torch / nsys / tensorboard profilers + roofline. Use these to find hotspots.
4. **Verify the model loads**: `HF_TOKEN` must be set (see `.env.example`). A first generate run should succeed end-to-end. Llama-3.2-11B-Vision is gated — make sure you have access.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline gets recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU against `meta-llama/Llama-3.2-11B-Vision`
in bfloat16. You launch a measurement with:

```
uv run benchmarks/run_baseline.py > run.log 2>&1
```

This writes its numbers to `benchmarks/results_baseline.json` and prints a
formatted summary. That is the ground-truth measurement.

**What you CAN do:**
- Modify anything under `model/`, `ops/`, `kernels/`, plus `generate.py`,
  `sampling.py`, `utilities.py`. Add new files freely.
- Rewrite kernels. Write new Triton kernels. Delete kernels that aren't
  worth their complexity.
- Change layouts, fuse ops, add CUDA graphs, change KV-cache strategy,
  quantize weights, try speculative decoding — anything that improves the
  metric without breaking correctness.

**What you CANNOT do:**
- Modify `config.py` (after initial 11B setup), `loader.py`, `tokenizer.py`,
  or `env_loader.py`. The model identity is fixed.
- Modify `benchmarks/run_baseline.py` or `benchmarks/bench_utils.py`. The
  harness is the ground-truth metric.
- Install new packages or add dependencies. You only get what's already in
  `pyproject.toml`.
- Change the target model. It is always Llama-3.2-11B-Vision.

**The goal: maximize decode throughput (`decode_tok_s`) on
Llama-3.2-11B-Vision** under the standard prompt in `generate.py`'s
`__main__`. Prefill latency (`ttft_ms`) is the secondary metric — don't
regress it badly chasing decode wins. Correctness is a hard gate: greedy
decode (`temperature=0`) of the standard prompt must match the baseline's
first 64 tokens exactly. Sampled outputs just need to look coherent.

**VRAM** is a soft constraint. Some increase is fine for real wins; OOM
on the target GPU is a crash.

**Simplicity criterion**: All else being equal, simpler is better. A 2%
decode_tok_s win that adds 200 lines of fragile Triton is probably not
worth it. Deleting a kernel and getting equal speed is a clear win. Weigh
complexity against magnitude. Trivial wins from deleting code are the
best wins.

**The first run**: Your very first run establishes the baseline. Run
`benchmarks/run_baseline.py` as-is against the unmodified base engine
from `iterations/03_engine.py`. Record those numbers as the row to beat.

## Output format

`run_baseline.py` prints a summary like:

```
---
prefill_tok_s_T512:   12480.3
decode_tok_s_B1:      118.7
decode_tok_s_B8:      642.1
peak_vram_gb:         24.1
ttft_ms_T512:         62.0
backend:              triton
```

Extract the key metric from the log file with:

```
grep "^decode_tok_s_B1:" run.log
```

## Version management with git

Git is the system of record. Every experiment is exactly one commit on
the run branch. Rules:

- One experiment = one commit. Never bundle two ideas in one commit.
- Commit message format:
  `<verb> <component>: <one-line idea> [decode_tok_s X.X → Y.Y]`
  e.g. `fuse rmsnorm into qkv proj [decode_tok_s 118.7 → 131.4]`.
- Commit BEFORE the run. If it crashes or regresses, `git reset --hard HEAD~1`
  to discard. Never amend a measured commit.
- Branch lineage = optimization history. Anyone should be able to
  `git log --oneline` and see the full story of wins.
- `results.tsv` is the parallel ledger. Untracked. Same row per commit.
- Never force-push, never rewrite shared history, never push to `main`.
  Only push the run branch (`inference/<tag>`).
- If you rewind (rarely): use `git reset --hard <hash>` to a known-good
  commit on this branch and continue forward. Log the rewind in
  `results.tsv` as a `rewind` status row so the audit trail is intact.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	decode_tok_s	ttft_ms	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `decode_tok_s_B1` achieved (e.g. 118.7) — use 0.0 for crashes
3. `ttft_ms_T512` (e.g. 62.0) — use 0.0 for crashes
4. peak VRAM in GB, round to .1f — use 0.0 for crashes
5. status: `keep`, `discard`, `crash`, or `rewind`
6. short text description of what this experiment tried

Example:

```
commit	decode_tok_s	ttft_ms	vram_gb	status	description
a1b2c3d	118.7	62.0	24.1	keep	baseline (03_engine.py, pytorch + existing triton kernels)
b2c3d4e	131.4	61.8	24.2	keep	fuse rmsnorm into attention qkv proj
c3d4e5f	119.1	60.2	24.1	discard	swap sampler to torch.multinomial (no decode win)
d4e5f6g	0.0	0.0	0.0	crash	int8 weight-only quant — accuracy gate failed
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `inference/mar5`).

LOOP FOREVER — always be optimizing:

1. Look at git state: the current branch / commit.
2. Pick an optimization idea and implement it by directly hacking the code
   under the editable paths. **Do not pause to plan an N-step roadmap. One
   small idea, one commit, one measurement.**
3. `git commit` with the format described above.
4. Run the measurement: `uv run benchmarks/run_baseline.py > run.log 2>&1`
   (redirect everything — do NOT use `tee` or flood your context).
5. Read the results: `grep "^decode_tok_s_B1:\|^ttft_ms_T512:\|^peak_vram_gb:" run.log`
6. Correctness gate: run greedy decode against the standard prompt at
   `temperature=0` and compare first 64 tokens to baseline. If they
   diverge, treat the change as a crash.
7. If the grep output is empty, the run crashed. `tail -n 50 run.log` for
   the stack trace and try to fix. After a few failed attempts, give up.
8. Record the row in `results.tsv` (leave it untracked by git).
9. If `decode_tok_s_B1` improved and correctness passed, advance — keep
   the commit. Update the commit message tail with the measured delta if
   you didn't already.
10. If it's equal or worse, or correctness failed,
    `git reset --hard HEAD~1` back to where you started.
11. Immediately pick the next idea. Go to step 2.

**Timeout**: A single measurement should take a couple of minutes. If a
run exceeds 10 minutes, kill it and treat it as a failure (discard and
revert).

**Crashes**: Dumb fix (typo, missing import) — fix and re-run. Idea is
fundamentally broken — log "crash" in the TSV and move on.

**Profiling**: Before swinging big optimizations, profile. Use
`profiling/profile-engine/torch-profiler/profile_engine_torch.py` for
end-to-end traces and `profiling/profile-kernels/` for kernel-level
breakdowns and roofline. Decide what to optimize based on what's actually
hot, not what feels exciting.

**NEVER STOP**: Once the experiment loop has begun (after the initial
setup), do NOT pause to ask the human if you should continue. Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human
might be asleep or away and expects you to continue working *indefinitely*
until you are manually stopped. You are autonomous. **Always be
optimizing.** If you run out of ideas: re-profile for new hotspots, re-read
the in-scope files for new angles, read the papers behind techniques you
haven't tried (FlashAttention variants, PagedAttention, Medusa, EAGLE,
INT4 / INT8 / FP8 quant, speculative decoding, CUDA graphs, tensor
parallel slicing, prefix caching), combine previous near-misses, or try a
more radical architectural rewrite. The loop runs until the human
interrupts you, period.

A user might leave you running while they sleep. If each experiment takes
~5 minutes, that's ~12/hour, ~100 over a night. They wake up to a stack
of measured wins — and the git log is the story of how you got there.
