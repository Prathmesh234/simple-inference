# simple-inference

This is an experiment to have an LLM autonomously optimize an inference engine.

The starting point is a working pure-PyTorch + Triton inference engine for
`meta-llama/Llama-3.2-3B`. Your job is to make it faster, end of story.
You decide which optimizations to try — KV-cache layout, fused kernels,
quantization, CUDA graphs, speculative decoding, batching tricks, whatever
you can justify. No optimization is pre-prescribed.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `inference/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b inference/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these for full context:
   - `config.py` — `ModelConfig` for Llama-3.2-3B. Frozen.
   - `loader.py` — pulls weights from HuggingFace. Frozen.
   - `tokenizer.py` — wraps the HF fast tokenizer. Frozen.
   - `generate.py` — top-level prefill + decode loop. **Editable.**
   - `sampling.py` — temperature/top-k/top-p sampler. **Editable.**
   - `utilities.py` — metric helpers. **Editable.**
   - `model/` — `LlamaModel`, `TransformerBlock`, `KVCache`. **Editable.**
   - `ops/` — RMSNorm, attention (GQA), MLP (SwiGLU), RoPE, embedding. **Editable.**
   - `kernels/` — existing Triton kernels (attention, rmsnorm, rope, swiglu). **Editable** (replace, add, or remove).
   - `benchmarks/run_baseline.py`, `benchmarks/bench_utils.py` — the measurement harness. **Frozen.** This is your ground truth.
   - `profiling/` — torch / nsys / tensorboard profilers + roofline. Use these to find hotspots.
4. **Verify the model loads**: `HF_TOKEN` must be set (see `.env.example`). A first generate run should succeed end-to-end.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline gets recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU against `meta-llama/Llama-3.2-3B` in
bfloat16. You launch a measurement with:

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
- Modify `config.py`, `loader.py`, `tokenizer.py`, `env_loader.py`. The
  model identity is fixed.
- Modify `benchmarks/run_baseline.py` or `benchmarks/bench_utils.py`. The
  harness is the ground-truth metric.
- Install new packages or add dependencies. You only get what's already in
  `pyproject.toml`.
- Change the target model. It's always Llama-3.2-3B.

**The goal: maximize decode throughput (`decode_tok_s`) on Llama-3.2-3B**
under the standard prompt in `generate.py`'s `__main__`. Prefill latency
(`ttft_ms`) is the secondary metric — don't regress it badly chasing
decode wins. Correctness is a hard gate: greedy decode (`temperature=0`)
of the standard prompt must match the baseline's first 64 tokens exactly.
Sampled outputs just need to look coherent.

**VRAM** is a soft constraint. The target hardware has 48 GB. Some
increase is fine for real wins; blowing past 48 GB is a crash.

**Simplicity criterion**: All else being equal, simpler is better. A 2%
decode_tok_s win that adds 200 lines of fragile Triton is probably not
worth it. Deleting a kernel and getting equal speed is a clear win.
Weigh complexity against magnitude. Trivial wins from deleting code are
the best wins.

**The first run**: Your very first run should always be to establish the
baseline, so you run `benchmarks/run_baseline.py` as-is. Record those
numbers as the row to beat.

## Output format

`run_baseline.py` prints a summary like:

```
---
prefill_tok_s_T512:   12480.3
decode_tok_s_B1:      118.7
decode_tok_s_B8:      642.1
peak_vram_gb:         8.4
ttft_ms_T512:         41.0
backend:              triton
```

You can extract the key metric from the log file:

```
grep "^decode_tok_s_B1:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	decode_tok_s	ttft_ms	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `decode_tok_s_B1` achieved (e.g. 118.7) — use 0.0 for crashes
3. `ttft_ms_T512` (e.g. 41.0) — use 0.0 for crashes
4. peak VRAM in GB, round to .1f — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	decode_tok_s	ttft_ms	vram_gb	status	description
a1b2c3d	118.7	41.0	8.4	keep	baseline (pytorch + existing triton kernels)
b2c3d4e	131.4	40.8	8.4	keep	fuse rmsnorm into attention qkv proj
c3d4e5f	119.1	39.2	8.4	discard	swap sampler to torch.multinomial (no decode win)
d4e5f6g	0.0	0.0	0.0	crash	int8 weight-only quant — accuracy gate failed
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `inference/mar5` or
`inference/mar5-gpu0`).

LOOP FOREVER:

1. Look at git state: the current branch / commit.
2. Pick an optimization idea and implement it by directly hacking the code
   under the editable paths.
3. `git commit` the change.
4. Run the measurement: `uv run benchmarks/run_baseline.py > run.log 2>&1`
   (redirect everything — do NOT use `tee` or flood your context).
5. Read the results: `grep "^decode_tok_s_B1:\|^ttft_ms_T512:\|^peak_vram_gb:" run.log`
6. Correctness gate: `uv run python -c "from generate import generate_with_stats; ..."`
   with `temperature=0` against the standard prompt, compare first 64 tokens
   to baseline. If they diverge, treat the change as a crash.
7. If the grep output is empty, the run crashed. `tail -n 50 run.log` for
   the stack trace and try to fix. After a few failed attempts, give up.
8. Record the row in `results.tsv` (leave it untracked by git).
9. If `decode_tok_s_B1` improved and correctness passed, "advance" the
   branch — keep the commit.
10. If it's equal or worse, or correctness failed, `git reset --hard` back
    to where you started.

If you feel stuck, you can rewind — but do this very sparingly. The point
is to keep advancing the branch and stacking real wins.

**Timeout**: A single measurement should take a couple of minutes. If a
run exceeds 10 minutes, kill it and treat it as a failure (discard and
revert).

**Crashes**: If a run crashes (OOM, bug, kernel error), use your
judgment. Dumb fix (typo, missing import) — fix and re-run. Idea is
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
until you are manually stopped. You are autonomous. If you run out of
ideas, think harder — re-read the in-scope files for new angles, profile
again for new hotspots, read the papers behind techniques you haven't
tried (FlashAttention variants, PagedAttention, Medusa, EAGLE, INT4,
FP8, etc.), try combining previous near-misses, try more radical changes.
The loop runs until the human interrupts you, period.

A user might leave you running while they sleep. If each experiment takes
~5 minutes, that's ~12/hour, ~100 over a night. They wake up to a stack
of measured wins.
