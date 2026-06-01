# simple-inference

This is an experiment to have an LLM autonomously optimize an inference engine.

The starting point is `iterations/03_engine.py` — a working pure-PyTorch
engine for `meta-llama/Llama-3.1-8B` with a prefill + decode loop, KV
cache, and temperature/top-k/top-p sampling. **That file is the base
engine you start from. Always optimize. Never stand still — every loop
iteration must attempt a new optimization, measure it, and either keep or
revert.**

You decide which optimizations to try. No optimization is pre-prescribed.
Build the engine up from `iterations/03_engine.py` and make it as fast as
it will go.

Be creative. The well-known tricks (fused kernels, quantization, CUDA
graphs, speculative decoding, paged attention, radix prefix sharing) are
fair game — but they are only the floor. Read the "How to think about
this" section below before you start swinging. The expectation is that
you also invent: borrow abstractions from operating systems, databases,
networking, compilers, and hardware architecture, name the analogy
explicitly, and verify with measurement.

## Target model

**`meta-llama/Llama-3.1-8B`** (text-only Llama 3.1 base). HuggingFace
spec:

| Property                  | Value                                              |
|---------------------------|----------------------------------------------------|
| `hidden_size`             | 4096                                               |
| `intermediate_size`       | 14336                                              |
| `num_hidden_layers`       | 32                                                 |
| `num_attention_heads`     | 32                                                 |
| `num_key_value_heads`     | 8 (GQA, 4:1 ratio)                                 |
| `head_dim`                | 128                                                |
| `vocab_size`              | 128256                                             |
| `max_position_embeddings` | 131072                                             |
| `rope_theta`              | 500000.0                                           |
| `rope_scaling`            | `{type: llama3, factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_position_embeddings: 8192}` |
| `rms_norm_eps`            | 1e-5                                               |
| `hidden_act`              | silu (SwiGLU MLP)                                  |
| `tie_word_embeddings`     | False                                              |
| Norm                      | RMSNorm                                            |
| Position encoding         | RoPE (Llama 3.1 scaled rope)                       |

### Totals

- 8.03 B parameters
- bfloat16 weights ≈ 16 GB on disk
- KV cache (1 sample, 8192 ctx) ≈ 4.3 GB in bf16
  (`2 × 32 layers × 8 kv-heads × 128 head_dim × 8192 tokens × 2 bytes`)

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `inference/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b inference/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these for full context:
   - `iterations/03_engine.py` — **the base engine you start from.** Pure PyTorch prefill + decode + KV cache + sampling. Every optimization is a delta against this.
   - `config.py` — `ModelConfig`. Initialize it for `Llama-3.1-8B` using the spec above, then frozen.
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
4. **Verify the model loads**: `HF_TOKEN` must be set (see `.env.example`). A first generate run should succeed end-to-end. Llama-3.1-8B is gated — make sure you have access.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline gets recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU against `meta-llama/Llama-3.1-8B`
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
- Modify `config.py` (after initial 8B setup), `loader.py`, `tokenizer.py`,
  or `env_loader.py`. The model identity is fixed.
- Modify `benchmarks/run_baseline.py` or `benchmarks/bench_utils.py`. The
  harness is the ground-truth metric.
- Install new packages or add dependencies. You only get what's already in
  `pyproject.toml`.
- Change the target model. It is always Llama-3.1-8B.

**The goal: maximize decode throughput (`decode_tok_s`) on Llama-3.1-8B**
under the standard prompt in `generate.py`'s
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

## How to think about this

The best ideas in systems engineering come from looking sideways —
borrowing a solved abstraction from one domain and recognizing it fits an
unsolved problem in another.

**PagedAttention** did not invent anything new. It looked at the KV cache
— a fragmented, wasteful memory problem — and recognized that operating
systems had already solved exactly this class of problem decades ago with
virtual memory and paging. The insight was not technical. It was
perceptual.

**RadixAttention** looked at the same space and saw prefix sharing — the
observation that many requests share common prefixes, and that a radix
tree already knows how to handle shared prefixes efficiently.

The pattern: *a hard inference problem is often a disguised instance of a
solved systems problem.*

Look at the components of inference — attention computation, memory
layout, scheduling, batching, eviction, communication — and ask: where
have humans solved something structurally identical before? The answers
are not always in ML research. They are in operating systems, databases,
networking, compilers, and hardware architecture. The discipline that
built the internet, the filesystem, and the CPU cache has been solving
your problems for fifty years.

Then go further. Once you have the borrowed analogies in hand, generate
your own. Stare at the dataflow of a decode step and notice what is
*wasted* — compute thrown away, bytes moved through memory twice, a tile
loaded that the next tile needs again, a stall waiting on a tensor that
was already in registers. Each waste is a hint at a structure waiting to
be named.

### Working principles

1. **Start from first principles.** Understand *why* a current approach
   is bottlenecked before proposing a fix. "It's slow" is not a diagnosis.
   "We are memory-bandwidth-bound at 870 GB/s of 1008 GB/s peak because
   the KV cache reload dominates the decode step" is a diagnosis.
2. **Name the analogy explicitly.** When you borrow from another domain,
   say so in the commit message and in `results.tsv`. "KV cache as
   page-cache with LRU eviction" is honest. "Tried a new caching idea" is
   not. Naming keeps reasoning transferable, lets you reuse the same
   abstraction next loop, and makes regressions easier to diagnose.
3. **Prefer structural solutions over numerical ones.** A better
   algorithm beats a faster kernel. A better abstraction beats a better
   algorithm. Shaving 2% off an op is local; restructuring the dataflow
   so the op runs half as often is global.
4. **Hunt the wasted work.** Compute that is discarded, memory that sits
   idle, transfers that move the same bytes twice, tiles that get loaded
   then evicted then reloaded — these are your targets. Every modern
   inference win is, at heart, the removal of waste.
5. **Iterate.** No design survives first contact with constraints. Write,
   measure, critique, revise. The loop is the methodology.

### Sources to mine for analogies (non-exhaustive)

- **Operating systems**: paging, virtual memory, copy-on-write, page
  cache, working sets, LRU/CLOCK eviction, slab allocation,
  scheduling (CFS, EDF), context switching.
- **Databases**: buffer pools, B-tree / LSM tree layouts, query plans,
  vectorized execution, materialized views, write-ahead logs, index
  prefix sharing.
- **Networking**: pipelining, head-of-line blocking, batching vs latency
  tradeoffs, congestion control, multiplexing, RDMA-style zero-copy.
- **Compilers**: register allocation, loop fusion / tiling / unrolling,
  SSA, dead-code elimination, profile-guided optimization, polyhedral
  scheduling.
- **Hardware architecture**: cache hierarchies, prefetching, branch
  prediction, speculative execution, out-of-order issue, NUMA, banked
  memory, scratchpads.
- **Distributed systems**: sharding, replication, consensus, work
  stealing, gossip, content-addressable storage.

These are starting points, not a checklist. The point is the *mode of
thought*: pattern-match across domains, then verify with first
principles. Your commit message should read like a hypothesis ("treat
the KV cache like a page cache") followed by a measurement ("decode_tok_s
118.7 → 131.4"). That is the unit of progress.

## Output format

`run_baseline.py` prints a summary like:

```
---
prefill_tok_s_T512:   12480.3
decode_tok_s_B1:      118.7
decode_tok_s_B8:      642.1
peak_vram_gb:         17.2
ttft_ms_T512:         54.0
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
a1b2c3d	118.7	54.0	17.2	keep	baseline (03_engine.py, pytorch + existing triton kernels)
b2c3d4e	131.4	53.8	17.3	keep	fuse rmsnorm into attention qkv proj
c3d4e5f	119.1	52.2	17.2	discard	swap sampler to torch.multinomial (no decode win)
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
