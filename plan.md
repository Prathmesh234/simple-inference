# Simple Inference Engine — Learning Plan

**Target model:** meta-llama/Llama-3.2-3B  
**Hardware:** RTX 6000 Ada (48 GB VRAM)  
**Goal:** Understand inference from scratch — weights on disk to generated tokens. Build every component class by class in pure PyTorch first, collect full benchmarks, then replace hot ops with Triton kernels and measure the delta.

**Discipline:** one class per section. No big files. Each class is written, verified correct against `transformers`, and benchmarked before moving on.

---

## Architecture cheat-sheet (Llama 3.2-3B)

| Property               | Value         |
|------------------------|---------------|
| Layers                 | 28            |
| Hidden dim             | 3072          |
| Intermediate dim (MLP) | 8192          |
| Attention heads (Q)    | 24            |
| Attention heads (KV)   | 8  ← GQA      |
| Head dim               | 128           |
| Vocab size             | 128,256       |
| Max seq len            | 131,072       |
| Norm                   | RMSNorm       |
| Activation             | SiLU (SwiGLU) |
| Position encoding      | RoPE          |
| Tied embeddings        | Yes           |

---

## Two-phase approach

### Phase 1 — Pure PyTorch baseline (Sections 1–13)
Build every component in plain PyTorch with no custom kernels. Goal is a working, correct inference engine. Every section ends with:
1. A numerical correctness check against `transformers`
2. A benchmark (latency + memory bandwidth) stored in `benchmarks/results_baseline.json`

### Phase 2 — Triton kernels (Section 14+)
Replace ops one at a time with Triton kernels. Re-run the same benchmarks after each replacement so you can see exactly what changed and why.

---

## ✅ Section 1 — Project Setup  `DONE`

**Files:** `requirements.txt`, `config.py`

- `ModelConfig` dataclass — every hyperparameter, loaded from HF `config.json`
- `verify_gpu()` — print device, VRAM, torch/triton versions
- `ModelConfig.kv_cache_bytes()` — estimate KV cache cost at various context lengths

**Learned:** HF config format, how to parameterize a model without magic numbers.

---

## ✅ Section 2 — Weight Loading  `DONE`

**Files:** `loader.py`

- `WeightLoader` — lazy shard loading via `safe_open`, HF name → our name mapping
- `print_manifest()` — shape, dtype, shard for every tensor without loading data
- `verify_parameter_count()` — confirmed 3.213B

**Learned:** safetensors format, HF weight naming, GQA visible in weight shapes (K/V are 3× smaller than Q).

---

## ✅ Section 3 — Tokenizer  `DONE`

**Files:** `tokenizer.py`

- `Tokenizer` — wraps `PreTrainedTokenizerFast`, exposes `encode` / `decode`
- `show_tokens()` — print each token and ID for any string
- Round-trip correctness test

**Learned:** BPE subword splitting, why vocab is 128,256, why BOS matters.

---

## Section 4 — RMSNorm  ← next

**Class:** `RMSNorm` in `ops/rmsnorm.py`

One class. One forward method. Nothing else in this file.

```python
class RMSNorm(nn.Module):
    # weight: (hidden_size,)
    # forward(x) → x / rms(x) * weight
    # rms(x) = sqrt(mean(x²) + eps)
```

Steps:
- Write `RMSNorm` in pure PyTorch
- Load `layers.0.attn_norm` weight from `WeightLoader`
- Verify output matches `transformers` `LlamaRMSNorm` within 1e-3 (bfloat16 tolerance)
- Benchmark: record latency with `triton.testing.do_bench` for shapes `[1, 128, 3072]` and `[1, 2048, 3072]`
- Store result in `benchmarks/results_baseline.json`

**Learned:** why RMSNorm instead of LayerNorm (no mean subtraction = cheaper), how a single learned scale vector works.

---

## Section 5 — Embeddings

**Classes:** `TokenEmbedding`, `OutputProjection` in `ops/embedding.py`

One file, two small classes.

```python
class TokenEmbedding(nn.Module):
    # weight: (vocab_size, hidden_size)
    # forward(token_ids) → (B, T, hidden_size)

class OutputProjection(nn.Module):
    # reuses TokenEmbedding's weight (tied embeddings)
    # forward(x) → (B, T, vocab_size)  via x @ weight.T
```

Steps:
- Load `embed_tokens` weight from `WeightLoader`
- Verify `OutputProjection` shares the exact same tensor (not a copy)
- Benchmark: embedding lookup latency at vocab_size=128256

**Learned:** tied embeddings save ~1.2 GB, why sharing weights is mathematically valid.

---

## Section 6 — RoPE

**Classes:** `RopeFrequencies`, `apply_rope` in `ops/rope.py`

```python
class RopeFrequencies:
    # precompute cos/sin tables up to max_seq_len
    # __init__: build (max_seq_len, head_dim) cos and sin tensors

def apply_rope(q, k, freqs):
    # rotate Q and K by position-dependent angles
    # input:  q/k shape (B, T, n_heads, head_dim)
    # output: same shape, rotated
```

Steps:
- Implement Llama 3's modified RoPE scaling (`rope_type="llama3"`)
- Verify Q/K output matches `transformers` within 1e-3
- Benchmark: `apply_rope` latency at decode (T=1) and prefill (T=512) sizes

**Learned:** why RoPE generalizes beyond training context length, how frequency pairs encode position, what the Llama 3 scaling modification does.

---

## Section 7 — Attention (GQA)

**Class:** `GroupedQueryAttention` in `ops/attention.py`

```python
class GroupedQueryAttention(nn.Module):
    # wq: (n_heads_q * head_dim, hidden)
    # wk: (n_heads_kv * head_dim, hidden)
    # wv: (n_heads_kv * head_dim, hidden)
    # wo: (hidden, n_heads_q * head_dim)
    #
    # forward(x, freqs, mask) → (B, T, hidden)
    # internally: project → reshape → repeat KV → rope → sdp → project out
```

Steps:
- Implement GQA: repeat K/V heads to match Q head count before attention
- Use `torch.nn.functional.scaled_dot_product_attention` (fused SDPA, not naive)
- Verify layer-0 output against `transformers` within 1e-2
- Benchmark: prefill (T=512) and decode (T=1) latency, note the difference

**Learned:** how GQA reduces KV memory 3×, what "repeat interleave" does to head tensors, why prefill and decode have very different latency profiles.

---

## Section 8 — MLP (SwiGLU)

**Class:** `SwiGLUMLP` in `ops/mlp.py`

```python
class SwiGLUMLP(nn.Module):
    # w_gate: (intermediate, hidden)
    # w_up:   (intermediate, hidden)
    # w_down: (hidden, intermediate)
    #
    # forward(x):
    #   gate = silu(x @ w_gate.T)
    #   up   = x @ w_up.T
    #   return (gate * up) @ w_down.T
```

Steps:
- Implement the three-matrix SwiGLU forward
- Verify layer-0 MLP output against `transformers`
- Benchmark: note MLP is typically the most compute-heavy op per layer

**Learned:** why gated activations (SwiGLU) outperform plain GELU, why there are 3 matrices not 2, how intermediate_size=8192 creates an expand-then-contract pattern.

---

## Section 9 — Transformer Block

**Class:** `TransformerBlock` in `model/block.py`

```python
class TransformerBlock(nn.Module):
    # attn_norm: RMSNorm
    # attn:      GroupedQueryAttention
    # mlp_norm:  RMSNorm
    # mlp:       SwiGLUMLP
    #
    # forward(x, freqs, mask):
    #   x = x + self.attn(self.attn_norm(x), freqs, mask)   # pre-norm
    #   x = x + self.mlp(self.mlp_norm(x))
    #   return x
```

Steps:
- Assemble from the classes above
- Load all 9 weights for layer 0 from `WeightLoader`
- Diff full block output against `transformers` layer 0
- Benchmark: full block latency (combines attn + mlp)

**Learned:** pre-norm residual stream, why residuals are critical for gradient flow in deep nets.

---

## Section 10 — Full Model (Prefill)

**Class:** `LlamaModel` in `model/llama.py`

```python
class LlamaModel(nn.Module):
    # embed:  TokenEmbedding
    # layers: nn.ModuleList of 28 TransformerBlocks
    # norm:   RMSNorm
    # head:   OutputProjection
    #
    # forward(token_ids) → logits (B, T, vocab_size)
```

Steps:
- Stack all 28 blocks, load all weights
- Run full prefill forward pass
- Verify final logits match `transformers` output (greedy next-token should agree)
- Benchmark: prefill latency at T=128, T=512, T=1024 — record tokens/sec

**Learned:** how the residual stream carries information across 28 layers, how tied embeddings appear in both embed and head.

---

## Section 11 — KV Cache

**Class:** `KVCache` in `model/kv_cache.py`

```python
class KVCache:
    # k_cache: (n_layers, batch, max_seq_len, n_heads_kv, head_dim)
    # v_cache: same shape
    # pos: int — current fill position
    #
    # update(layer_idx, k, v) → (k_full, v_full up to pos)
    # reset()
```

Modify `GroupedQueryAttention.forward` to accept an optional `KVCache`:
- If cache provided and `pos > 0`: append new K/V to cache, attend over full cached sequence
- If no cache: standard full-sequence attention (prefill)

Steps:
- Implement static pre-allocated cache
- Run prefill → populate cache → run 10 decode steps
- Verify each decode step output matches `transformers` with `use_cache=True`
- Benchmark: decode latency per token with and without cache

**Learned:** why O(T²) attention without cache is prohibitive, prefill vs decode memory access patterns, the two-phase inference loop.

---

## Section 12 — Sampling

**Functions** in `sampling.py` (no class needed — these are pure functions):

```python
def greedy(logits)           # argmax over vocab
def sample_top_k(logits, k)  # keep top-k, sample
def sample_top_p(logits, p)  # nucleus: keep tokens with cumulative prob <= p
def apply_temperature(logits, temp)  # divide by temp before softmax
def sample(logits, temp, top_k, top_p)  # compose all of the above
```

Steps:
- Implement each function independently
- Unit test: greedy always picks the highest logit
- Unit test: top_p with p=1.0 is equivalent to full distribution sampling
- Benchmark: sampling latency is trivial vs generation, but measure anyway

**Learned:** deterministic vs stochastic decoding, why temperature flattens/sharpens the distribution, what nucleus sampling actually filters.

---

## ✅ Section 13 — Generation Loop + Full Benchmark Suite  `DONE`

**File:** `generate.py`

```python
def generate(prompt, model, tokenizer, kv_cache, max_new_tokens,
             temp=1.0, top_k=50, top_p=0.9):
    # 1. encode prompt
    # 2. prefill: model.forward(prompt_tokens) → populates kv_cache
    # 3. decode loop: one token at a time, streaming
    # 4. stop at EOS or max_new_tokens
    # yields tokens as strings as they're produced
```

**Benchmark suite** in `benchmarks/run_baseline.py`:
- Prefill throughput: tokens/sec at T = 64, 128, 256, 512, 1024
- Decode throughput: tokens/sec (this is what users feel)
- Per-op breakdown: time spent in RMSNorm / Attention / MLP / RoPE across one full forward pass
- Memory: peak VRAM during prefill and decode
- Save everything to `benchmarks/results_baseline.json`
- Print a formatted table

**Learned:** end-to-end inference loop, prefill vs decode latency split, where time actually goes (spoiler: matmuls in MLP and attention dominate).

---

## Section 14 — Triton Kernels (one at a time)

**Rule:** one kernel per PR. Swap it in, re-run `benchmarks/run_baseline.py`, compare the delta.

### 14a — RMSNorm kernel  `kernels/rmsnorm_kernel.py`  ✅ DONE
- One Triton program per row
- Accumulate sum-of-squares in a loop over tiles, compute scale, write output in one pass
- Expected win: moderate — RMSNorm is memory-bound, kernel fusion eliminates one read+write round-trip
- **Result: 3.5–7.8× speedup, 81% of peak BW at T=8192. Autotuned (warps=8, stages=3-4).**

### 14b — SwiGLU fusion kernel  `kernels/swiglu_kernel.py`  ✅ DONE
- **Level 1** (activation fusion): fused `silu(gate) * up` into one Triton kernel
  - Result: 1.3–1.7× on the isolated step (84% of peak BW); 1–6% end-to-end MLP gain since GEMMs dominate. Autotuned (BLOCK_SIZE=2048, warps=4).
- **Level 2** (gate+up matmul concat): single `w_gate_up` of shape `(2*I, H)`, one fused GEMM + chunk(2)
  - Result on RTX 6000 Ada single-GPU bf16: **wash (0.95×–1.02×)**. cuBLAS launches are ~5µs and tile utilization is already saturated at these shapes.
  - Kept anyway — this is the production pattern (vLLM, SGLang, TensorRT-LLM all do this). The win materializes under tensor parallelism (fewer all-gathers), CUDA graphs (fewer ops in graph), and quantization (fewer dequant kernels).
- **Level 3/4** (matmul+activation epilogue, fuse down matmul): deferred — requires beating cuBLAS at GEMM, multi-week project.

### 14c — RoPE kernel  `kernels/rope_kernel.py`  ✅ DONE
- One Triton program per (token, head); processes both halves of the rotation
  in a single pass — no `rotate_half` materialization.
- **Model-agnostic:** `INTERLEAVED` constexpr selects pair layout at compile
  time. Supports NEOX-style split-half (Llama, Mistral, Qwen, Yi, DeepSeek,
  Phi-3) and GPT-J adjacent-pair (GPT-J, GPT-NeoX-original, ChatGLM).
- Accepts cos/sin in either `(T, head_dim)` (HF duplicated) or `(T, head_dim/2)`
  (raw) layout via a runtime row stride. Zero-copy — no slice + contiguous.
- Q and K share the same kernel via two launches in `rope_triton(q, k, ...)`.
- Autotuned over `num_warps ∈ {1,2,4}`, `num_stages ∈ {1,2,3}`.
- **Result on RTX 6000 Ada bf16: 2.1–4.0× speedup. 72.7% peak BW at T=8192
  (vs 22.1% for PyTorch). Small-T wins are launch-overhead-limited
  (4× at T=1 but still only 0.3% peak BW).**
- **Optimization attempts tried and rejected:**
  - *Multi-head per program* (`BLOCK_H` tile to amortize cos/sin loads):
    regressed T=128–2048 (2.13× vs 3.29×). cos/sin is only ~5–10% of total
    traffic so the saving was real but the 2D indexing/masking overhead
    in Triton dominated. Reverted.

### 14d — Attention kernel  `kernels/attention_kernel.py`
- Write a naive (non-flash) Triton attention kernel:
  - Compute full QK^T, apply mask, softmax row-wise, multiply V
  - This materializes the full T×T matrix — pedagogically important, shows the memory wall
- Then switch to `flash_attn` or `torch.sdpa` and show why tiling matters
- Expected win: large at long sequences — naive attention is O(T²) memory, flash is O(T)

### 14e — Final benchmark comparison
Re-run `benchmarks/run_baseline.py` with all kernels active.  
Print side-by-side: baseline vs Triton, delta per op, overall tokens/sec improvement.  
Plot the roofline: is each op compute-bound or memory-bound on RTX 6000 Ada (960 GB/s bandwidth, 1457 TFLOPS BF16)?

---

## Phase 3 — Serving Optimizations (Sections 15–19)

> Up to here we've optimized **one request**. Phase 3 shifts the question from
> *"how fast can I run one prompt?"* to *"how many concurrent requests can I
> serve, with what latency?"*. This is where production engines (vLLM, SGLang,
> TensorRT-LLM) actually live, and where the big wins come from.
>
> **New mental model:** the engine is now a long-running loop that ingests
> requests, schedules them into batches per decode step, and streams tokens
> back. The forward pass is a tool the scheduler calls — not the top-level
> object.
>
> **Independent branches:** Sections 16 (PagedAttention) and 17 (RadixAttention)
> each build directly on Section 15 — they do **not** depend on each other.
> Implement and benchmark each in isolation first to see what each one
> contributes on its own. The natural combination (radix sharing on top of
> paged blocks) can be done later as a follow-up; the goal here is clean
> measurement of each idea separately.

---

## Section 15 — Continuous Batching + Scheduler  *(vLLM-style)*

**Files:** `serving/request.py`, `serving/scheduler.py`, `serving/engine.py`

**Problem it solves**
Static batching pads every sequence to the longest one and waits for all
sequences to finish before admitting the next batch. A 2048-token request in
the same batch as a 64-token request keeps the short one idle for ~30× longer
than necessary. GPU utilization plummets at mixed-length workloads.

**Solution: iteration-level scheduling**
At every decode step the scheduler:
1. Evicts finished requests (hit EOS or max_new_tokens) from the running batch
2. Admits waiting requests up to a token budget (e.g., 4096 tokens across all running requests)
3. Calls one forward pass that mixes prefills + decodes of different requests

```python
class Request:
    # id, prompt_tokens, max_new_tokens, generated_tokens
    # state: WAITING | PREFILL | DECODE | FINISHED
    # kv_cache_slot: assigned when admitted

class Scheduler:
    waiting:  deque[Request]
    running:  list[Request]
    def step() -> list[Request]:           # picks next batch each iteration
    def admit(r: Request): ...
    def evict(r: Request): ...

class InferenceEngine:
    model:     LlamaModel
    kv_cache:  KVCache
    scheduler: Scheduler
    def add_request(prompt, max_new_tokens) -> request_id
    def step() -> dict[request_id, new_token]      # one iteration
```

The model forward must now accept **ragged batches**: a packed `(total_tokens, hidden)`
tensor with a `cu_seqlens` array of cumulative offsets. SDPA gets called per-request
(or use a varlen-aware attention op).

**Steps**
- Define `Request` lifecycle states and transitions
- Implement FCFS scheduler with a token budget
- Refactor model forward to accept packed/ragged inputs
- Replace `iterations/03_sampling.py` style single-request loop with engine.step()
- Verify generation still matches transformers for each request

**Benchmark workload (add to suite as workload #6):**
32 requests, prompt lengths uniform in [64, 1024], 256 decode tokens each.

**Expected win:** 2-4× total throughput vs static batching on mixed-length workloads.
TTFT drops sharply for short requests that no longer wait behind long ones.

**Learned:** iteration-level vs request-level scheduling, ragged batching,
why the forward pass is no longer the unit of work.

---

## Section 16 — PagedAttention  *(vLLM's signature optimization — independent)*

**Builds on:** Section 15 (continuous batching). Independent of Section 17.

**Files:** `serving/paged_kv_cache.py`, `serving/block_allocator.py`,
modify `ops/attention.py` to add `paged_attention_forward()`

**Problem it solves**
The Section 11 KV cache pre-allocates `(max_batch, max_seq_len, n_kv_heads, head_dim)`
per request. With `max_seq_len=131072`, one slot reserves ~1 GB per layer just-in-case
the request might be long. Most requests are 200-1000 tokens → 99% of that memory is
wasted. You can fit maybe 4-8 concurrent requests instead of 64+.

**Solution: paged memory for KV cache**
Borrow virtual memory ideas from operating systems:
- Physically: KV cache is one big pool of fixed-size **blocks** (e.g., 16 tokens each)
- Logically: each request has a **block table** mapping its logical positions to
  physical block ids
- Allocate one block at a time, on demand, as a request grows
- No fragmentation, no pre-allocation, near-100% memory utilization

```python
BLOCK_SIZE = 16   # tokens per block

class BlockAllocator:
    free_blocks: list[int]
    ref_counts:  dict[int, int]
    def alloc() -> block_id
    def free(block_id): ...
    def incref(block_id): ...     # ref-counting kept for future radix integration

class PagedKVCache:
    # one big pool: (n_blocks, n_layers, 2, BLOCK_SIZE, n_kv_heads, head_dim)
    block_tables: dict[request_id, list[block_id]]
    def append_token(req_id, layer, k, v): ...
    def gather_kv(req_id, layer) -> (K, V)         # for attention

def paged_attention_forward(q, kv_cache, block_tables, ...):
    # for each query token, look up its request's block table,
    # gather the right physical blocks for K and V, run attention
```

The attention kernel changes: instead of contiguous K/V, it reads via the
block table. PyTorch SDPA can't do this directly — either materialize gathered
K/V (works but copies memory) or write a custom Triton kernel later.

**Steps**
- Implement `BlockAllocator` with a free-list
- Implement `PagedKVCache` with a block table per request
- Modify attention to gather K/V via block table (start with materialization; Triton paged-attn kernel can come later)
- Update scheduler to allocate/free blocks instead of full slots
- Verify decode output still matches transformers per request

**Benchmark workload (add as #7):**
Same as #6 but ramp concurrent requests until OOM. Report **max concurrency**.

**Expected win:**
- 5-10× more concurrent requests fit in same VRAM
- Often 3-5× throughput improvement at high load
- This is THE optimization that made vLLM the de-facto standard

**Learned:** OS-style paging for tensors, block tables, why fragmentation is the
hidden killer in naive KV caches, the gather-based attention pattern.

---

## Section 17 — RadixAttention / Prefix Caching  *(SGLang's signature optimization — independent)*

**Builds on:** Section 15 (continuous batching) **directly** — does *not* require
PagedAttention. We use a simple per-prefix KV store as a sidecar to the Section 11
contiguous KV cache. This isolates the *prefix-sharing* idea from the *paging* idea
so each can be benchmarked on its own.

**Files:** `serving/radix_cache.py`, integrate with `serving/scheduler.py`

**Problem it solves**
Real workloads have massive prefix overlap:
- Chat: every turn shares the same system prompt + conversation history
- Few-shot: every request shares the same N examples
- Agents: many parallel branches share the same prefix tool history

Without sharing: every request recomputes K/V for the same tokens, wastes both
compute (prefill) and memory (duplicate stored K/V). RadixAttention shares stored
K/V across requests via a radix tree keyed by token sequences.

**Solution: radix tree over token sequences (cache-aside the KV store)**
- Each tree node stores: (token sequence segment, K/V tensors for those tokens, ref_count)
- `match(tokens)` walks the tree to find the longest cached prefix → returns
  cached K/V (per layer) + remaining tokens to compute
- `insert(tokens, k_per_layer, v_per_layer)` adds a new path; shared segments
  bump ref_count so they aren't evicted while in use
- LRU eviction when total cached tokens exceed a configured budget; pinned
  (ref_count > 0) segments are skipped

```python
class RadixNode:
    children:   dict[tuple[int,...], RadixNode]
    k_cache:    torch.Tensor   # (n_layers, seg_len, n_kv_heads, head_dim)
    v_cache:    torch.Tensor   # same shape
    ref_count:  int
    last_used:  int            # for LRU

class RadixCache:
    root:         RadixNode
    total_tokens: int          # current cached size, capped at a budget
    def match(tokens) -> (cached_k, cached_v, remaining_tokens)
    def insert(tokens, k_per_layer, v_per_layer): ...
    def release(ref_handle): ...
    def evict_lru(target_tokens): ...
```

Scheduler integration: when admitting a request, first call
`radix.match(prompt_tokens)` →
1. The matched K/V is copied into the request's contiguous KV cache slot at positions [0, matched_len)
2. Prefill only runs on the remaining tail tokens → skipped compute
3. On request completion, optionally `insert()` the request's final K/V into the radix
4. The matched segment's ref_count is bumped while the request is live

**Steps**
- Implement radix tree with `match` / `insert` / `release` / `evict_lru`
- Store K/V per node as one tensor per layer (or one big stacked tensor across layers)
- Modify scheduler: prefix-match before admitting, only prefill the tail tokens
- Verify correctness: shared-prefix requests produce identical logits to non-shared baseline
- Watch for: cache invalidation on eviction must not happen while a request is still using a segment (ref_count guards this)

**Trade-off vs PagedAttention version**
Without paged blocks the radix cache stores K/V as fixed contiguous segments per
node, so memory layout is less efficient — but the logic is simpler and the
prefix-sharing win is fully measurable. After Section 17, integrating with the
Section 16 block pool is straightforward (replace per-node K/V tensors with
per-node block lists) and is the natural follow-up.

**Benchmark workload (add as #8):**
64 requests, all sharing a 256-token "system prompt", each with a unique 64-token user query, 128 decode tokens.

**Expected win:**
- 2-5× throughput on shared-prefix workloads
- TTFT drops to near-zero for cache hits (no prefill cost on the shared portion)
- This is what makes SGLang dominate in agent/chat workloads

**Learned:** radix trees, ref-counted memory, LRU eviction with pinning, why
shared compute is the highest-leverage optimization for real workloads.

---

## Section 18 — Chunked Prefill

**Builds on:** Section 15 (continuous batching). Orthogonal to 16/17 —
works with whichever KV cache you picked.

**Files:** modify `serving/scheduler.py`, no new classes

**Problem it solves**
A long prefill (4096 tokens) takes ~50-100ms of GPU time. During that pass, no
decode tokens can be produced for the *already-running* requests → their
inter-token latency spikes and tail latency goes through the roof.

Worse: prefill is compute-bound (large matmuls), decode is memory-bound
(loading weights for one token). Running them separately leaves one resource
idle the whole time.

**Solution: split prefills into chunks, mix with decode in one batch**
- Pick a prefill chunk size (e.g., 512 tokens)
- Long-prefill requests are split into chunks; each iteration the scheduler
  picks ONE chunk + the decode tokens of all running requests
- Total tokens per iteration capped at a budget (e.g., 2048)
- This packs both resources: matmuls saturate FLOPS, decode reads saturate bandwidth

```python
# inside scheduler.step()
batch_tokens = 0
batch_requests = []

# admit decodes first (cheap, one token each)
for r in self.running_decode:
    if batch_tokens + 1 <= TOKEN_BUDGET:
        batch_requests.append((r, [next_token]))
        batch_tokens += 1

# fill the rest with one prefill chunk
if self.waiting_or_partial_prefill:
    r = self.next_prefill_request()
    chunk_size = min(PREFILL_CHUNK, TOKEN_BUDGET - batch_tokens, r.remaining_prefill_tokens)
    batch_requests.append((r, r.next_chunk(chunk_size)))
```

**Steps**
- Add `prefill_progress` field to `Request`
- Split scheduler admission: handle decode and partial-prefill paths
- Token budget per iteration as a config
- Verify no per-request correctness regression (split prefill must produce same K/V as full prefill)

**Benchmark workload (add as #9):**
Mix: 8 requests with 4096-token prompts + 16 requests with 64-token prompts,
all with 128 decode tokens. Measure p50/p95/p99 TTFT and ITL.

**Expected win:**
- 30-50% reduction in tail decode latency (p95 / p99)
- Slightly higher *average* prefill latency (intentional trade-off)
- Smoother TTFT distribution under load

**Learned:** compute-bound vs memory-bound op mixing, token budgets,
trading worst-case for tail-case.

---

## Section 19 — CUDA Graphs  *(decode-path latency killer)*

**Builds on:** Section 15+ (any combination of 16/17/18). Works with the
Section 11 contiguous KV cache and with PagedKVCache — both pre-allocate
pointers, which is the only requirement for graph capture.

**Files:** `serving/cuda_graph_pool.py`, modify decode path in `serving/engine.py`

**Problem it solves**
A single decode token through 28 layers issues ~300 CUDA kernel launches.
At ~5-10 µs launch overhead each, that's 1.5-3 ms per token of pure launch
cost — often **larger than the actual GPU compute** for small-batch decode.
The CPU becomes the bottleneck even though the GPU is sitting half-idle.

**Solution: capture the whole decode forward as a CUDA graph, replay it**
- During warmup: run forward once with `torch.cuda.graph()` capturing → records
  the kernel sequence into a graph object
- During inference: `graph.replay()` re-executes all 300 kernels with **zero
  CPU overhead** — single API call to the driver
- Graphs require **static shapes** and **static memory addresses**:
  - Pad batch to one of N preset sizes (e.g., 1, 2, 4, 8, 16, 32, 64)
  - Capture one graph per batch size → `CudaGraphPool`
  - Input/output tensors must live in fixed buffers; copy data in/out per step
- Only applies to **decode** (prefill has variable token count, not worth it)

```python
class CudaGraphPool:
    graphs: dict[int, torch.cuda.CUDAGraph]
    static_input:  dict[int, torch.Tensor]      # persistent input buffer per size
    static_output: dict[int, torch.Tensor]
    def capture(batch_size, model_forward): ...
    def run(batch_size, real_input) -> output:
        self.static_input[bs].copy_(real_input)
        self.graphs[bs].replay()
        return self.static_output[bs].clone()
```

**Steps**
- Identify pad sizes (powers of 2 up to max_batch is a good default)
- Warmup loop: capture one graph per size
- Modify decode path: round actual batch up to nearest captured size, copy inputs to static buffer, replay
- Handle KV cache pointer stability: the KV cache (either Section 11's contiguous one or Section 16's paged one) must hand out the same buffer pointers across replays. Both schemes satisfy this since memory is pre-allocated; the gotcha is only with dynamic re-allocation between iterations — guard against it.
- Verify decode output bit-identical to non-graph path

**Caveats:**
- Memory: each graph holds a copy of intermediate tensors → ~100-300 MB per captured size
- Doesn't help prefill at all — keep prefill on the eager path
- If shapes change (e.g., dynamic block tables), need to capture with placeholders or fall back to eager

**Benchmark:**
Re-run workloads #6-#9 with graphs on/off. Decode-heavy workloads see the largest win.

**Expected win:**
- 1.5-2× decode throughput at batch ≤ 8 (where launch overhead dominates)
- Diminishing returns at large batches (compute dominates launch overhead)

**Learned:** CPU vs GPU as the bottleneck for small-batch decode, graph
capture/replay semantics, why static shapes + static memory are the cost of admission.

---

## Section 20 — Speculative Decoding  *(decode-throughput multiplier)*

**Builds on:** Section 19. Works with paged or contiguous KV.

**Files:** `serving/speculative.py`, `model/draft_model.py`

**Problem it solves**
Autoregressive decode is fundamentally sequential — one token per forward
pass. Even with CUDA graphs, a 30B model decodes at maybe 30–60 tok/s on a
single 48 GB Ada. Memory bandwidth, not compute, is the bottleneck: each
decode step reads the full weights once to emit one token.

**Solution: emit K tokens per heavy-model forward**
- **Draft** model proposes K candidate tokens cheaply (small model, EAGLE
  head, n-gram match against the prompt, or Medusa heads on the base model)
- **Verify** by running the heavy model on all K positions **in parallel**
  (one forward, K logits, no K× cost — the model is memory-bound at batch ≈1)
- Accept the longest prefix where draft and verify agree under the sampling
  policy. Discard the rest. **Acceptance rate of 60–80 % is typical.**

```python
# Per decode step
draft_tokens = draft.propose(last_tok, K=5)            # K candidates
logits       = heavy.forward(draft_tokens, kv_cache)   # one parallel forward
accept_len   = longest_prefix_where(sample(logits) == draft_tokens)
emit(draft_tokens[:accept_len])
rollback_cache(K - accept_len)                         # drop rejected K/V
```

**Three flavors to study (build in this order)**
1. **N-gram speculation** — no draft model. Match the last few decoded tokens
   against the prompt itself; if matched, propose the next K tokens from
   the prompt. Free, zero-VRAM, surprisingly effective on RAG / code / quote-heavy workloads.
2. **Small draft model** — e.g. Llama-3.2-1B drafting for Llama-3.1-32B.
   Plain two-model setup, simplest to reason about.
3. **EAGLE / Medusa heads** — small extra heads on the base model predict
   the *features* of the next K tokens, then a tiny LM head decodes them.
   Higher acceptance than a separate draft (shares features), and no second
   model to load.

**Steps**
- Implement n-gram first (no extra weights, isolates the verification logic)
- Then wire the draft model + KV cache for *both* models
- Carefully handle: rollback of rejected K/V in BOTH caches, sampling-policy-correct verification (greedy is easy, top-p requires Leviathan-style rejection sampling for unbiased acceptance)
- Capture a CUDA graph for the K-position verify forward

**Benchmark:**
Decode tok/s with K=1 (= no speculation), K=3, K=5, K=8. Acceptance rate per workload.

**Expected win:**
- 1.5–2.5× decode throughput on dense models (memory-bound regime)
- Larger wins as model size grows — this is **the** lever for 30B+ dense models

**Learned:** memory-bandwidth bottleneck of decode, parallel verification as a free lunch in the memory-bound regime, draft-quality vs acceptance-rate trade-off, why speculative decoding *doesn't* help when you're already compute-bound (large batches).

---

## Section 21 — Quantization  *(makes big dense models fit)*

**Builds on:** Sections 7, 8 (matmuls), 11/16 (KV cache).

**Files:** `quant/w4a16.py`, `quant/fp8_kv.py`, optionally `kernels/marlin_kernel.py`

**Problem it solves**
A 32B dense model is ~64 GB in BF16 → does not fit on a single 48 GB Ada.
At W4A16 (4-bit weights, 16-bit activations) it's ~16 GB. With FP8 KV cache
on top, a 32B model with 8k context fits comfortably with room for batching.
**Quantization is the single biggest knob for "which models we can run."**

**Two independent axes**

### 21a — Weight quantization (W4A16, W8A16)
- Per-channel or per-group (group_size=128 is the de-facto standard)
- Quantize offline; load quantized weights at startup
- At decode the matmul is W4A16: dequantize a tile of W to BF16 in registers, fuse into the matmul
- **Kernel matters enormously** — naive dequant-then-matmul loses to BF16; the win comes from Marlin/Machete-style kernels that dequantize *inside* the MMA loop
- Calibration: AWQ (activation-aware scaling) is the cheap-and-good default; GPTQ is a more thorough Hessian-based method

### 21b — KV cache quantization (FP8 / INT8)
- KV cache often eats more memory than weights at long context — quantizing it doubles concurrency or context length
- Per-head, per-tile dynamic scaling
- Quality cost: negligible at FP8, mild at INT8

**Steps**
- 21a: implement a reference W4A16 dequant + matmul in PyTorch, verify vs BF16
- 21a: replace the slow path with a Marlin-style Triton kernel (could be Section 14f material)
- 21b: add FP8 mode to the KV cache; convert on write, dequant on read inside attention
- Verify perplexity on a small calibration set vs full BF16 (target: < 1 % degradation)

**Benchmark:**
- Same workload, BF16 vs W4A16 vs W4A16+FP8-KV
- Per-token latency, peak VRAM, **max concurrent requests** (this is where the real win shows)

**Expected win:**
- W4A16: ~3.5× weight memory reduction; matmul throughput ~equal or slightly faster than BF16 with a good kernel (it's memory-bound!)
- FP8 KV: 2× more concurrent requests or 2× context length at iso-VRAM

**Learned:** memory hierarchy of LLM inference (weights vs KV vs activations), why W4A16 is faster than BF16 in the memory-bound regime, calibration trade-offs, kernel-level fusion as the price of admission.

---

## Section 22 — Zero-overhead Scheduler  *(Python overhead killer)*

**Builds on:** Section 15 (scheduler), Section 19 (CUDA graphs).

**Files:** `serving/async_scheduler.py`, `serving/tokenizer_worker.py`

**Problem it solves**
Once CUDA graphs collapse GPU launch overhead and quantization makes weights
cheap to read, **the bottleneck moves to Python**. A naive scheduler does
~3–5 ms of work per step (token sampling, sequence updates, prepping the
next batch's metadata, detokenizing finished tokens). At 200 tok/s that's
the entire step time — the GPU sits idle waiting for Python.

**Solution: overlap scheduler work with GPU compute**

```
GPU:    [forward step N] [forward step N+1] [forward step N+2] ...
CPU:               [sched N+1, detok N-1] [sched N+2, detok N] ...
```

- **Async output thread** — detokenization runs on a worker; the engine
  pushes raw token ids into a queue and returns immediately
- **Multi-step lookahead** — prepare the *next* step's batch metadata
  (block tables, position ids, sampling params) while the GPU runs the
  current step
- **Async tokenizer** — new requests are tokenized on a CPU thread, not
  in the hot path
- **Persistent decoder state** — keep sequence objects, sampling state,
  and stopping criteria on the engine side so the scheduler never copies
  big Python objects per step

**Steps**
- Profile where Python time actually goes (cProfile + py-spy) — surprises
  abound (often it's `tokenizer.decode` or list comprehensions)
- Hoist all per-step Python work into either: pre-computed at request
  admission, or post-step on a worker thread
- Use lock-free `collections.deque` / `asyncio.Queue` between engine and workers
- Verify: GPU SM utilization should approach 100 % during sustained decode

**Benchmark:**
Decode tok/s on workload #5 (1024 sequences, 32 batch) with naive scheduler vs async.
GPU SM utilization sampled with `nvidia-smi dmon`.

**Expected win:**
- 1.3–1.8× decode throughput at small batch (where Python time is most exposed)
- GPU utilization climbs from ~60 % to ~95 %

**Learned:** profiling Python overhead in tight inference loops, async pipelining as a near-free lunch, why the last 30 % of GPU utilization usually lives in your scheduler.

---

## Section 23 — Prefill / Decode Disaggregation  *(optional, multi-process)*

**Builds on:** Section 18 (chunked prefill is a single-GPU alternative).
Requires Section 16 (paged cache) for cross-process KV handoff.

**Files:** `serving/prefill_worker.py`, `serving/decode_worker.py`, `serving/kv_transfer.py`

**Problem it solves**
Prefill is compute-bound; decode is memory-bound. Mixing them on one GPU
means either prefill steals decode tokens' latency (head-of-line blocking)
or decode underutilizes the prefill-sized batch. Chunked prefill (§18)
mitigates this on a single GPU; *disaggregation* solves it by giving each
phase its own GPU(s).

**Architecture**
- **Prefill workers** run prompts → produce KV cache. Optimized for big
  prefill batches, no decode latency target.
- **Decode workers** receive KV from prefill workers, run the decode loop.
  Optimized for low-latency decode (CUDA graphs, speculation).
- **KV transfer** moves the cache between workers — over NVLink if
  co-located, RDMA / shared memory / NCCL P2P otherwise.

**Why we put this last in Phase 3**
- It's a *systems* optimization, not a model optimization
- The wins only materialize at production-scale traffic with mixed
  long-prefill + steady-decode load
- Single-GPU deployments get most of the benefit from chunked prefill (§18) — disaggregation is the next step *only* when you have multiple GPUs and an SLO-sensitive workload

**Benchmark:**
Mixed workload (#7 + #5 concurrent) on (a) single GPU with chunked prefill, (b) two GPUs with disaggregation. Measure: p50/p95/p99 TTFT, p50/p95/p99 ITL, total throughput.

**Learned:** how Pareto-optimal SLOs differ for compute-bound vs memory-bound phases, KV-transfer engineering as a real bottleneck (often the new critical path), when disaggregation is *not* worth it (low traffic, balanced workloads).

---

## Phase 4 — Dense Model Expansion  *(generalize beyond 3B)*

This is where the engine stops being a Llama-3.2-3B project and becomes a
real dense-model inference engine. **Everything from Phases 1–3 must
continue to work**, validated on each new model.

Target models (single-GPU on 48 GB Ada, with §21 quantization):
- **Llama-3.1-8B**  (BF16 fits; baseline for "bigger than 3B")
- **Qwen2.5-14B**   (BF16 fits with tight margins; W8A16 comfortable)
- **Qwen2.5-32B / Llama-3.1-Nemotron-32B-class**  (requires W4A16)
- **Mistral-Small-24B-class**

We deliberately stay with **dense** transformers — no MoE in this phase.
Sparse models change the parallelism story; we cover that separately in
Phase 6.

---

## Section 24 — Generic Dense-Model Support

**Files:** `config.py` (generalize), `loader.py` (broaden), `model/llama.py` → `model/dense_lm.py`

**Goal**
Replace the hardcoded Llama-3.2-3B config with a model-family abstraction
that can express the variations found across modern dense LMs:
- Hidden size, layer count, head count, KV-head count (GQA / MHA / MQA)
- RoPE base / scaling (linear, YaRN, NTK-aware) — Llama-3 vs Qwen2.5 differ
- Activation (SwiGLU vs GeGLU)
- Norm placement (pre-norm vs sandwich-norm)
- Tied vs untied embeddings
- Vocab size (and special-token handling per tokenizer family)

**Steps**
- Refactor `ModelConfig` into a registry: `from_hf_config(hf_cfg) → ModelConfig`
- Audit `ops/rope.py` for scaling-method support (Llama-3 uses scaled RoPE, Qwen2.5 plain)
- Audit `model/block.py` for activation-fn and norm-placement parametrization
- Audit `loader.py` weight-name maps for each model family
- Verify every model against HF `transformers` (same correctness harness as Section 10)

**Deliverable:** a single `serve(model_id, prompt)` entry point that works on Llama-3.1-8B, Qwen2.5-14B, Qwen2.5-32B-W4A16, and Llama-3.2-3B (no regressions on the original).

**Learned:** the small but real divergence across "Llama-like" families, why most papers ship a model-family adapter rather than reimplementing the world.

---

## Section 25 — Cross-model Validation Suite

**Files:** `benchmarks/run_dense_suite.py`

**Goal**
Re-run the iteration-12 (full Phase-3) engine on every supported model and
produce a comparison matrix:

| model            | bf16/quant | prefill tok/s | decode tok/s | peak VRAM | max concurrent |
|------------------|-----------|--------------:|-------------:|----------:|---------------:|
| Llama-3.2-3B     | BF16      |               |              |           |                |
| Llama-3.1-8B     | BF16      |               |              |           |                |
| Qwen2.5-14B      | BF16      |               |              |           |                |
| Qwen2.5-14B      | W4A16     |               |              |           |                |
| Qwen2.5-32B      | W4A16     |               |              |           |                |

Every Phase-3 optimization (paged, radix, chunked, graphs, speculation,
quant) must be on simultaneously. If any single one regresses for a
specific model family, that's the bug to fix before Phase 6.

**Learned:** how engine performance characteristics shift with model size — what was decode-latency-bound at 3B may become weight-memory-bound at 32B.

---

## Phase 5 — Distributed Serving with Ray  *(production layer)*

> Up to here the engine is a Python object you call from a script. Phases 1–4
> made it *fast* and *general*; they did not make it a *service*. Phase 5 turns
> the in-process engine into a long-running, fault-tolerant, horizontally
> scalable service — without changing a line of the model or scheduler code.
>
> **Why Ray:** it is the de-facto orchestration substrate for production LLM
> serving (vLLM and friends drive their distributed workers as Ray actors). It
> gives us three things we don't want to hand-roll: (1) a stateful **actor**
> model so the engine loop lives in its own process with its own GPU, (2) **Ray
> Serve** for HTTP/streaming ingress, replicas, and autoscaling, and (3) a clean
> path into Phase 6 — multi-GPU workers become just more actors under a
> placement group.
>
> **Scope discipline:** one GPU per engine actor here. We are NOT sharding a
> model across GPUs yet (that's Phase 6). We ARE running N independent engine
> replicas behind one API, each serving the whole model. Everything from
> Phases 1–4 keeps working *unchanged* inside the actor — this phase is a
> wrapper, not a rewrite.

---

## Section 26 — Engine as a Ray Actor

**Builds on:** all of Phase 3 (the `InferenceEngine`), and Phase 4 if serving the larger models.

**Files:** `serving/ray_engine.py`

**Problem it solves**
The Phase 3 `InferenceEngine` is a synchronous object: you call `step()` in a
loop on the main thread. To serve concurrent clients the engine loop must run
continuously in its own process, own its GPU, hold the KV cache and scheduler
state across requests, and accept/stream requests asynchronously. That is
exactly a stateful actor.

**Solution: wrap the engine in a long-lived async actor**
- `@ray.remote(num_gpus=1)` actor owns one `InferenceEngine` (model + KV cache + scheduler)
- An async background task drives `engine.step()` continuously while there is work, and sleeps when idle
- `generate()` enqueues into the scheduler and returns an async stream of tokens
- Per-token outputs are pushed back to callers via per-request `asyncio.Queue`s

```python
@ray.remote(num_gpus=1)
class EngineActor:
    def __init__(self, model_id, engine_config):
        self.engine = InferenceEngine(...)          # Phase 3 engine, unchanged
        self._streams: dict[int, asyncio.Queue] = {}
        self._wakeup = asyncio.Event()

    async def _run_loop(self):
        while True:
            if self.engine.idle():
                await self._wakeup.wait(); self._wakeup.clear()
            outputs = self.engine.step()            # one scheduler iteration
            for req_id, tok in outputs.items():
                self._streams[req_id].put_nowait(tok)

    async def generate(self, prompt_tokens, params):   # async generator
        req_id = self.engine.add_request(prompt_tokens, params)
        self._streams[req_id] = asyncio.Queue()
        self._wakeup.set()
        async for tok in self._drain(req_id):
            yield tok
```

**Steps**
- Add `ray` to deps; `ray.init()` local bring-up
- Wrap `InferenceEngine` in `EngineActor`; move the `step()` loop into an async task
- Make `add_request` safe to call from RPC handlers while the loop is running (the scheduler is now touched from two places)
- Stream tokens out via per-request `asyncio.Queue`; clean up state on completion/cancel
- Verify: generations are bit-identical to the in-process Phase 3 engine on the five workloads

**Benchmark (iteration `20_ray_engine.py`):**
Run the five workloads through the actor. Compare against the in-process engine —
the delta is pure actor/serialization overhead.

**Expected win:** none on raw throughput — this is a *refactor*, not an
optimization. Target: actor overhead < a few %, and the engine now runs detached
from the caller. That decoupling is the deliverable.

**Learned:** stateful actors, the async-loop-plus-queue pattern every production
engine uses (vLLM's `AsyncLLMEngine` is exactly this), why the serving loop must
be decoupled from request submission.

---

## Section 27 — Ray Serve Production API

**Builds on:** Section 26.

**Files:** `serving/ray_serve_app.py`, `serving/openai_schema.py`

**Problem it solves**
An actor handle is still an in-cluster Python API. Real clients speak HTTP, want
token streaming (SSE), expect an OpenAI-compatible schema so existing tooling
"just works", and the service must scale replicas up/down with load and survive
a worker crash.

**Solution: a Ray Serve deployment in front of the engine actor(s)**
- An ingress deployment exposes `/v1/chat/completions` and `/v1/completions` (OpenAI-compatible), with SSE streaming
- It applies the chat template, validates, routes to an `EngineActor` replica, and streams tokens back
- Autoscaling config scales engine replicas with load; health checks + automatic restart give fault tolerance
- Ingress (CPU-bound: tokenization, templating) and engine (GPU-bound) are *separate* deployments so they scale independently

```python
@serve.deployment(autoscaling_config={"min_replicas": 1, "max_replicas": 4},
                  ray_actor_options={"num_gpus": 1})
class EngineReplica:
    def __init__(self):
        self.engine = EngineActor.remote(...)
    async def stream(self, prompt_tokens, params):
        async for tok in self.engine.generate.remote(prompt_tokens, params):
            yield tok

@serve.deployment
@serve.ingress(app)                                  # FastAPI app
class OpenAIIngress:
    @app.post("/v1/chat/completions")
    async def chat(self, body: ChatRequest):
        toks = self.tokenizer.apply_chat_template(body.messages)
        return StreamingResponse(self._sse(self.engine.stream(toks, body.params)))
```

**Steps**
- Define OpenAI-compatible request/response models (chat + completions) in `openai_schema.py`
- Build the FastAPI ingress; wire SSE streaming for `stream=true`
- Bind ingress → `EngineReplica`; set autoscaling target (e.g. on ongoing-requests / queue depth)
- Apply the tokenizer chat template at the ingress (keep the engine token-only)
- Verify: the `openai` Python client and `curl` both stream correctly; multiple concurrent clients are served

**Benchmark (iteration `21_ray_serve.py`):**
Closed-loop load test (async client / locust): requests/sec, TTFT and
inter-token-latency distributions at 1 / 8 / 32 / 128 concurrent clients;
autoscaling reaction time as load ramps up and down.

**Expected win:** this is about *serving*, not tok/s — confirm HTTP+SSE overhead
is small vs generation time, throughput scales ~linearly with replicas until
GPU-bound, and the API is drop-in OpenAI-compatible.

**Learned:** Ray Serve deployments/replicas/ingress, the OpenAI API surface, SSE
streaming, autoscaling on serving signals, why ingress and engine are split into
separate deployments.

---

## Section 28 — Observability & Production Hardening

**Builds on:** Section 27.

**Files:** `serving/metrics.py`, `serving/production.py`

**Problem it solves**
A service you can't see into isn't production-ready. You need per-request and
per-replica metrics, graceful shutdown that drains in-flight requests,
backpressure when overloaded (shed/queue rather than OOM), and request
cancellation when a client disconnects mid-stream (free its KV blocks
immediately, not at EOS).

**Solution: instrument and harden the serving layer**
- **Metrics:** TTFT, ITL, tokens/sec, queue depth, running/waiting counts, KV-cache utilization, GPU memory — exported to Prometheus (Ray ships a metrics endpoint), with a Grafana panel set
- **Backpressure:** cap admitted requests / total KV budget; past the limit return 429 (or queue with a timeout) instead of OOMing
- **Graceful drain:** on `SIGTERM`, stop admitting, finish in-flight requests, then tear down actors
- **Cancellation:** client disconnect → cancel the request → scheduler evicts it and frees KV blocks the same step

**Steps**
- Emit metrics from the engine loop (cheap counters/histograms; sample — never log per token in the hot path)
- Add a Prometheus scrape target; sketch the four golden signals + KV utilization panel
- Implement backpressure in the admission path
- Wire client-disconnect → request cancellation → KV free
- Implement graceful drain

**Benchmark / validation:**
- **Overload test** past capacity → service sheds load (429s, bounded latency) instead of OOMing or melting tail latency
- **Kill an engine actor** mid-load → Serve restarts it; in-flight requests on healthy replicas are unaffected
- **Disconnect a streaming client** → its KV blocks free within one step

**Learned:** the four golden signals for an inference service, backpressure vs
graceful degradation, why request cancellation is a memory-management concern in
LLM serving, what "production-ready" requires beyond raw speed.

---

## Phase 6 — Multi-GPU Parallelism  *(only after Phase 4 works)*

Single-node, 2 or 4 GPU configurations (NVLink between RTX 6000 Ada cards,
or the next-tier hardware we move to). **All earlier phases continue to
hold**; parallelism is an additive layer.

We do **not** cover multi-node here. Multi-node adds an interconnect
(InfiniBand / RoCE) and a fault-tolerance story that doubles the project's
surface area for marginal pedagogical gain.

---

## Section 29 — Tensor Parallelism  *(weight split across GPUs)*

**Builds on:** all of Phases 1–4.

**Files:** `parallel/tp.py`, `parallel/layers.py`

**The Megatron split** (read the Megatron-LM paper first)
- **Attention QKV:** column-parallel — each rank holds `n_heads / TP` heads
  for Q, and `n_heads_kv / TP` heads for K, V (requires TP divides KV head count — for GQA-8 → TP ∈ {1, 2, 4, 8})
- **Attention output proj:** row-parallel
- **MLP up + gate:** column-parallel
- **MLP down:** row-parallel
- **One all-reduce per attention block, one per MLP block** — two collectives per layer

```python
# Column-parallel linear: split output dim
class ColumnParallelLinear(nn.Module):
    def forward(self, x):           # x is replicated across ranks
        return F.linear(x, self.weight_shard)   # output is sharded

# Row-parallel linear: split input dim, all-reduce the result
class RowParallelLinear(nn.Module):
    def forward(self, x_shard):     # x is sharded across ranks
        y_local = F.linear(x_shard, self.weight_shard)
        return all_reduce(y_local)  # SUM across ranks
```

**Steps**
- Initialize NCCL process group; one process per GPU
- Shard weights at load time (not at runtime) — `loader.py` learns about TP rank
- Wire all-reduce into the residual path *exactly* where Megatron specifies
- Verify: TP=2 output bit-similar (BF16 tolerance) to TP=1 on the same prompts
- Profile: where does NCCL all-reduce show up in the timeline? (Hint: it dominates decode.)

**Benchmark:**
Llama-3.1-32B (BF16) on TP=2 vs Qwen2.5-32B-W4A16 on TP=1.
Compare: prefill throughput, decode latency, peak VRAM per GPU.

**Expected win:**
- Lets us run models that don't fit on one GPU (32B BF16, or 70B W4A16)
- Per-GPU decode latency typically gets *worse* than single-GPU due to all-reduce — TP is for *capacity*, not for speed at the same model size

**Learned:** the difference between weight-parallel and data-parallel inference, why TP is a capacity tool not a latency tool, the all-reduce as the new critical path.

---

## Section 30 — Custom All-reduce for Decode  *(small-message latency killer)*

**Builds on:** Section 29.

**Files:** `parallel/custom_ar.py`, `kernels/all_reduce_kernel.py`

**Problem it solves**
NCCL is tuned for big payloads (gradient sync in training: tens of MB).
Decode all-reduce sends one activation tensor of shape `[batch, hidden]`
≈ 50–200 KB. NCCL's launch + scheduling overhead is comparable to the
actual data movement at this size, leaving 30–50 % of the all-reduce time
on the table.

**Solution: a CUDA-IPC-based custom reduce**
- Map each rank's output buffer into every other rank's address space via
  `cudaIpcOpenMemHandle`
- A single fused kernel reads from all peer pointers and writes the
  sum locally — no driver-side collective scheduling
- Falls back to NCCL above a payload threshold (~1 MB)

**Steps**
- Stand up CUDA IPC handle exchange at process-group init
- Implement the reduce kernel (Triton or CUDA C++)
- Hook into TP via a flag — same numerics as NCCL, swap-in only
- Measure: per-op all-reduce latency at 1 KB, 10 KB, 100 KB, 1 MB

**Benchmark:**
Single-token decode latency on a TP=2 32B model, NCCL vs custom AR.

**Expected win:**
- ~2× faster all-reduce in the < 100 KB regime
- 10–20 % end-to-end decode latency improvement

**Learned:** when collective-comm libraries are over-engineered for your payload size, CUDA IPC as a sharp tool, fused reduce kernels.

---

## Section 31 — Sequence Parallelism  *(halves activation memory in TP)*

**Builds on:** Section 29.

**Files:** `parallel/sp.py`

**The idea (Megatron-SP paper)**
TP shards weights but *replicates* activations across ranks for the
non-linear regions (RMSNorm, residual, dropout-equivalent). At long
context that activation memory is huge. Sequence Parallelism splits those
activations along the **sequence dimension** instead of replicating them.

**Net effect**
- Activation memory of norm/residual blocks shrinks by `1/TP`
- The two all-reduces per layer become `all-gather + reduce-scatter` —
  same total bytes, but the activation never lives un-sharded
- Enables 2–3× longer context at iso-VRAM

**Steps**
- Replace `all_reduce` after row-parallel matmul with `reduce_scatter`
- Replace activation broadcast before column-parallel matmul with `all_gather`
- Make sure RMSNorm operates on the sequence-sharded activation correctly
  (each rank holds a slice of tokens, computes its own row-wise norm)
- Verify numerics vs plain TP

**Learned:** how to find the activation-memory wins inside a parallelism scheme without changing the collectives' total cost.

---

## Section 32 — Two-batch Overlap  *(hide collectives under compute)*

**Builds on:** Sections 29–31.

**Files:** `parallel/tbo.py`

**The idea**
In TP, decode time is `compute + all_reduce + compute + all_reduce + ...`.
If we split the active batch in two micro-batches and pipeline them, we
can run `compute(A_layer_i)` *concurrently with* `all_reduce(B_layer_i-1)`
on a separate CUDA stream. The all-reduce becomes free as long as it
fits under the compute.

```
Stream 0 (compute): [A.L0] [B.L0] [A.L1] [B.L1] [A.L2] [B.L2] ...
Stream 1 (comm):           [A.L0]      [B.L0]      [A.L1]     ...
```

**Steps**
- Run forward on two CUDA streams, with explicit events for hand-offs
- Split the active batch by 2 just before the TP layers
- Re-merge before sampling
- Verify decode tok/s win vs vanilla TP

**Caveat:** this works for prefill more easily than decode (more compute to hide behind). For decode it needs decoders with enough hidden-dim that the compute > comm.

**Learned:** stream-level parallelism inside a single GPU, when comm/compute overlap is actually beneficial.

---

## Section 33 — Pipeline & Expert Parallelism  *(optional)*

**Pipeline parallelism (PP)** — split *layers* across GPUs, micro-batch the
input, pipeline forward passes. Useful only when TP saturates intra-node
NVLink and you need *more* GPUs (e.g. > 4) or when crossing nodes.
For ≤ 4 GPUs on one node, **TP almost always beats PP** for inference.

**Expert parallelism (EP)** — only relevant if we ever serve MoE models
(Mixtral, DeepSeek-V3, Qwen2-MoE). Different routing semantics, different
collective pattern (all-to-all instead of all-reduce). Big topic; treat as
a *separate* phase if we go there. Out of scope for the dense-model goal.

---

## Build Order Summary

```
Phase 1 — PyTorch baseline
  Section 1  ✅  config.py
  Section 2  ✅  loader.py
  Section 3  ✅  tokenizer.py
  Section 4  ✅  ops/rmsnorm.py
  Section 5  ✅  ops/embedding.py
  Section 6  ✅  ops/rope.py
  Section 7  ✅  ops/attention.py
  Section 8  ✅  ops/mlp.py
  Section 9  ✅  model/block.py
  Section 10 ✅  model/llama.py
  Section 11     model/kv_cache.py
  Section 12     sampling.py
  Section 13     generate.py + benchmarks/run_baseline.py

Phase 2 — Triton kernels (swap in one at a time)
  Section 14a    kernels/rmsnorm_kernel.py
  Section 14b    kernels/swiglu_kernel.py
  Section 14c    kernels/rope_kernel.py
  Section 14d    kernels/attention_kernel.py
  Section 14e    final benchmark comparison

Phase 3 — Serving optimizations (multi-request)
  Section 15     serving/scheduler.py        continuous batching
  Section 16     serving/paged_kv_cache.py   PagedAttention
  Section 17     serving/radix_cache.py      RadixAttention / prefix caching
  Section 18     serving/scheduler.py (mod)  chunked prefill
  Section 19     serving/cuda_graph_pool.py  CUDA graphs for decode
  Section 20     serving/speculative.py      speculative decoding (n-gram → draft → EAGLE)
  Section 21     quant/                       W4A16 weights + FP8 KV cache
  Section 22     serving/async_scheduler.py  zero-overhead scheduler / async pipeline
  Section 23     serving/{prefill,decode}_worker.py  prefill/decode disaggregation (optional)

Phase 4 — Dense model expansion (beyond Llama-3.2-3B)
  Section 24     model/dense_lm.py            generic dense-LM support (Llama 8B/Qwen 14B/32B/…)
  Section 25     benchmarks/run_dense_suite.py cross-model validation matrix

Phase 5 — Distributed serving with Ray (production layer)
  Section 26     serving/ray_engine.py        engine as a Ray actor (async loop)
  Section 27     serving/ray_serve_app.py     Ray Serve OpenAI-compatible API + autoscaling
  Section 28     serving/production.py        observability + production hardening

Phase 6 — Multi-GPU parallelism (single-node, 2 or 4 GPU)
  Section 29     parallel/tp.py               tensor parallelism (Megatron split)
  Section 30     parallel/custom_ar.py        custom all-reduce for decode
  Section 31     parallel/sp.py               sequence parallelism
  Section 32     parallel/tbo.py              two-batch overlap (comm/compute)
  Section 33     parallel/{pp,ep}.py          pipeline / expert parallelism (optional)
```

---

## Iterations Track

The `iterations/` folder captures how the engine improves as we add
optimizations. **Each file runs the exact same five workloads** so results
are directly comparable across iterations.

### The five workloads (fixed across all iterations)

| # | Total sequences | Batch size | Notes |
|---|----------------|------------|-------|
| 1 | 1              | 1          | single-sequence baseline |
| 2 | 8              | 8          | small batch, fits in one pass |
| 3 | 24             | 3          | 8 micro-batches of 3 |
| 4 | 128            | 16         | 8 micro-batches of 16 |
| 5 | 1024           | 32         | 32 micro-batches of 32 |

Each workload reports:
- **Latency per batch** (ms)
- **Total wall time** to process all N sequences (ms)
- **Throughput** (sequences/sec, tokens/sec)
- **Peak VRAM** (MB)

### Iteration files

```
iterations/
  01_naive_engine.py        ✅  LlamaModel, no KV cache, full prefill per step
  02_kv_cache.py                + KVCache: decode reuses past K/V (Section 11)
  03_sampling.py                + sampling loop: greedy / top-p / top-k (Section 12-13)
  04_triton_rmsnorm.py          + Triton RMSNorm kernel swap (Section 14a)
  05_triton_swiglu.py           + Triton SwiGLU fusion kernel (Section 14b)
  06_triton_rope.py             + Triton RoPE kernel (Section 14c)
  07_triton_attention.py        + Triton attention kernel (Section 14d)
  08_continuous_batching.py     + iteration-level scheduler (Section 15)
                                 ├─ branch A ─┐
  09_paged_attention.py         + PagedAttention KV cache (Section 16) — built on 08
                                 ├─ branch B ─┘
  10_radix_attention.py         + RadixAttention prefix sharing (Section 17) — built on 08, NOT 09
                                 │
  11_chunked_prefill.py         + chunked prefill scheduling (Section 18) — built on 09 (the main path)
  12_cuda_graphs.py             + CUDA graphs for decode (Section 19) — built on 11
  13_speculative.py             + speculative decoding (Section 20) — built on 12
  14_quantized.py               + W4A16 + FP8 KV (Section 21) — built on 13
  15_async_engine.py            + zero-overhead scheduler (Section 22) — built on 14
  16_disagg.py                  + prefill/decode disaggregation (Section 23, optional) — built on 15

  ── Phase 4: re-validate the engine on bigger dense models ──
  17_dense_lm_8b.py             same engine as 15, target = Llama-3.1-8B (BF16)
  18_dense_lm_14b.py            same engine as 15, target = Qwen2.5-14B (BF16 / W8A16)
  19_dense_lm_32b.py            same engine as 15, target = Qwen2.5-32B (W4A16 required)

  ── Phase 5: distributed serving with Ray (production layer) ──
  20_ray_engine.py              engine wrapped as a Ray actor (Section 26)
  21_ray_serve.py               Ray Serve OpenAI-compatible API + autoscaling (Section 27)

  ── Phase 6: multi-GPU (run via torchrun, 2–4 processes) ──
  22_tp.py                      tensor parallelism (Section 29)
  23_tp_custom_ar.py            + custom all-reduce (Section 30) — built on 22
  24_tp_sp.py                   + sequence parallelism (Section 31) — built on 23
  25_tp_tbo.py                  + two-batch overlap (Section 32) — built on 24
```

**Why 09 and 10 fork from 08 independently:** PagedAttention and RadixAttention
solve different problems (memory fragmentation vs prefix sharing). By forking
both from continuous batching (08), each iteration measures *only* what that
optimization contributes. After running both, you can integrate them as a
follow-up — `radix_cache.py` simply gets a `BlockAllocator` instead of its
per-node K/V tensors.

**Why 11 builds on 09 (not 10):** chunked prefill needs flexible KV allocation
during partial-prefill admission, which the paged allocator gives for free.
You can still measure 11's win without radix integrated.

Workloads #6–#9 are added in Phase 3 (mixed-length, max-concurrency,
shared-prefix, long-prefill) to actually exercise the new features.
The original five remain so single-request progress stays comparable
across **all** iterations.

### What to look for between iterations

- `01 → 02`: decode throughput jumps dramatically (no more full-seq recompute)
- `02 → 03`: adds real generation; measures time-to-first-token + decode tok/s
- `03 → 04–07`: each Triton swap shows isolated per-op gains
- `07 → 08`: throughput on mixed-length batches jumps 2-4×
- `08 → 09` (branch A): max concurrent requests goes from ~8 to ~64+ in same VRAM
- `08 → 10` (branch B): shared-prefix workloads see 2-5× speedup; TTFT collapses on workload #8
- **Compare 09 vs 10 head-to-head** on workloads #6, #7, #8 — paged dominates max-concurrency, radix dominates shared-prefix
- `09 → 11`: p95/p99 tail latency drops 30-50% under load
- `11 → 12`: small-batch decode latency drops 1.5-2× from killing launch overhead
- `12 → 13`: decode tok/s jumps 1.5–2.5× via speculation (acceptance-rate-dependent)
- `13 → 14`: max concurrent requests + max model size jump (quantization)
- `14 → 15`: sustained-decode GPU SM utilization climbs ~60 % → ~95 %
- `15 → 16`: TTFT under mixed prefill/decode load improves (disaggregation)
- `16 → 17–19`: same engine code, model size sweep — performance characteristics shift from latency-bound (3B) to weight-memory-bound (32B)
- `19 → 20`: engine logic unchanged, now driven inside a Ray actor — measure actor/serialization overhead (target < a few %); the serving refactor should be ~free
- `20 → 21`: add the Ray Serve HTTP/SSE ingress — measure API overhead and replica autoscaling under concurrent clients (serving metrics, not the five-workload engine bench)
- `21 → 22`: enables models that don't fit on one GPU; per-GPU decode latency typically *increases* — TP is a capacity tool
- `22 → 23–25`: each multi-GPU optimization claws back the latency TP costs

---

## What This Is NOT

- Not flash-attention from scratch beyond pedagogical level (Section 14d studies it; production use can call FlashAttention-3 / FlashInfer)
- Not multi-node distributed inference (single-node 2–4 GPU only)
- Not MoE / sparse models (expert parallelism only mentioned, not built)
- Not training, not fine-tuning, not LoRA serving
- Not multimodal (vision encoders, audio)
- Not structured-output decoding (JSON schema / XGrammar / Outlines)

Phases 1–3 give us the engine. Phase 4 generalizes it across dense models.
Phase 5 turns it into a production service with Ray. Phase 6 scales it across
GPUs. The goal — **phenomenal performance on dense models** — is met when the
Phase 6 engine, running a 30B-class dense
model under W4A16 + FP8-KV + speculative decoding + paged attention +
radix prefix caching + CUDA graphs + TP-2, beats a stock vLLM/SGLang
deployment on the same hardware on at least one workload class. That's the bar.

---

## Key Invariants at Every Step

1. **One class per section** — no monolithic files
2. **Shape checks** — print shapes before and after every op during development
3. **Dtype** — bfloat16 on GPU throughout; float32 only for CPU correctness checks
4. **Numerical match** — diff against `transformers`; bfloat16 tolerance ~1e-2, float32 ~1e-5
5. **Benchmark before moving on** — record to `results_baseline.json` so Phase 2 has a clean baseline
6. **Same five workloads in every `iterations/` file** — apples-to-apples comparison
