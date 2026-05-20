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

## Section 13 — Generation Loop + Full Benchmark Suite

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

### 14a — RMSNorm kernel  `kernels/rmsnorm_kernel.py`
- One Triton program per row
- Accumulate sum-of-squares in a loop over tiles, compute scale, write output in one pass
- Expected win: moderate — RMSNorm is memory-bound, kernel fusion eliminates one read+write round-trip

### 14b — SwiGLU fusion kernel  `kernels/swiglu_kernel.py`
- Fuse `silu(gate) * up` into one element-wise kernel
- Without fusion: two separate reads of the intermediate tensor; with fusion: one
- Expected win: meaningful — this is the most fusion-friendly op in the stack

### 14c — RoPE kernel  `kernels/rope_kernel.py`
- Apply cos/sin rotation to Q and K in a single fused kernel
- Expected win: small — RoPE is memory-bound but fast relative to matmuls

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

---

## Key Invariants at Every Step

1. **One class per section** — no monolithic files
2. **Shape checks** — print shapes before and after every op during development
3. **Dtype** — bfloat16 on GPU throughout; float32 only for CPU correctness checks
4. **Numerical match** — diff against `transformers`; bfloat16 tolerance ~1e-2, float32 ~1e-5
5. **Benchmark before moving on** — record to `results_baseline.json` so Phase 2 has a clean baseline
6. **Same five workloads in every `iterations/` file** — apples-to-apples comparison

---

## What This Is NOT

- Not flash-attention (we study it, then optionally write it in Section 14d)
- Not quantization (INT4/AWQ/GPTQ/FP8)
- Not speculative decoding (Medusa, EAGLE, draft+verify)
- Not multi-GPU (tensor / pipeline / sequence parallelism)
- Not structured outputs (JSON schema constrained decoding)
- Not disaggregated prefill/decode serving

All natural next steps once you understand Phases 1-3.
