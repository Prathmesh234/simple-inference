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

## Build Order Summary

```
Phase 1 — PyTorch baseline
  Section 1  ✅  config.py
  Section 2  ✅  loader.py
  Section 3  ✅  tokenizer.py
  Section 4      ops/rmsnorm.py
  Section 5      ops/embedding.py
  Section 6      ops/rope.py
  Section 7      ops/attention.py
  Section 8      ops/mlp.py
  Section 9      model/block.py
  Section 10     model/llama.py
  Section 11     model/kv_cache.py
  Section 12     sampling.py
  Section 13     generate.py + benchmarks/run_baseline.py

Phase 2 — Triton kernels (swap in one at a time)
  Section 14a    kernels/rmsnorm_kernel.py
  Section 14b    kernels/swiglu_kernel.py
  Section 14c    kernels/rope_kernel.py
  Section 14d    kernels/attention_kernel.py
  Section 14e    final benchmark comparison
```

---

## Key Invariants at Every Step

1. **One class per section** — no monolithic files
2. **Shape checks** — print shapes before and after every op during development
3. **Dtype** — bfloat16 on GPU throughout; float32 only for CPU correctness checks
4. **Numerical match** — diff against `transformers`; bfloat16 tolerance ~1e-2, float32 ~1e-5
5. **Benchmark before moving on** — record to `results_baseline.json` so Phase 2 has a clean baseline
6. **Never run files after writing** — hand the file to the user and let them run it; Claude writes, user executes

---

## What This Is NOT

- Not flash-attention (we study it, then optionally write it in Section 14d)
- Not continuous batching or PagedAttention
- Not quantization (INT4/AWQ/GPTQ)
- Not speculative decoding
- Not multi-GPU

All natural next steps once you understand what we build here.
