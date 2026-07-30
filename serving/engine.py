"""
InferenceEngine — continuous-batching serving loop (Section 15).

What changed vs Section 13's generate()
---------------------------------------
generate() owned one request: prefill once, then loop decode until EOS. The
engine instead runs a *scheduler-driven* loop where the batch is recomposed
every iteration, so many requests of different lengths share decode steps and
short ones don't wait behind long ones.

Design choice: reuse, don't rewrite
-----------------------------------
Every position-agnostic module of the model is reused verbatim — `embed`, each
block's `attn_norm` / `mlp_norm` / `mlp`, the final `norm`, and `head`. The only
thing continuous batching actually changes is attention, because attention is
the one op that depends on absolute position and on per-request history. So the
engine implements just the attention plumbing itself (Q/K/V projections via
the block's own weights, RoPE, KV read/write, and serving-aware masks/positions)
and leaves the entire rest of the model untouched.

The two batched forward shapes
------------------------------
Both phases run over a BATCH of requests — there is no single-request path, just
like vLLM. Each iteration runs a batched prefill of any newly admitted requests
followed by one batched decode of the steady-state requests.

  - PREFILL  (P requests, prompts right-padded to Lmax = max prompt_len): write
    each request's K/V for positions [0, prompt_len) into its slot, run causal
    self-attention per row, take each request's LAST REAL position logits → its
    first token. Padding queries are discarded; the per-row causal mask keeps
    padding keys unreachable.
  - DECODE   (R requests, T = 1 each, but every request at a DIFFERENT absolute
    position): write each request's single new K/V at its own position into its
    own KV-cache slot, then attend over the fixed cache with each row's position
    supplied to the kernel. This "ragged" step is the heart of continuous batching.

KV cache as a slot pool
-----------------------
We reuse the Section-11 KVCache, but address it by SLOT (one row per running
request) instead of treating the batch dim as "the current sequence". A request
holds slot s for its lifetime; row s of the cache is its private history. When
the request finishes, the scheduler hands slot s to the next waiting request.

With `USE_TRITON=true`, prefill uses the custom variable-length FlashAttention
kernel and decode uses the fixed-cache custom kernel with one position per row.
`USE_TRITON=false` retains PyTorch SDPA as the correctness fallback.

Optional CUDA graphs for decode (Section 19)
--------------------------------------------
When `use_cuda_graphs` is set, the *decode* step is captured into per-batch-size
CUDA graphs and replayed, collapsing the ~300 per-token kernel launches into one
replay. Capture needs fixed shapes, so the ragged batch is bucketed to preset
sizes (1, 2, 4, … max_running). Contiguous mode pads with a scratch KV row;
paged mode pads with a scratch physical page and updates fixed-size block-table,
position, token, and RoPE buffers in place before replay. Prefill stays eager.

PagedAttention (Section 16)
---------------------------
Each request owns a block table rather than a fixed `max_seq_len` KV row. Eager
decode gathers pages into a dense temporary before SDPA, prioritizing inspectable
correctness. CUDA-graph decode instead uses `PagedGraphDecoder` and a Triton
kernel that follows a fixed GPU block-table buffer directly into physical pages,
so no gather or dynamic K/V shape appears inside the captured graph.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from model.cuda_graph import BatchedGraphDecoder
from model.kv_cache import KVCache
from model.llama import LlamaModel
from ops.rope import rotate_half
from sampling import sample
from serving.paged_cuda_graph import PagedGraphDecoder
from serving.paged_kv_cache import PagedKVCache, paged_attention_forward
from serving.request import Request, RequestState
from serving.scheduler import Scheduler

# Capture the batched decode step into per-batch-size CUDA graphs and replay
# them, collapsing the ~300 per-token kernel launches into one replay (the main
# driver of memory-bound decode's CPU overhead). Prefill stays eager. Defaults
# off; toggle with the same USE_CUDA_GRAPHS env var the single-stream path uses.
USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "false").lower() in ("1", "true", "yes", "on")
USE_TRITON_ATTENTION = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE on x (B, T, H, D). cos/sin broadcastable to (B, T, 1, D)."""
    return x * cos + rotate_half(x) * sin


class InferenceEngine:
    def __init__(
        self,
        model: LlamaModel,
        max_running: int,
        max_seq_len: int,
        block_size: int = 16,
        num_kv_blocks: int | None = None,
        token_budget: int | None = None,
        eos_id: int | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        warmup: bool = True,
        use_cuda_graphs: bool | None = None,
        use_paged_attention: bool | None = None,
    ):
        """
        Args:
            model:        a loaded LlamaModel (eval, on device).
            max_running:  max concurrent requests admitted by the scheduler.
            max_seq_len:  maximum prompt + generated tokens for one request.
            block_size:   tokens per physical KV page in paged mode.
            num_kv_blocks: physical page-pool size. Defaults to the old
                          contiguous cache capacity, leaving callers free to
                          raise max_running independently.
            token_budget: soft cap on Σ context tokens admitted per step
                          (defaults to max_running * max_seq_len = no extra cap).
            eos_id:       stop token (defaults to model.cfg.eos_token_id).
            temperature/top_k/top_p: sampling knobs (temperature=0 → greedy).
            warmup:       run a dummy prefill+decode at construction to prime
                          kernels/allocator so the first real request is fast.
            use_cuda_graphs: capture the batched decode into per-batch-size CUDA
                          graphs and replay them (eager prefill unchanged).
                          Defaults to the USE_CUDA_GRAPHS env var.
            use_paged_attention: use page tables. Eager decode gathers into
                          SDPA; CUDA-graph decode reads pages directly from a
                          fixed GPU block-table buffer. Defaults to True.
        """
        self.model = model
        self.cfg = model.cfg
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self.max_running = max_running
        self.max_seq_len = max_seq_len
        self.eos_id = eos_id if eos_id is not None else self.cfg.eos_token_id
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.use_cuda_graphs = USE_CUDA_GRAPHS if use_cuda_graphs is None else use_cuda_graphs
        self.use_paged_attention = (
            True if use_paged_attention is None else use_paged_attention
        )
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.n_heads_q = self.cfg.num_attention_heads
        self.n_heads_kv = self.cfg.num_key_value_heads
        self.head_dim = self.cfg.head_dim
        self.kv_groups = self.cfg.num_kv_groups

        budget = token_budget if token_budget is not None else max_running * max_seq_len
        self.scheduler = Scheduler(max_running=max_running, token_budget=budget)

        self.block_size = block_size
        if self.use_paged_attention:
            default_blocks = max_running * math.ceil(max_seq_len / block_size)
            self.num_kv_blocks = num_kv_blocks if num_kv_blocks is not None else default_blocks
            self.kv = PagedKVCache(
                n_layers=self.cfg.num_hidden_layers,
                num_blocks=self.num_kv_blocks,
                block_size=block_size,
                n_heads_kv=self.n_heads_kv,
                head_dim=self.head_dim,
                num_scratch_blocks=1 if self.use_cuda_graphs else 0,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.num_kv_blocks = None
            # One KV-cache row per slot. Reuses the Section-11 cache verbatim;
            # CUDA graphs reserve one extra scratch row for padding.
            self.kv = KVCache(
                n_layers=self.cfg.num_hidden_layers,
                max_batch=max_running + (1 if self.use_cuda_graphs else 0),
                max_seq_len=max_seq_len,
                n_heads_kv=self.n_heads_kv,
                head_dim=self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )

        # Lazily-captured per-batch-size decode graphs (captured on first use of
        # each bucket, or all at once in warmup()). None when graphs are off.
        # Takes primitives — not `self` — so model/ stays free of serving/ imports.
        if self.use_cuda_graphs and self.use_paged_attention:
            self.graph_decoder = PagedGraphDecoder(
                model,
                self.kv,
                max_running=max_running,
                max_seq_len=max_seq_len,
                n_heads_q=self.n_heads_q,
                n_heads_kv=self.n_heads_kv,
                head_dim=self.head_dim,
                kv_groups=self.kv_groups,
            )
        elif self.use_cuda_graphs:
            self.graph_decoder = BatchedGraphDecoder(
                model,
                self.kv,
                max_running=max_running,
                n_heads_q=self.n_heads_q,
                n_heads_kv=self.n_heads_kv,
                head_dim=self.head_dim,
                kv_groups=self.kv_groups,
            )
        else:
            self.graph_decoder = None

        if warmup:
            self.warmup()

    # ── public API ────────────────────────────────────────────────────────

    def add_request(self, prompt_tokens: list[int], max_new_tokens: int) -> Request:
        """Queue a new request. Returns the Request (read its .id / .generated)."""
        if not prompt_tokens:
            raise ValueError("prompt_tokens is empty")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1 (got {max_new_tokens})")
        # A request occupies positions [0, prompt_len + max_new_tokens) at most;
        # reject up front rather than overflow the KV cache mid-decode.
        if len(prompt_tokens) + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt_len ({len(prompt_tokens)}) + max_new_tokens ({max_new_tokens}) "
                f"> max_seq_len ({self.max_seq_len})"
            )
        if self.use_paged_attention:
            required_blocks = self.kv.blocks_for_tokens(len(prompt_tokens) + max_new_tokens)
            if required_blocks > self.kv.num_blocks:
                raise ValueError(
                    "request needs more KV blocks than the entire paged cache: "
                    f"{required_blocks} > {self.kv.num_blocks}"
                )
        vocab = self.cfg.vocab_size
        if any(t < 0 or t >= vocab for t in prompt_tokens):
            raise ValueError("prompt contains out-of-range token ids")
        req = Request(prompt_tokens=list(prompt_tokens), max_new_tokens=max_new_tokens)
        self.scheduler.add(req)
        return req

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def reset(self) -> None:
        """Drop all queued/running requests and reclaim every slot (pristine state)."""
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.free_slots = list(range(self.max_running))
        self.kv.reset()

    @torch.no_grad()
    def warmup(self, num_seqs: int = 2, prompt_len: int = 8, decode_steps: int = 4) -> None:
        """
        Exercise the exact prefill + ragged-decode path on dummy requests so the
        first real request doesn't eat one-time costs: Triton kernel
        compilation/autotuning (RMSNorm, MLP, attention) and CUDA
        caching-allocator growth. Engine state is fully reset afterwards.

        When CUDA graphs are enabled, every per-batch-size decode graph is also
        captured here (after the eager warmup primes/autotunes the kernels) so
        the serving loop only ever *replays* and never pays capture cost mid-
        flight. With graphs off, the same custom decode kernel runs eagerly;
        graph bucketing removes its Python/kernel-launch overhead.
        """
        num_seqs = max(1, min(num_seqs, self.max_running))
        prompt_len = max(1, min(prompt_len, self.max_seq_len - decode_steps - 1))
        dummy = [self.cfg.bos_token_id] + [(i % 1024) for i in range(prompt_len - 1)]
        for _ in range(num_seqs):
            self.add_request(list(dummy), max_new_tokens=decode_steps + 1)
        max_iters = prompt_len + decode_steps + num_seqs + 8
        iters = 0
        while self.has_work() and iters < max_iters:
            self.step()
            iters += 1
        if self.graph_decoder is not None:
            try:
                self.graph_decoder.capture_all()
            except Exception as e:  # noqa: BLE001 — capture is fragile; degrade safely
                # CUDA-graph capture is driver/arch/memory-dependent and can fail
                # (e.g. capture-unsafe op, OOM on the static buffers). A failure
                # here must NOT kill the engine: drop the graph decoder so every
                # decode runs the eager custom-kernel path instead.
                print(
                    f"[engine] CUDA-graph capture failed ({type(e).__name__}: {e}); "
                    f"falling back to eager decode."
                )
                self.graph_decoder = None
                self.use_cuda_graphs = False
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.reset()

    @torch.no_grad()
    def step(self) -> dict[int, int]:
        """
        Run one engine iteration.

        Returns a dict {request_id: new_token_id} for every request that emitted
        a token this iteration (prefilled requests emit their first token; decode
        requests emit their next).
        """
        with record_function("engine.schedule"):
            # Release page capacity before FCFS admission so a completed
            # request can make room for the next waiting request immediately.
            evicted = self.scheduler.evict_finished()
            if self.use_paged_attention:
                for req in evicted:
                    self.kv.release_request(req.id)
                admitted = self.scheduler.admit(can_admit=self._can_reserve_pages)
                for req in admitted:
                    self.kv.reserve_request(req.id, req.prompt_len + req.max_new_tokens)
            else:
                self.scheduler.admit()

        emitted: dict[int, int] = {}

        # Snapshot steady-state decoders BEFORE prefill flips newly-admitted
        # requests to DECODE — otherwise a just-prefilled request would also be
        # decoded this same step and emit two tokens at once. Also defensively
        # retire any request that would write past the KV cache (should never
        # happen given add_request's check, but never index out of bounds).
        decode_reqs: list[Request] = []
        for r in self.scheduler.running:
            if r.state is RequestState.DECODE:
                if r.pos >= self.max_seq_len:
                    r.state = RequestState.FINISHED
                else:
                    decode_reqs.append(r)

        # 1. Batched prefill of all newly admitted requests (variable prompt
        #    lengths are padded to the batch max; padding queries are discarded).
        prefill_reqs = [r for r in self.scheduler.running if r.state is RequestState.PREFILL]
        if prefill_reqs:
            with record_function("engine.prefill_batch"):
                toks = self._prefill_batch(prefill_reqs)
            for req, tok in zip(prefill_reqs, toks):
                emitted[req.id] = tok

        # 2. One ragged decode over every request already in steady state.
        if decode_reqs:
            with record_function("engine.decode_batch"):
                toks = self._decode_batch(decode_reqs)
            for req, tok in zip(decode_reqs, toks):
                emitted[req.id] = tok

        return emitted

    def run(self) -> dict[int, Request]:
        """Drive steps until all queued requests finish; return them by id."""
        seen: dict[int, Request] = {}
        for r in list(self.scheduler.waiting):
            seen[r.id] = r
        while self.has_work():
            for r in self.scheduler.running:
                seen[r.id] = r
            self.step()
        return seen

    # ── prefill (batched, padded) ─────────────────────────────────────────

    def _prefill_batch(self, reqs: list[Request]) -> list[int]:
        """
        Prefill a batch of requests in ONE forward pass (vLLM-style — there is no
        single-request path). Prompts of different lengths are right-padded to the
        batch max `Lmax`; every row attends only over its OWN slot with a causal
        mask, so the per-row causal structure makes padding keys unreachable to
        real query positions and padding-query outputs are simply discarded.

        Each request's first token is sampled from its LAST REAL prompt position.
        """
        R = len(reqs)
        lens = [r.prompt_len for r in reqs]
        Lmax = max(lens)
        device = self.device

        # Right-padded prompt ids (pad value 0 — its projections are harmless,
        # they only ever land in masked/discarded positions).
        ids = torch.zeros(R, Lmax, dtype=torch.long, device=device)
        for i, r in enumerate(reqs):
            ids[i, : lens[i]] = torch.tensor(r.prompt_tokens, dtype=torch.long, device=device)

        slots = torch.tensor([r.slot for r in reqs], dtype=torch.long, device=device)
        h = self.model.embed(ids)  # (R, Lmax, hidden)
        cos = self.model.rope_freqs.cos[:Lmax].to(self.dtype).view(1, Lmax, 1, self.head_dim)
        sin = self.model.rope_freqs.sin[:Lmax].to(self.dtype).view(1, Lmax, 1, self.head_dim)

        for layer in self.model.layers:
            h = self._attn_prefill(h, layer, reqs, lens, slots, Lmax, cos, sin, R)
            h = h + layer.mlp(layer.mlp_norm(h))

        # Gather each request's last real position, then norm+head only there.
        last_idx = torch.tensor([l - 1 for l in lens], dtype=torch.long, device=device)
        h_last = h[torch.arange(R, device=device), last_idx]   # (R, hidden)
        h_last = self.model.norm(h_last.unsqueeze(1))          # (R, 1, hidden)
        logits = self.model.head(h_last)[:, -1, :]             # (R, vocab)
        toks = self._sample(logits).tolist()

        for req, tok in zip(reqs, toks):
            req.pos = req.prompt_len                           # next write goes here
            req.generated.append(tok)
            if tok == self.eos_id:
                req.eos_hit = True
            req.state = RequestState.FINISHED if req.should_finish() else RequestState.DECODE
        return toks

    def _attn_prefill(self, h, layer, reqs, lens, slots, Lmax, cos, sin, R) -> torch.Tensor:
        attn = layer.attn
        li = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(R, Lmax, self.n_heads_q, self.head_dim)
        k = F.linear(x, attn.wk).view(R, Lmax, self.n_heads_kv, self.head_dim)
        v = F.linear(x, attn.wv).view(R, Lmax, self.n_heads_kv, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        qT = q.transpose(1, 2)                                 # (R, Hq, Lmax, D)
        if self.use_paged_attention:
            # Padding never gets a page-table entry: each request writes only
            # its real prompt positions, then gathers its own logical prefix.
            for row, req in enumerate(reqs):
                real_length = lens[row]
                self.kv.append(
                    li,
                    req.id,
                    0,
                    k[row, :real_length].transpose(0, 1).contiguous(),
                    v[row, :real_length].transpose(0, 1).contiguous(),
                )
            K, V = self.kv.gather_batch(
                li, [req.id for req in reqs], lens, pad_to=Lmax
            )
            query_positions = torch.arange(Lmax, device=self.device).view(1, Lmax).expand(R, -1)
            key_lengths = torch.tensor(lens, dtype=torch.long, device=self.device)
            out = paged_attention_forward(qT, K, V, query_positions, key_lengths, self.kv_groups)
            out = out.transpose(1, 2)
        else:
            # Write each row's K/V into its slot at columns [0, Lmax).
            self.kv.k_cache[li][slots, :, :Lmax, :] = k.permute(0, 2, 1, 3)
            self.kv.v_cache[li][slots, :, :Lmax, :] = v.permute(0, 2, 1, 3)
            K = self.kv.k_cache[li][slots, :, :Lmax, :]
            V = self.kv.v_cache[li][slots, :, :Lmax, :]
            if USE_TRITON_ATTENTION and qT.is_cuda:
                from kernels.attention_kernel import attention_prefill_triton
                out = attention_prefill_triton(
                    qT,
                    K,
                    V,
                    causal=True,
                    return_transposed=True,
                )
            else:
                out = F.scaled_dot_product_attention(
                    qT, K, V, is_causal=True, enable_gqa=(self.kv_groups > 1)
                )
                out = out.transpose(1, 2)
        out = out.reshape(R, Lmax, self.n_heads_q * self.head_dim)
        return h + F.linear(out, attn.wo)

    # ── decode (ragged batch) ─────────────────────────────────────────────

    def _decode_batch(self, reqs: list[Request]) -> list[int]:
        """
        One decode step over the steady-state requests. Computes next-token
        logits either by replaying a captured CUDA graph (when enabled and the
        batch fits a bucket) or by launching the same custom decode kernel
        eagerly, then samples and advances each request.
        """
        logits = None
        if self.graph_decoder is not None:
            positions = [r.pos for r in reqs]
            last_tokens = [r.last_token for r in reqs]
            try:
                # None when the batch is larger than the biggest captured bucket.
                if self.use_paged_attention:
                    logits = self.graph_decoder.logits(
                        [r.id for r in reqs], positions, last_tokens
                    )
                else:
                    logits = self.graph_decoder.logits(
                        [r.slot for r in reqs], positions, last_tokens
                    )
            except Exception as e:  # noqa: BLE001 — never crash a live request
                # A lazy capture (warmup=False) or replay can still fail at
                # runtime; permanently drop to eager rather than drop the request.
                print(
                    f"[engine] CUDA-graph decode failed ({type(e).__name__}: {e}); "
                    f"falling back to eager decode."
                )
                self.graph_decoder = None
                self.use_cuda_graphs = False
                logits = None
        if logits is None:
            logits = self._decode_logits_eager(reqs)

        toks = self._sample(logits).tolist()
        for req, tok in zip(reqs, toks):
            req.generated.append(tok)
            req.pos += 1
            if tok == self.eos_id:
                req.eos_hit = True
            if req.should_finish():
                req.state = RequestState.FINISHED
        return toks

    def _decode_logits_eager(self, reqs: list[Request]) -> torch.Tensor:
        """Eager ragged decode using the same per-row-position kernel as graphs."""
        R = len(reqs)
        slots = torch.tensor([r.slot for r in reqs], dtype=torch.long, device=self.device)
        positions = torch.tensor([r.pos for r in reqs], dtype=torch.long, device=self.device)
        last = torch.tensor([[r.last_token] for r in reqs], dtype=torch.long, device=self.device)

        h = self.model.embed(last)                   # (R, 1, hidden)
        cos = self.model.rope_freqs.cos[positions].to(self.dtype).view(R, 1, 1, self.head_dim)
        sin = self.model.rope_freqs.sin[positions].to(self.dtype).view(R, 1, 1, self.head_dim)

        mask = None
        Lmax = self.max_seq_len
        if self.use_paged_attention:
            # The gathered paged path creates its own per-row causal mask.
            pass
        elif not (USE_TRITON_ATTENTION and h.is_cuda):
            # PyTorch fallback reads only the longest valid prefix and needs an
            # explicit per-row length mask. The Triton path reads the fixed full
            # cache and applies positions inside the kernel, matching graph mode.
            Lmax = int(positions.max().item()) + 1
            cols = torch.arange(Lmax, device=self.device)
            allowed = cols[None, :] <= positions[:, None]
            mask = torch.zeros(R, 1, 1, Lmax, dtype=self.dtype, device=self.device)
            mask.masked_fill_(~allowed.view(R, 1, 1, Lmax), float("-inf"))

        for layer in self.model.layers:
            h = self._attn_decode(h, layer, reqs, slots, positions, Lmax, cos, sin, mask, R)
            h = h + layer.mlp(layer.mlp_norm(h))

        h = self.model.norm(h)
        return self.model.head(h)[:, -1, :]          # (R, vocab)

    def _attn_decode(self, h, layer, reqs, slots, positions, Lmax, cos, sin, mask, R) -> torch.Tensor:
        attn = layer.attn
        li = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(R, 1, self.n_heads_q, self.head_dim)
        k = F.linear(x, attn.wk).view(R, 1, self.n_heads_kv, self.head_dim)
        v = F.linear(x, attn.wv).view(R, 1, self.n_heads_kv, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        qT = q.transpose(1, 2)                                   # (R, Hq, 1, D)
        if self.use_paged_attention:
            lengths = [req.pos + 1 for req in reqs]
            for row, req in enumerate(reqs):
                self.kv.append(
                    li,
                    req.id,
                    req.pos,
                    k[row, 0].unsqueeze(1).contiguous(),
                    v[row, 0].unsqueeze(1).contiguous(),
                )
            K, V = self.kv.gather_batch(li, [req.id for req in reqs], lengths)
            out = paged_attention_forward(
                qT,
                K,
                V,
                positions.view(R, 1),
                torch.tensor(lengths, dtype=torch.long, device=self.device),
                self.kv_groups,
            )
        else:
            # Scatter each row's single new K/V into (slot_r, position_r).
            self.kv.k_cache[li][slots, :, positions, :] = k[:, 0]
            self.kv.v_cache[li][slots, :, positions, :] = v[:, 0]
            K = self.kv.k_cache[li][slots, :, :Lmax, :]
            V = self.kv.v_cache[li][slots, :, :Lmax, :]
            if USE_TRITON_ATTENTION and qT.is_cuda:
                from kernels.attention_kernel import attention_decode_triton
                out = attention_decode_triton(qT, K, V, positions)
            else:
                out = F.scaled_dot_product_attention(
                    qT, K, V, attn_mask=mask, enable_gqa=(self.kv_groups > 1)
                )
        out = out.transpose(1, 2).reshape(R, 1, self.n_heads_q * self.head_dim)
        return h + F.linear(out, attn.wo)

    # ── sampling ──────────────────────────────────────────────────────────

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        return sample(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

    def _can_reserve_pages(self, req: Request) -> bool:
        """Admission callback: request capacity must fit the paged pool."""
        return self.kv.can_reserve(req.id, req.prompt_len + req.max_new_tokens)
