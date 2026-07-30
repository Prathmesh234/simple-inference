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

PagedAttention (Section 16)
---------------------------
Each request owns a block table rather than a fixed `max_seq_len` KV row. Eager
prefill and decode gather pages into dense temporaries before SDPA, prioritizing
inspectable correctness over launch and copy efficiency.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from model.llama import LlamaModel
from ops.rope import rotate_half
from sampling import sample
from .paged_kv_cache import PagedKVCache, paged_attention_forward
from .request import Request, RequestState
from .scheduler import Scheduler

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
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.n_heads_q = self.cfg.num_attention_heads
        self.n_heads_kv = self.cfg.num_key_value_heads
        self.head_dim = self.cfg.head_dim
        self.kv_groups = self.cfg.num_kv_groups

        budget = token_budget if token_budget is not None else max_running * max_seq_len
        self.scheduler = Scheduler(max_running=max_running, token_budget=budget)

        self.block_size = block_size
        default_blocks = max_running * math.ceil(max_seq_len / block_size)
        self.num_kv_blocks = num_kv_blocks if num_kv_blocks is not None else default_blocks
        self.kv = PagedKVCache(
            n_layers=self.cfg.num_hidden_layers,
            num_blocks=self.num_kv_blocks,
            block_size=block_size,
            n_heads_kv=self.n_heads_kv,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

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

        This milestone deliberately keeps decode eager: pages are gathered into
        a dense temporary and passed to SDPA.
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
            for req in evicted:
                self.kv.release_request(req.id)
            admitted = self.scheduler.admit(can_admit=self._can_reserve_pages)
            for req in admitted:
                self.kv.reserve_request(req.id, req.prompt_len + req.max_new_tokens)

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
        batch max `Lmax`; every row attends only over its own gathered pages with
        a causal mask, so padding keys are unreachable to real query positions
        and padding-query outputs are simply discarded.

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

        h = self.model.embed(ids)  # (R, Lmax, hidden)
        cos = self.model.rope_freqs.cos[:Lmax].to(self.dtype).view(1, Lmax, 1, self.head_dim)
        sin = self.model.rope_freqs.sin[:Lmax].to(self.dtype).view(1, Lmax, 1, self.head_dim)

        for layer in self.model.layers:
            h = self._attn_prefill(h, layer, reqs, lens, Lmax, cos, sin, R)
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

    def _attn_prefill(self, h, layer, reqs, lens, Lmax, cos, sin, R) -> torch.Tensor:
        attn = layer.attn
        li = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(R, Lmax, self.n_heads_q, self.head_dim)
        k = F.linear(x, attn.wk).view(R, Lmax, self.n_heads_kv, self.head_dim)
        v = F.linear(x, attn.wv).view(R, Lmax, self.n_heads_kv, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        qT = q.transpose(1, 2)                                 # (R, Hq, Lmax, D)
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
        out = out.reshape(R, Lmax, self.n_heads_q * self.head_dim)
        return h + F.linear(out, attn.wo)

    # ── decode (ragged batch) ─────────────────────────────────────────────

    def _decode_batch(self, reqs: list[Request]) -> list[int]:
        """
        One eager decode step over the steady-state requests, followed by
        sampling and request-state updates.
        """
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
        positions = torch.tensor([r.pos for r in reqs], dtype=torch.long, device=self.device)
        last = torch.tensor([[r.last_token] for r in reqs], dtype=torch.long, device=self.device)

        h = self.model.embed(last)                   # (R, 1, hidden)
        cos = self.model.rope_freqs.cos[positions].to(self.dtype).view(R, 1, 1, self.head_dim)
        sin = self.model.rope_freqs.sin[positions].to(self.dtype).view(R, 1, 1, self.head_dim)

        for layer in self.model.layers:
            h = self._attn_decode(h, layer, reqs, positions, cos, sin, R)
            h = h + layer.mlp(layer.mlp_norm(h))

        h = self.model.norm(h)
        return self.model.head(h)[:, -1, :]          # (R, vocab)

    def _attn_decode(self, h, layer, reqs, positions, cos, sin, R) -> torch.Tensor:
        attn = layer.attn
        li = attn.layer_idx
        x = layer.attn_norm(h)
        q = F.linear(x, attn.wq).view(R, 1, self.n_heads_q, self.head_dim)
        k = F.linear(x, attn.wk).view(R, 1, self.n_heads_kv, self.head_dim)
        v = F.linear(x, attn.wv).view(R, 1, self.n_heads_kv, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        qT = q.transpose(1, 2)                                   # (R, Hq, 1, D)
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
