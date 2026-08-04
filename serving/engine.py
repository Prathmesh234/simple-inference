"""
InferenceEngine — continuous-batching serving loop (Section 15).

What changed vs Section 13's generate()
---------------------------------------
generate() owned one request: prefill once, then loop decode until EOS. The
engine instead runs a *scheduler-driven* loop where the batch is recomposed
every iteration, so many requests of different lengths share decode steps and
short ones don't wait behind long ones.

Engine iteration flow
---------------------
`step()` intentionally reads like the serving pipeline:

    receive requests through add_request()
      -> schedule and batch waiting/running requests
      -> longest-prefix lookup during paged admission
      -> prefill only unmatched prompt suffixes
      -> keep the populated per-request KV block table
      -> optimized ragged decode (CUDA graph when available, eager fallback)

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

  - PREFILL  (P requests, unmatched prompt suffixes right-padded to Lmax):
    cached prefix pages stay attached to each request; only positions
    [cached_prefix_len, prompt_len) are computed and written. The last real
    suffix position produces the first generated token.
  - DECODE   (R requests, T = 1 each, but every request at a DIFFERENT absolute
    position): write each request's single new K/V at its own position into its
    own KV-cache slot, then attend over the fixed cache with each row's position
    supplied to the kernel. This "ragged" step is the heart of continuous batching.

Paged KV cache
--------------
The production path uses `PagedKVCache`: each request owns a logical block table
whose entries point into one shared physical K/V page pool. Scheduler slots only
limit admission; request IDs address paged cache state.

With `USE_TRITON=true`, decode always uses direct physical-page reads through
`attention_paged_decode_triton`. CUDA graphs capture that exact forward; turning
graphs off runs it eagerly. `USE_TRITON=false` retains gathered PyTorch SDPA as
the correctness fallback.

Optional CUDA graphs for decode (Section 19)
--------------------------------------------
When `use_cuda_graphs` is set, the *decode* step is captured into per-batch-size
CUDA graphs and replayed, collapsing the ~300 per-token kernel launches into one
replay. Capture needs fixed shapes, so the ragged batch is bucketed to preset
sizes (1, 2, 4, … max_running). Padding rows use a reserved scratch page while
fixed block-table, position, token, and RoPE buffers are updated before replay.
Prefill stays eager.

PagedAttention (Section 16)
---------------------------
Each request owns a block table rather than a fixed `max_seq_len` KV row.
`PagedDecoder` follows that table directly inside the Triton attention kernel,
both eagerly and under CUDA-graph replay. Page gather + SDPA is fallback-only.
"""

from __future__ import annotations

import math
import os

import torch
from torch.profiler import record_function

from model.llama import LlamaModel
from sampling import sample
from serving.paged_decoder import PagedDecoder
from serving.paged_kv_cache import PagedKVCache
from serving.prefill import PrefillRunner
from serving.radix_cache import RadixCache
from serving.request import Request, RequestState
from serving.scheduler import Scheduler

USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "false").lower() in ("1", "true", "yes", "on")
USE_TRITON_ATTENTION = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


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
        use_triton: bool | None = None,
        use_prefix_cache: bool = False,
        prefix_cache_blocks: int | None = None,
    ):
        """
        Args:
            model:        a loaded LlamaModel (eval, on device).
            max_running:  max concurrent requests admitted by the scheduler.
            max_seq_len:  maximum prompt + generated tokens for one request.
            block_size:   tokens per physical KV page in paged mode.
            num_kv_blocks: physical page-pool size. Defaults to
                          max_running * pages_per_max_length_request.
            token_budget: soft cap on Σ context tokens admitted per step
                          (defaults to max_running * max_seq_len = no extra cap).
            eos_id:       stop token (defaults to model.cfg.eos_token_id).
            temperature/top_k/top_p: sampling knobs (temperature=0 → greedy).
            warmup:       run a dummy prefill+decode at construction to prime
                          kernels/allocator so the first real request is fast.
            use_cuda_graphs: capture the batched decode into per-batch-size CUDA
                          graphs and replay them (eager prefill unchanged).
                          Defaults to the USE_CUDA_GRAPHS env var.
            use_triton: use the direct paged Triton decode kernel. False keeps
                          gathered PyTorch SDPA as a correctness fallback.
            use_prefix_cache: reuse complete paged KV blocks through a radix
                          tree and prefill only the unmatched prompt suffix.
            prefix_cache_blocks: maximum KV pages pinned by the radix cache.
                          Defaults to the full pool; admission evicts cache
                          leaves whenever live requests need those pages.
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
        requested_graphs = (
            USE_CUDA_GRAPHS if use_cuda_graphs is None else use_cuda_graphs
        )
        requested_triton = (
            USE_TRITON_ATTENTION if use_triton is None else use_triton
        )
        self.use_triton = requested_triton and self.device.type == "cuda"
        self.use_cuda_graphs = (
            requested_graphs
            and self.use_triton
            and self.device.type == "cuda"
        )
        self.use_prefix_cache = use_prefix_cache
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if requested_graphs and not self.use_cuda_graphs:
            print(
                "[engine] CUDA graphs require CUDA + Triton; "
                "using gathered SDPA fallback."
            )

        self.n_heads_q = self.cfg.num_attention_heads
        self.n_heads_kv = self.cfg.num_key_value_heads
        self.head_dim = self.cfg.head_dim
        self.kv_groups = self.cfg.num_kv_groups

        budget = token_budget if token_budget is not None else max_running * max_seq_len
        self.scheduler = Scheduler(max_running=max_running, token_budget=budget)

        self.block_size = block_size
        default_blocks = max_running * math.ceil(max_seq_len / block_size)
        self.num_kv_blocks = (
            num_kv_blocks if num_kv_blocks is not None else default_blocks
        )
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

        if self.use_prefix_cache:
            cache_blocks = (
                prefix_cache_blocks
                if prefix_cache_blocks is not None
                else self.num_kv_blocks
            )
            self.prefix_cache = RadixCache(self.kv, max_blocks=cache_blocks)
        else:
            self.prefix_cache = None

        self.prefill = PrefillRunner(
            model,
            self.kv,
            self.prefix_cache,
        )

        self.decoder = PagedDecoder(
            model,
            self.kv,
            max_running=max_running,
            max_seq_len=max_seq_len,
            n_heads_q=self.n_heads_q,
            n_heads_kv=self.n_heads_kv,
            head_dim=self.head_dim,
            use_triton=self.use_triton,
            use_cuda_graphs=self.use_cuda_graphs,
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
        required_blocks = self.kv.blocks_for_tokens(
            len(prompt_tokens) + max_new_tokens
        )
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

    @property
    def cuda_graphs_active(self) -> bool:
        return self.decoder.use_cuda_graphs

    @property
    def decode_backend(self) -> str:
        return self.decoder.backend

    def reset(self) -> None:
        """Drop all queued/running requests and reclaim every slot (pristine state)."""
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.free_slots = list(range(self.max_running))
        self.kv.reset()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

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
        if self.decoder.use_cuda_graphs:
            try:
                self.decoder.capture_all()
            except Exception as e:  # noqa: BLE001 — capture is fragile; degrade safely
                print(
                    f"[engine] CUDA-graph capture failed ({type(e).__name__}: {e}); "
                    f"falling back to eager decode."
                )
                self.decoder.disable_cuda_graphs()
                self.use_cuda_graphs = False
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.reset()

    @torch.no_grad()
    def step(self) -> dict[int, int]:
        """
        Run one engine iteration.

        Lifecycle:
          1. Evict finished requests and admit new requests.
          2. During paged admission, look up the longest cached prefix and
             attach any matching physical KV pages.
          3. Prefill only each new request's unmatched prompt suffix.
          4. Decode one token for requests that were already in DECODE state.

        Returns a dict {request_id: new_token_id} for every request that emitted
        a token this iteration (prefilled requests emit their first token; decode
        requests emit their next).
        """
        prefill_reqs, decode_reqs = self._schedule_iteration()
        emitted: dict[int, int] = {}

        if prefill_reqs:
            with record_function("engine.prefill_batch"):
                tokens = self._prefill_unmatched_suffixes(prefill_reqs)
            emitted.update(
                (request.id, token)
                for request, token in zip(prefill_reqs, tokens)
            )

        if decode_reqs:
            with record_function("engine.decode_batch"):
                tokens = self._decode_batch(decode_reqs)
            emitted.update(
                (request.id, token)
                for request, token in zip(decode_reqs, tokens)
            )

        return emitted

    def _schedule_iteration(self) -> tuple[list[Request], list[Request]]:
        """
        Evict completed work, admit FCFS requests, and form this iteration's
        prefill and decode groups.

        Prefix lookup is part of paged admission because shared pages reduce the
        number of new blocks a request must reserve.
        """
        with record_function("engine.schedule"):
            evicted = self.scheduler.evict_finished()
            for req in evicted:
                self.kv.release_request(req.id)
            self.scheduler.admit(can_admit=self._reserve_paged_request)

        # Snapshot steady-state decoders BEFORE prefill flips newly-admitted
        # requests to DECODE — otherwise a just-prefilled request would also be
        # decoded this same step and emit two tokens at once.
        decode_reqs: list[Request] = []
        for request in self.scheduler.running:
            if request.state is RequestState.DECODE:
                if request.pos >= self.max_seq_len:
                    request.state = RequestState.FINISHED
                else:
                    decode_reqs.append(request)
        prefill_reqs = [
            request
            for request in self.scheduler.running
            if request.state is RequestState.PREFILL
        ]
        return prefill_reqs, decode_reqs

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

    # ── prefix lookup and paged admission ─────────────────────────────────

    def _reserve_paged_request(self, request: Request) -> bool:
        """
        Attach a cached prefix, if present, then reserve blocks for the
        unmatched prompt suffix and future decode tokens.
        """
        max_tokens = request.prompt_len + request.max_new_tokens
        shared_block_ids, cached_tokens = self._match_cached_prefix(request)
        required_new_blocks = (
            self.kv.blocks_for_tokens(max_tokens) - len(shared_block_ids)
        )

        if self.prefix_cache is not None:
            if not self.prefix_cache.evict_until_available(
                required_new_blocks,
                protected_block_ids=set(shared_block_ids),
            ):
                return False
        elif not self.kv.can_reserve(request.id, max_tokens):
            return False

        try:
            self.kv.reserve_request(
                request.id,
                max_tokens,
                shared_block_ids=shared_block_ids,
                cached_tokens=cached_tokens,
            )
        except MemoryError:
            return False
        request.cached_prefix_len = cached_tokens
        return True

    def _match_cached_prefix(self, request: Request) -> tuple[list[int], int]:
        """Return shared physical pages and the matched complete-token count."""
        if self.prefix_cache is None:
            return [], 0
        match = self.prefix_cache.match(request.prompt_tokens)
        return match.block_ids, match.token_count

    # ── prefill unmatched prompt suffixes ─────────────────────────────────

    def _prefill_unmatched_suffixes(self, reqs: list[Request]) -> list[int]:
        """
        Compute only the tokens not covered by prefix-cache pages.

        Cached prefix pages are already attached to each request's paged block
        table during admission. This stage right-pads the unmatched suffixes,
        writes their K/V into the remaining pages, and samples the first token.
        """
        tokens = self._sample(self.prefill.logits(reqs)).tolist()
        self.prefill.publish(reqs)
        self._finish_prefill(reqs, tokens)
        return tokens

    def _finish_prefill(self, reqs: list[Request], tokens: list[int]) -> None:
        """Hand populated KV state to decode and advance request lifecycle."""
        for request, token in zip(reqs, tokens):
            request.pos = request.prompt_len
            request.generated.append(token)
            if token == self.eos_id:
                request.eos_hit = True
            request.state = (
                RequestState.FINISHED
                if request.should_finish()
                else RequestState.DECODE
            )

    # ── decode (ragged batch) ─────────────────────────────────────────────

    def _decode_batch(self, reqs: list[Request]) -> list[int]:
        """
        Consume the KV state handed off by prefill and emit one token per
        steady-state request.
        """
        logits = self._decode_logits_optimized(reqs)
        tokens = self._sample(logits).tolist()
        self._finish_decode(reqs, tokens)
        return tokens

    def _decode_logits_optimized(self, reqs: list[Request]) -> torch.Tensor:
        """Run paged decode using the configured optimized/fallback backend."""
        positions = [request.pos for request in reqs]
        last_tokens = [request.last_token for request in reqs]
        try:
            return self.decoder.logits(
                [request.id for request in reqs],
                positions,
                last_tokens,
            )
        except Exception as e:  # noqa: BLE001 — graph fallback is intentional
            if not self.decoder.use_cuda_graphs:
                raise
            print(
                f"[engine] CUDA-graph decode failed ({type(e).__name__}: {e}); "
                "retrying the same direct paged Triton forward eagerly."
            )
            self.decoder.disable_cuda_graphs()
            self.use_cuda_graphs = False
            return self.decoder.logits(
                [request.id for request in reqs],
                positions,
                last_tokens,
            )

    def _finish_decode(self, reqs: list[Request], tokens: list[int]) -> None:
        """Advance positions and retire requests that reached EOS or their cap."""
        for request, token in zip(reqs, tokens):
            request.generated.append(token)
            request.pos += 1
            if token == self.eos_id:
                request.eos_hit = True
            if request.should_finish():
                request.state = RequestState.FINISHED

    # ── sampling ──────────────────────────────────────────────────────────

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        return sample(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
