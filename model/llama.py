"""
LlamaModel — the full Llama 3.2-3B prefill forward pass.

Architecture recap
------------------
token_ids
  → TokenEmbedding          (vocab_size, hidden_size) lookup
  → 28 × TransformerBlock   each = RMSNorm + Attention + RMSNorm + MLP
  → RMSNorm                 final layer norm
  → OutputProjection        (hidden_size → vocab_size) logits, tied weights

The residual stream flows through all 28 blocks unchanged in shape:
  (B, T, 3072)  the entire time.

Weight manifest
---------------
  embed_tokens.weight              (128256, 3072)   ← TokenEmbedding
  layers.{0..27}.*                 9 tensors each   ← TransformerBlock
  norm.weight                      (3072,)          ← final RMSNorm
  lm_head.weight                   (128256, 3072)   ← tied to embed_tokens

Parameter count sanity check
-----------------------------
  Embed + head (tied):   128256 × 3072 × 2 bytes = 787 MB  (counted once)
  28 × attention:        28 × (9.44+3.14+3.14+9.44)M = 28 × 25.16M = 704.5M
  28 × MLP:              28 × (25.17+25.17+25.17)M   = 28 × 75.5M  = 2114.3M
  28 × 2 norms:          28 × 2 × 3072               = negligible
  Final norm:            3072                         = negligible
  Total:                 ~3.213B parameters
"""

import os

import torch
import torch.nn as nn

from config import ModelConfig
from loader import WeightLoader
from ops.embedding import TokenEmbedding, OutputProjection
from ops.rmsnorm import RMSNorm
from ops.rope import RopeFrequencies
from model.block import TransformerBlock

# torch.compile dispatch.
#   Controlled by the USE_COMPILE env var (set in .env or shell). Defaults to
#   False so the eager path stays the baseline; flip it on to wrap the forward
#   pass with torch.compile and let Inductor fuse ops + cut per-step kernel
#   launch overhead (the main driver of CPU utilization during decode).
#   COMPILE_MODE picks the torch.compile mode ("default", "reduce-overhead",
#   "max-autotune", ...). "default" maps to torch.compile's own default.
USE_COMPILE  = os.environ.get("USE_COMPILE", "false").lower() in ("1", "true", "yes", "on")
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")

# Custom CUDA graph dispatch.
#   Controlled by the USE_CUDA_GRAPHS env var (set in .env or shell). Defaults
#   to False. When True, the autoregressive DECODE step (T=1) is captured once
#   into a torch.cuda.CUDAGraph and replayed every token — collapsing the
#   hundreds of per-op kernel launches into a single replay, which is the main
#   driver of decode CPU overhead. Prefill stays eager (variable length).
#   This is mutually exclusive with USE_COMPILE for the decode path; if both are
#   set, CUDA graphs win for decode (the model is still compiled for prefill).
USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "false").lower() in ("1", "true", "yes", "on")


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        """
        Args:
            cfg:    ModelConfig with all hyperparameters
            device: target device (cuda)
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Precompute RoPE tables once — shared across all blocks
        rope_cfg = cfg.rope_scaling
        self.rope_freqs = RopeFrequencies(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
            rope_type=rope_cfg.rope_type,
            factor=rope_cfg.factor,
            low_freq_factor=rope_cfg.low_freq_factor,
            high_freq_factor=rope_cfg.high_freq_factor,
            original_max_seq_len=rope_cfg.original_max_position_embeddings,
            device=device,
        )

        # Token embedding table (owns the weight; OutputProjection will share it)
        self.embed = TokenEmbedding(cfg.vocab_size, cfg.hidden_size)

        # 28 transformer blocks — all share the same RopeFrequencies instance
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size, 
                num_heads_q=cfg.num_attention_heads,
                num_heads_kv=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                rope_freqs=self.rope_freqs,
                norm_eps=cfg.rms_norm_eps,
                layer_idx=i,
            )
            for i in range(cfg.num_hidden_layers)
        ])
        # Final layer norm (applied after all 28 blocks)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # Output projection — shares embed's weight tensor (tied embeddings)
        self.head = OutputProjection(self.embed)

        # Lazily-built CUDA-graph decode runner (see model/cuda_graph.py),
        # created on the first decode_step when USE_CUDA_GRAPHS is set.
        self._graph_decoder = None
        # When True, decode_step refuses to capture a new graph (used to assert a
        # profiled region only replays). Set via freeze_graph().
        self._graph_frozen = False

    def load_weights(self, loader: WeightLoader):
        """Load all weights from the checkpoint into this model."""
        dtype = self.cfg.model_dtype()

        # Embedding table
        self.embed.load_weight(loader.get("embed_tokens").to(dtype))

        # All 28 blocks
        for i, block in enumerate(self.layers):
            block.load_weights(
                attn_norm_weight=loader.get(f"layers.{i}.attn_norm").to(dtype),
                wq=loader.get(f"layers.{i}.attn.wq").to(dtype),
                wk=loader.get(f"layers.{i}.attn.wk").to(dtype),
                wv=loader.get(f"layers.{i}.attn.wv").to(dtype),
                wo=loader.get(f"layers.{i}.attn.wo").to(dtype),
                mlp_norm_weight=loader.get(f"layers.{i}.mlp_norm").to(dtype),
                w_gate=loader.get(f"layers.{i}.mlp.w_gate").to(dtype),
                w_up=loader.get(f"layers.{i}.mlp.w_up").to(dtype),
                w_down=loader.get(f"layers.{i}.mlp.w_down").to(dtype),
            )

        # Final norm
        self.norm.load_weight(loader.get("norm").to(dtype))

        # lm_head: in Llama 3.2 tied embeddings means lm_head == embed_tokens.
        # The OutputProjection already references embed.weight, so nothing to do.
        # But if the checkpoint has an explicit lm_head key that differs, load it:
        try:
            lm_head_w = loader.get("lm_head")
            # Only override if it's actually a different tensor (non-tied checkpoint)
            if not torch.equal(lm_head_w.cpu(), self.embed.weight.data.cpu()):
                with torch.no_grad():
                    self.embed.weight.copy_(lm_head_w.to(dtype))
        except (KeyError, Exception):
            pass  # tied — no separate lm_head weight

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache=None,
    ) -> torch.Tensor:
        """
        Full prefill forward pass.

        Args:
            token_ids: (B, T) integer token IDs
            start_pos: position offset for RoPE (0 for prefill, >0 for decode)
            kv_cache:  optional KVCache (Section 11) — None for now

        Returns:
            logits: (B, T, vocab_size)
        """
        # 1. Embed tokens → residual stream
        x = self.embed(token_ids)                      # (B, T, hidden_size)

        # 2. Pass through all 28 transformer blocks
        for layer in self.layers:
            x = layer(x, start_pos=start_pos, kv_cache=kv_cache)

        # 3. Final layer norm
        x = self.norm(x)                               # (B, T, hidden_size)

        # 4. Project to vocabulary logits
        logits = self.head(x)                          # (B, T, vocab_size)

        return logits

    def compile_model(
        self,
        mode: str | None = None,
        dynamic: bool | None = True,
        fullgraph: bool = False,
    ) -> "LlamaModel":
        """
        Wrap the forward pass with torch.compile (in place) to reduce CPU
        overhead from per-op kernel launches.

        Why this helps CPU utilization
        ------------------------------
        Each decode step launches hundreds of small CUDA kernels (one per op
        across 28 blocks). At decode the GPU is memory-bound and often idle
        between launches, so the wall-clock is dominated by the CPU dispatching
        those launches. torch.compile traces the forward into an FX graph and
        lets TorchInductor fuse adjacent ops into fewer, larger kernels — fewer
        launches means less CPU work per token.

        Args:
            mode:      torch.compile mode. None falls back to COMPILE_MODE from
                       the environment ("default" → torch.compile's own default;
                       "reduce-overhead", "max-autotune", ... otherwise).
            dynamic:   compile a single dynamic-shape graph. True here because
                       prefill length varies per prompt and the cached K/V slice
                       grows by one every decode step — without it Dynamo would
                       recompile on every new sequence length.
            fullgraph: require a single graph with no graph breaks. Left False
                       because the lazy Triton-kernel imports and the
                       env-var/`is_cuda` Python branches in ops/ legitimately
                       break the graph; breaks degrade but don't disable the win.

        Note:
            The biggest fusion win usually comes from letting Inductor compile
            the pure-PyTorch op path (USE_TRITON=false) into its own fused
            kernels. With USE_TRITON=true the hand-written Triton kernels are
            called as opaque ops, so expect more graph breaks around them.
        """
        if mode is None:
            mode = None if COMPILE_MODE in ("", "default") else COMPILE_MODE
        # nn.Module.compile() compiles self.forward in place (sets up the
        # compiled call impl), so `model(...)` transparently uses the compiled
        # path afterwards.
        self.compile(mode=mode, dynamic=dynamic, fullgraph=fullgraph)
        return self

    def maybe_compile(self) -> bool:
        """
        Compile the model iff USE_COMPILE is set in the environment.

        Returns:
            True if the model was compiled, False otherwise. Lets call sites
            print an accurate backend banner.
        """
        if USE_COMPILE:
            self.compile_model()
            return True
        return False

    def decode_step(
        self,
        token_ids: torch.Tensor,
        pos: int,
        kv_cache,
    ) -> torch.Tensor:
        """
        Single autoregressive decode step (T=1), routed through a captured CUDA
        graph when USE_CUDA_GRAPHS is set, otherwise a normal eager forward.

        Drop-in for `model(token_ids, start_pos=pos, kv_cache=kv_cache)` in the
        decode loop. Prefill must NOT use this — only fixed (B, 1) decode steps
        are graph-capturable.

        The graph is captured lazily on the first decode step (after prefill has
        populated the cache) and reused for all later steps and sequences of the
        same batch size; a batch-size change triggers a re-capture.

        Args:
            token_ids: (B, 1) or (B,) long tensor.
            pos:       absolute position of these tokens.
            kv_cache:  the KVCache populated by prefill.

        Returns:
            logits (B, 1, vocab).
        """
        token_ids = token_ids.view(token_ids.shape[0], 1)
        B = token_ids.shape[0]
        if not (USE_CUDA_GRAPHS and token_ids.is_cuda):
            return self(token_ids, start_pos=pos, kv_cache=kv_cache)

        needs_capture = self._graph_decoder is None or self._graph_decoder.B != B
        if needs_capture:
            # Graph capture is heavy, one-time work (buffer alloc + warmup +
            # record). When frozen (e.g. during a profiler's active region) we
            # refuse to capture here and fail loudly — capture must have been
            # done in warmup so the profiled region only ever replays.
            if self._graph_frozen:
                raise RuntimeError(
                    "decode graph not captured before a frozen region (batch "
                    f"size {B}). Call model.warmup_decode_graph(...) during "
                    "warmup, before profiling."
                )
            from model.cuda_graph import CUDAGraphDecoder
            self._graph_decoder = CUDAGraphDecoder(self, kv_cache, B)
            self._graph_decoder.capture(pos)
        return self._graph_decoder.decode(token_ids, pos)

    def warmup_decode_graph(self, kv_cache, batch_size: int, pos: int = 0) -> None:
        """
        Eagerly capture the decode CUDA graph for `batch_size` so later
        `decode_step` calls only replay. Call this during warmup — never inside
        a profiled/timed region — so the expensive capture is excluded.

        No-op unless USE_CUDA_GRAPHS is set. `pos` only seeds capture; the real
        position is written into static buffers on every replay, so any value in
        [0, max_seq_len) works.
        """
        if not USE_CUDA_GRAPHS:
            return
        if self._graph_decoder is None or self._graph_decoder.B != batch_size:
            from model.cuda_graph import CUDAGraphDecoder
            self._graph_decoder = CUDAGraphDecoder(self, kv_cache, batch_size)
            self._graph_decoder.capture(pos)

    def freeze_graph(self, frozen: bool = True) -> None:
        """
        Forbid (frozen=True) or re-allow new decode-graph capture in
        `decode_step`. Use to assert that a profiled region only replays an
        already-captured graph.
        """
        self._graph_frozen = frozen

    def reset_graph(self) -> None:
        """Drop any captured decode graph (e.g. when the KV cache is replaced)."""
        self._graph_decoder = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: torch.device | str = "cuda",
    ) -> "LlamaModel":
        """Convenience constructor: build + load weights in one call."""
        from config import ModelConfig
        from loader import WeightLoader

        device = torch.device(device)
        cfg = ModelConfig.llama_3_2_3b()
        loader = WeightLoader.from_pretrained(model_id)

        model = cls(cfg, device)
        model.load_weights(loader)
        model.to(device)
        model.eval()
        model.maybe_compile()
        return model
