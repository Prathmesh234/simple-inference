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

import torch
import torch.nn as nn

from config import ModelConfig
from loader import WeightLoader
from ops.embedding import TokenEmbedding, OutputProjection
from ops.rmsnorm import RMSNorm
from ops.rope import RopeFrequencies
from model.block import TransformerBlock


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
            )
            for _ in range(cfg.num_hidden_layers)
        ])
        # Final layer norm (applied after all 28 blocks)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # Output projection — shares embed's weight tensor (tied embeddings)
        self.head = OutputProjection(self.embed)

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
        return model
