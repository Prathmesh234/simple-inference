"""
ModelConfig: holds every hyperparameter for the model.

Loaded from a HuggingFace config.json so there are no magic numbers
anywhere else in the codebase.
"""

from __future__ import annotations

import env_loader
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class RopeScalingConfig:
    """Llama 3 uses a modified RoPE scaling scheme called 'llama3'."""
    rope_type: str                          # "llama3" for Llama 3.x
    factor: float                           # overall scale factor (32.0)
    low_freq_factor: float                  # frequencies below this get full scaling
    high_freq_factor: float                 # frequencies above this get no scaling
    original_max_position_embeddings: int   # context length the model was pretrained at

    @classmethod
    def from_dict(cls, d: dict) -> RopeScalingConfig:
        return cls(
            rope_type=d.get("rope_type", "default"),
            factor=float(d.get("factor", 1.0)),
            low_freq_factor=float(d.get("low_freq_factor", 1.0)),
            high_freq_factor=float(d.get("high_freq_factor", 4.0)),
            original_max_position_embeddings=int(d.get("original_max_position_embeddings", 8192)),
        )


@dataclass
class ModelConfig:
    # --- dimensions ---
    hidden_size: int            # residual stream width (3072)
    intermediate_size: int      # MLP hidden dim (8192)
    num_hidden_layers: int      # number of transformer blocks (28)
    vocab_size: int             # token vocabulary size (128256)

    # --- attention ---
    num_attention_heads: int    # Q heads (24)
    num_key_value_heads: int    # KV heads — < Q heads means GQA (8)

    # --- normalization ---
    rms_norm_eps: float         # epsilon inside RMSNorm (1e-5)

    # --- position encoding ---
    rope_theta: float           # RoPE base frequency (500000.0 for Llama 3)
    max_position_embeddings: int  # maximum supported sequence length (131072)
    rope_scaling: Optional[RopeScalingConfig] = None

    # --- token ids ---
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    # --- misc ---
    tie_word_embeddings: bool = True  # output projection reuses embedding weights
    torch_dtype: str = "bfloat16"

    # --- derived (not in config.json) ---
    # set in __post_init__ so they're always consistent
    head_dim: int = field(init=False)
    num_kv_groups: int = field(init=False)  # how many Q heads share each KV head

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by "
            f"num_attention_heads {self.num_attention_heads}"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads {self.num_attention_heads} must be divisible by "
            f"num_key_value_heads {self.num_key_value_heads}"
        )
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_config(cls, path: str | Path) -> ModelConfig:
        """Load from a HuggingFace config.json file."""
        with open(path) as f:
            d = json.load(f)

        rope_scaling = None
        if "rope_scaling" in d and d["rope_scaling"] is not None:
            rope_scaling = RopeScalingConfig.from_dict(d["rope_scaling"])

        return cls(
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            vocab_size=d["vocab_size"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=float(d["rope_theta"]),
            max_position_embeddings=d["max_position_embeddings"],
            rope_scaling=rope_scaling,
            bos_token_id=d.get("bos_token_id", 128000),
            eos_token_id=d.get("eos_token_id", 128001),
            tie_word_embeddings=d.get("tie_word_embeddings", True),
            torch_dtype=d.get("torch_dtype", "bfloat16"),
        )

    @classmethod
    def llama_3_2_3b(cls) -> ModelConfig:
        """Hardcoded config for Llama-3.2-3B — useful before you have the files."""
        return cls(
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=28,
            vocab_size=128256,
            num_attention_heads=24,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            max_position_embeddings=131072,
            rope_scaling=RopeScalingConfig(
                rope_type="llama3",
                factor=32.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=8192,
            ),
            bos_token_id=128000,
            eos_token_id=128001,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
        )

    def model_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.torch_dtype]

    def kv_cache_bytes(self, batch_size: int, seq_len: int) -> int:
        """Estimate KV cache memory in bytes for a given batch and sequence length."""
        # each layer stores K and V: [batch, seq_len, n_kv_heads, head_dim]
        elements_per_layer = 2 * batch_size * seq_len * self.num_key_value_heads * self.head_dim
        bytes_per_element = 2  # bfloat16
        return elements_per_layer * self.num_hidden_layers * bytes_per_element

    def print_summary(self):
        print("=" * 52)
        print("  ModelConfig")
        print("=" * 52)
        print(f"  Hidden size          : {self.hidden_size}")
        print(f"  Intermediate size    : {self.intermediate_size}")
        print(f"  Layers               : {self.num_hidden_layers}")
        print(f"  Vocab size           : {self.vocab_size:,}")
        print(f"  Q heads              : {self.num_attention_heads}")
        print(f"  KV heads             : {self.num_key_value_heads}  (GQA, {self.num_kv_groups} Q per KV)")
        print(f"  Head dim             : {self.head_dim}")
        print(f"  RoPE theta           : {self.rope_theta:,.0f}")
        print(f"  Max seq len          : {self.max_position_embeddings:,}")
        print(f"  Tied embeddings      : {self.tie_word_embeddings}")
        print(f"  dtype                : {self.torch_dtype}")

        kv_1k   = self.kv_cache_bytes(1, 1_024)   / 1e9
        kv_32k  = self.kv_cache_bytes(1, 32_768)  / 1e9
        kv_128k = self.kv_cache_bytes(1, 131_072) / 1e9
        print(f"  KV cache @ 1k tokens : {kv_1k:.2f} GB")
        print(f"  KV cache @ 32k tokens: {kv_32k:.2f} GB")
        print(f"  KV cache @ 128k tokens:{kv_128k:.2f} GB")
        print("=" * 52)


def verify_gpu():
    print("=" * 52)
    print("  GPU Check")
    print("=" * 52)
    if not torch.cuda.is_available():
        print("  [WARN] CUDA not available — running on CPU")
        return

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    free_mem, _ = torch.cuda.mem_get_info(device)
    free_mem /= 1e9

    print(f"  Device   : {name}")
    print(f"  Total    : {total_mem:.1f} GB")
    print(f"  Free     : {free_mem:.1f} GB")
    print(f"  Torch    : {torch.__version__}")

    try:
        import triton
        print(f"  Triton   : {triton.__version__}")
    except ImportError:
        print("  Triton   : not installed")

    print("=" * 52)


if __name__ == "__main__":
    verify_gpu()
    print()
    cfg = ModelConfig.llama_3_2_3b()
    cfg.print_summary()

    # sanity: load from file if it already exists
    hf_config = Path("weights/config.json")
    if hf_config.exists():
        print("\nLoading from HF config.json ...")
        cfg_from_file = ModelConfig.from_hf_config(hf_config)
        cfg_from_file.print_summary()
