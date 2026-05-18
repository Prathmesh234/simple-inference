"""
WeightLoader: download and lazily load Llama safetensors shards.

How HuggingFace stores large models
------------------------------------
Large models are split across multiple .safetensors shard files.
An index file called model.safetensors.index.json sits next to them
and maps every tensor name to the shard that contains it:

    {
      "metadata": { "total_size": 6435651584 },
      "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        ...
      }
    }

The safetensors format itself stores a JSON header at the top of each file
with each tensor's dtype, shape, and byte offsets. This means we can read
shapes and dtypes without loading the actual weight data into RAM.

Lazy loading: we open a shard file handle the first time any tensor from
that shard is requested, and keep the handle open. Tensor data is only
read from disk when .get() is called.

HuggingFace name → our name mapping
--------------------------------------
HF uses verbose names like model.layers.0.self_attn.q_proj.weight.
We map these to shorter names used throughout our own code:

    model.layers.{i}.self_attn.q_proj.weight  → layers.{i}.attn.wq
    model.layers.{i}.self_attn.k_proj.weight  → layers.{i}.attn.wk
    model.layers.{i}.self_attn.v_proj.weight  → layers.{i}.attn.wv
    model.layers.{i}.self_attn.o_proj.weight  → layers.{i}.attn.wo
    model.layers.{i}.mlp.gate_proj.weight     → layers.{i}.mlp.w_gate
    model.layers.{i}.mlp.up_proj.weight       → layers.{i}.mlp.w_up
    model.layers.{i}.mlp.down_proj.weight     → layers.{i}.mlp.w_down
    model.layers.{i}.input_layernorm.weight   → layers.{i}.attn_norm
    model.layers.{i}.post_attention_layernorm.weight → layers.{i}.mlp_norm
    model.embed_tokens.weight                 → embed_tokens
    model.norm.weight                         → norm
    lm_head.weight                            → lm_head
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

def hf_to_ours(name: str) -> str:
    """
    Translate a HuggingFace tensor name to our internal naming convention.
    Returns the original name unchanged if no rule matches.
    """
    # per-layer rules
    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if m:
        layer_idx = m.group(1)
        rest = m.group(2)
        _layer_map = {
            "self_attn.q_proj.weight":           "attn.wq",
            "self_attn.k_proj.weight":           "attn.wk",
            "self_attn.v_proj.weight":           "attn.wv",
            "self_attn.o_proj.weight":           "attn.wo",
            "mlp.gate_proj.weight":              "mlp.w_gate",
            "mlp.up_proj.weight":                "mlp.w_up",
            "mlp.down_proj.weight":              "mlp.w_down",
            "input_layernorm.weight":            "attn_norm",
            "post_attention_layernorm.weight":   "mlp_norm",
        }
        if rest in _layer_map:
            return f"layers.{layer_idx}.{_layer_map[rest]}"

    # top-level rules
    _top_map = {
        "model.embed_tokens.weight": "embed_tokens",
        "model.norm.weight":         "norm",
        "lm_head.weight":            "lm_head",
    }
    return _top_map.get(name, name)


def ours_to_hf(name: str) -> str:
    """Reverse mapping: our name → HuggingFace name."""
    # per-layer
    m = re.match(r"layers\.(\d+)\.(.*)", name)
    if m:
        layer_idx = m.group(1)
        rest = m.group(2)
        _layer_map = {
            "attn.wq":   "self_attn.q_proj.weight",
            "attn.wk":   "self_attn.k_proj.weight",
            "attn.wv":   "self_attn.v_proj.weight",
            "attn.wo":   "self_attn.o_proj.weight",
            "mlp.w_gate":"mlp.gate_proj.weight",
            "mlp.w_up":  "mlp.up_proj.weight",
            "mlp.w_down":"mlp.down_proj.weight",
            "attn_norm": "input_layernorm.weight",
            "mlp_norm":  "post_attention_layernorm.weight",
        }
        if rest in _layer_map:
            return f"model.layers.{layer_idx}.{_layer_map[rest]}"

    _top_map = {
        "embed_tokens": "model.embed_tokens.weight",
        "norm":         "model.norm.weight",
        "lm_head":      "lm_head.weight",
    }
    return _top_map.get(name, name)


# ---------------------------------------------------------------------------
# WeightLoader
# ---------------------------------------------------------------------------

class WeightLoader:
    """
    Lazily load weights from HuggingFace safetensors shards.

    Usage
    -----
        loader = WeightLoader.from_pretrained("meta-llama/Llama-3.2-3B")
        q_weight = loader.get("layers.0.attn.wq")   # our name
        # or
        q_weight = loader.get_hf("model.layers.0.self_attn.q_proj.weight")
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        # hf_name -> shard filename
        self._tensor_to_shard: Dict[str, str] = {}
        # shard filename -> open safe_open handle
        self._open_shards: Dict[str, safe_open] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ) -> WeightLoader:
        """
        Download (or find cached) a HuggingFace model and return a loader.

        On first call this downloads all safetensors shards — for Llama-3.2-3B
        that is ~6 GB. Subsequent calls are instant because huggingface_hub
        caches to ~/.cache/huggingface/hub/.
        """
        print(f"Locating model: {repo_id}")
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=["*.bin", "*.pt", "original/"],  # only safetensors
        )
        print(f"Model directory: {local_dir}")
        return cls(local_dir)

    def _build_index(self):
        """
        Populate self._tensor_to_shard from the shard index file.
        Also handles single-file models (no index).
        """
        index_path = self.model_dir / "model.safetensors.index.json"

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            self._tensor_to_shard = index["weight_map"]
            shard_files = sorted(set(self._tensor_to_shard.values()))
            print(f"  Found {len(self._tensor_to_shard)} tensors across {len(shard_files)} shards")
        else:
            # Single-shard model
            shard_path = self.model_dir / "model.safetensors"
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"No safetensors index or single-file found in {self.model_dir}"
                )
            handle = safe_open(str(shard_path), framework="pt", device="cpu")
            self._open_shards["model.safetensors"] = handle
            for name in handle.keys():
                self._tensor_to_shard[name] = "model.safetensors"
            print(f"  Single shard: {len(self._tensor_to_shard)} tensors")

    def _open_shard(self, filename: str) -> safe_open:
        """Open a shard lazily and cache the file handle."""
        if filename not in self._open_shards:
            path = self.model_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Shard not found: {path}")
            self._open_shards[filename] = safe_open(str(path), framework="pt", device="cpu")
        return self._open_shards[filename]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hf(self, hf_name: str, device: str = "cuda") -> torch.Tensor:
        """Load a tensor by its HuggingFace name and move it to device."""
        if hf_name not in self._tensor_to_shard:
            raise KeyError(
                f"Tensor '{hf_name}' not found.\n"
                f"Did you mean one of: {self._suggest(hf_name)}"
            )
        shard = self._open_shard(self._tensor_to_shard[hf_name])
        return shard.get_tensor(hf_name).to(device)

    def get(self, our_name: str, device: str = "cuda") -> torch.Tensor:
        """Load a tensor by our internal name."""
        return self.get_hf(ours_to_hf(our_name), device=device)

    def hf_names(self) -> list[str]:
        """All tensor names in HuggingFace convention, sorted."""
        return sorted(self._tensor_to_shard.keys())

    def our_names(self) -> list[str]:
        """All tensor names in our convention, sorted."""
        return sorted(hf_to_ours(n) for n in self._tensor_to_shard)

    def _iter_metadata(self) -> Iterator[Tuple[str, str, tuple, str]]:
        """Yield (hf_name, our_name, shape, dtype_str) without loading tensor data."""
        for hf_name in self.hf_names():
            shard = self._open_shard(self._tensor_to_shard[hf_name])
            sl = shard.get_slice(hf_name)
            shape = tuple(sl.get_shape())
            dtype = str(sl.get_dtype())
            our_name = hf_to_ours(hf_name)
            yield hf_name, our_name, shape, dtype

    def print_manifest(self, show_our_names: bool = True):
        """
        Print the full weight manifest without loading any tensor data.
        Shows: HF name, our name, shape, dtype, which shard.
        """
        col_hf    = 58
        col_ours  = 32
        col_shape = 24
        col_dtype = 12

        header = f"{'HF name':<{col_hf}} {'Our name':<{col_ours}} {'Shape':<{col_shape}} {'dtype':<{col_dtype}} Shard"
        print("=" * len(header))
        print(header)
        print("=" * len(header))

        total_params = 0
        total_bytes  = 0
        dtype_bytes  = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "I32": 4}

        for hf_name, our_name, shape, dtype in self._iter_metadata():
            n_params = 1
            for d in shape:
                n_params *= d
            total_params += n_params
            total_bytes  += n_params * dtype_bytes.get(dtype, 2)

            shard_short = self._tensor_to_shard[hf_name].replace("model-", "").replace(".safetensors", "")
            print(
                f"{hf_name:<{col_hf}} "
                f"{our_name:<{col_ours}} "
                f"{str(shape):<{col_shape}} "
                f"{dtype:<{col_dtype}} "
                f"{shard_short}"
            )

        print("=" * len(header))
        print(f"Total tensors    : {len(self._tensor_to_shard):,}")
        print(f"Total parameters : {total_params:,}  ({total_params / 1e9:.3f}B)")
        print(f"Total size       : {total_bytes / 1e9:.2f} GB  (on disk, in stored dtype)")
        print("=" * len(header))

    def count_parameters(self) -> int:
        """Sum of all parameter elements without loading tensor data."""
        total = 0
        for _, _, shape, _ in self._iter_metadata():
            n = 1
            for d in shape:
                n *= d
            total += n
        return total

    def verify_parameter_count(self, expected_billions: float = 3.212, tol: float = 0.01):
        """
        Assert that the loaded model has ~expected_billions parameters.
        Llama-3.2-3B has 3.212B parameters (not exactly 3B, as is common).
        """
        actual = self.count_parameters()
        actual_b = actual / 1e9
        diff = abs(actual_b - expected_billions) / expected_billions
        status = "OK" if diff < tol else "MISMATCH"
        print(f"Parameter count: {actual:,} ({actual_b:.3f}B)  expected ~{expected_billions}B  [{status}]")
        if diff >= tol:
            raise ValueError(
                f"Parameter count {actual_b:.3f}B differs from expected {expected_billions}B "
                f"by {diff*100:.1f}%"
            )
        return actual

    def _suggest(self, query: str, n: int = 5) -> list[str]:
        """Find tensor names that contain the query string."""
        return [k for k in self.hf_names() if query in k][:n]


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    repo_id = "meta-llama/Llama-3.2-3B"
    token = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"\n{'='*60}")
    print("  Section 2 — Weight Loader")
    print(f"{'='*60}\n")

    loader = WeightLoader.from_pretrained(repo_id, token=token)

    print()
    loader.print_manifest()

    print()
    loader.verify_parameter_count(expected_billions=3.212)

    # Spot-check: load a single small tensor and inspect it
    print("\nSpot-check: loading embed_tokens weight...")
    w = loader.get("embed_tokens", device="cpu")
    print(f"  embed_tokens shape : {w.shape}   (vocab_size={w.shape[0]}, hidden={w.shape[1]})")
    print(f"  dtype              : {w.dtype}")
    print(f"  min/max            : {w.min():.4f} / {w.max():.4f}")
    print("\nDone.")
