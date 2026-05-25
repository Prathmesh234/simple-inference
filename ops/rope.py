"""
Rotary Position Embeddings (RoPE).

Why position encoding at all?
------------------------------
The attention operation (Q @ K^T) is permutation-invariant — if you shuffled
the tokens, the dot products would be the same. The model has no idea which
token came first. Position encoding injects that information.

Why RoPE instead of learned position embeddings?
-------------------------------------------------
Earlier models (GPT-2, BERT) added a learned vector to each position:
    x_pos = x + embedding[position]

Problem: if you trained on sequences up to length 2048, position 2049 has
no learned embedding. The model breaks outside its training length.

RoPE encodes position as a *rotation* of the query and key vectors instead
of an addition. The rotation angle depends on position, and the math works
out so that the dot product Q·K only depends on the *relative* distance
between positions, not absolute position. This generalises naturally beyond
the training length (with the scaling trick Llama 3 uses).

How RoPE works
--------------
Each head has head_dim=128 dimensions. RoPE treats these as 64 pairs:
    (d0, d1), (d2, d3), ..., (d126, d127)

For position p, pair i gets rotated by angle:
    θ_i = p / (rope_theta ^ (2i / head_dim))

The rotation is:
    [x_0, x_1] → [x_0·cos(θ) - x_1·sin(θ),  x_0·sin(θ) + x_1·cos(θ)]

Lower dimensions (small i) rotate slowly — they encode coarse position.
Higher dimensions rotate quickly — they encode fine-grained position.
This is analogous to how a clock has hour/minute/second hands at different
frequencies.

Llama 3 RoPE scaling
---------------------
Llama 3.2-3B was pretrained on sequences up to 8192 tokens but supports
131,072 tokens at inference via a scaling trick:
  - Frequencies below low_freq_factor (1.0) get divided by factor (32.0)
    → slow rotation → can handle long distances
  - Frequencies above high_freq_factor (4.0) are unchanged
    → fast rotation unchanged → short-range still works
  - Frequencies in between get a smooth linear interpolation

Shape contract
--------------
RopeFrequencies:
  cos_freqs: (max_seq_len, head_dim)
  sin_freqs: (max_seq_len, head_dim)

apply_rope:
  input q/k: (batch, seq_len, n_heads, head_dim)
  output:    (batch, seq_len, n_heads, head_dim)  same shape, rotated
"""

import math
import os
import torch

# Triton kernel dispatch (Section 14c).
#   Controlled by the USE_TRITON env var (set in .env or shell).
#   Defaults to True — Triton is the production path. Set USE_TRITON=false
#   to force the pure-PyTorch reference (useful for debugging / parity checks).
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


class RopeFrequencies:
    """
    Precompute the cos and sin tables for all positions up to max_seq_len.
    Built once at model load time, reused for every forward pass.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 500_000.0,
        # Llama 3 scaling parameters
        rope_type: str = "llama3",
        factor: float = 32.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_seq_len: int = 8192,
        device: torch.device = None,
    ):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: base frequencies — one per dimension pair
        # freq_i = 1 / (rope_theta ^ (2i / head_dim))
        dim_idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        base_freqs = 1.0 / (rope_theta ** (dim_idx / head_dim))  # (head_dim/2,)

        # Step 2: apply Llama 3 long-context scaling
        if rope_type == "llama3":
            base_freqs = self._apply_llama3_scaling(
                base_freqs, factor, low_freq_factor, high_freq_factor,
                original_max_seq_len,
            )

        # Step 3: build the full (max_seq_len, head_dim/2) angle table
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # outer product: angles[p, i] = position p * frequency i
        angles = torch.outer(positions, base_freqs)  # (max_seq_len, head_dim/2)

        # Step 4: duplicate each angle to cover both elements of each pair
        # pair (d0, d1) both use the same angle θ_i
        angles = torch.cat([angles, angles], dim=-1)  # (max_seq_len, head_dim)

        # Store as float32 — we cast to bfloat16 at application time
        self.cos = angles.cos()  # (max_seq_len, head_dim)
        self.sin = angles.sin()  # (max_seq_len, head_dim)

    @staticmethod
    def _apply_llama3_scaling(
        freqs: torch.Tensor,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_seq_len: int,
    ) -> torch.Tensor:
        """
        Scale base frequencies for long-context extrapolation.

        Wavelength of a frequency = 2π / freq.
        If the wavelength is longer than the original training context,
        the model hasn't seen enough rotations of that frequency to have
        learned it — so we scale it down (slow it further).
        """
        low_freq_wavelen  = original_max_seq_len / low_freq_factor
        high_freq_wavelen = original_max_seq_len / high_freq_factor

        new_freqs = []
        for freq in freqs.tolist():
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                # Short wavelength — no change
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                # Very long wavelength — scale down by full factor
                new_freqs.append(freq / factor)
            else:
                # In between — smooth linear interpolation
                smooth = (original_max_seq_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)

        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def get(self, seq_len: int, start_pos: int = 0):
        """
        Return (cos, sin) slices for positions [start_pos, start_pos + seq_len).
        start_pos is non-zero during decode (we're at position > 0 in the sequence).
        """
        cos = self.cos[start_pos : start_pos + seq_len]  # (seq_len, head_dim)
        sin = self.sin[start_pos : start_pos + seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange x so the rotation formula can be applied as a simple multiply.

    RoPE rotation of pair (x0, x1):
        out = (x0·cos - x1·sin,  x1·cos + x0·sin)
            = x * cos + rotate_half(x) * sin

    rotate_half turns [x0, x1, x2, x3, ...]
                  into [-x_{n/2}, ..., -x_{n-1}, x_0, ..., x_{n/2-1}]
    so that multiplying by sin gives the correct cross terms.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q:   (batch, seq_len, n_heads_q,  head_dim)
        k:   (batch, seq_len, n_heads_kv, head_dim)
        cos: (seq_len, head_dim)   from RopeFrequencies.get()
        sin: (seq_len, head_dim)

    Returns:
        q_rot, k_rot — same shapes as input
    """
    if USE_TRITON and q.is_cuda:
        from kernels.rope_kernel import rope_triton
        return rope_triton(q, k, cos, sin)

    # Broadcast cos/sin over batch and head dimensions
    # (seq_len, head_dim) → (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot.to(q.dtype), k_rot.to(k.dtype)
