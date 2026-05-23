"""
Sampling — turn next-token logits into an actual token id.

Until now we've been using pure greedy decoding (argmax) in
`iterations/02_kv_cache.py`. That's deterministic and reproduces HF's
`do_sample=False` reference exactly, but it produces robotic, repetitive
text. Real generation needs randomness — and a way to control HOW random.

Three knobs (composed in this order)
------------------------------------

1. **temperature** — divide logits by `T` before softmax.
     T = 1.0 → unchanged
     T < 1.0 → distribution sharpens   (more confident, closer to greedy)
     T > 1.0 → distribution flattens   (more diverse, more random)
     T → 0   → argmax (greedy)

2. **top-k** — keep only the K highest-logit tokens, set the rest to −∞.
     Removes long-tail garbage tokens entirely.
     K = 0 disables top-k.

3. **top-p (nucleus)** — keep the smallest set of tokens whose cumulative
     probability ≥ p; set the rest to −∞.
     Adapts the cutoff per-step: peaky distributions keep few tokens,
     flat ones keep many. p = 1.0 disables top-p.

After filtering, softmax + `torch.multinomial` picks one token.

All functions are written batch-wise: input `(B, vocab)`, output `(B,)`.
This matches what `model(...)[:, -1, :]` produces in the decode loop.
"""

from __future__ import annotations

import torch


# ── temperature ──────────────────────────────────────────────────────────────

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits / temperature.
    temperature == 1.0 is a no-op; we keep the branch so callers can pass
    1.0 without paying for a divide.
    """
    if temperature == 1.0:
        return logits
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0 (got {temperature}); "
            f"use sample(..., temperature=0) or greedy() for argmax."
        )
    return logits / temperature


# ── greedy ───────────────────────────────────────────────────────────────────

def greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    Pick the single highest-logit token per row.
        logits: (..., vocab)
        return: (...,)  long tensor of token ids
    """
    return logits.argmax(dim=-1)


# ── top-k ────────────────────────────────────────────────────────────────────

def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask everything below the k-th largest logit to −∞.
    Does NOT sample — returns filtered logits ready for softmax+multinomial.

    k <= 0 or k >= vocab → no-op.
    """
    vocab = logits.shape[-1]
    if k <= 0 or k >= vocab:
        return logits

    # values: the k largest logits per row, sorted descending
    # values[..., -1] is the threshold — anything below it gets masked
    values, _ = torch.topk(logits, k, dim=-1)
    threshold = values[..., -1:].expand_as(logits)
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Convenience wrapper: top-k filter, then sample one token."""
    filtered = filter_top_k(logits, k)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ── top-p (nucleus) ──────────────────────────────────────────────────────────

def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Mask the long tail past cumulative probability `p` to −∞.

    Algorithm (the standard HF/Holtzman-2019 formulation):
      1. sort logits descending
      2. softmax → cumulative sum
      3. mark tokens whose CUMSUM is past p as "remove"
      4. shift the mask right by one position so the first token that
         crosses p is INCLUDED (otherwise a peaky distribution where
         the top token alone has prob > p would remove everything)
      5. unsort the mask back to the original token order
    """
    if p >= 1.0:
        return logits
    if p <= 0.0:
        # Degenerate — fall back to greedy by keeping only the top-1.
        return filter_top_k(logits, 1)

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs              = torch.softmax(sorted_logits, dim=-1)
    cum_probs                 = sorted_probs.cumsum(dim=-1)

    # Tokens to remove in SORTED order: cumprob already past p.
    to_remove_sorted = cum_probs > p
    # Shift right by 1 so the token that PUSHED us past p is kept.
    to_remove_sorted[..., 1:] = to_remove_sorted[..., :-1].clone()
    to_remove_sorted[..., 0]  = False

    # Scatter the mask back to original token order.
    to_remove = torch.zeros_like(to_remove_sorted)
    to_remove.scatter_(-1, sorted_idx, to_remove_sorted)

    return logits.masked_fill(to_remove, float("-inf"))


def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Convenience wrapper: top-p filter, then sample one token."""
    filtered = filter_top_p(logits, p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ── composed sampler ─────────────────────────────────────────────────────────

def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    The one function the generation loop calls.

        logits: (..., vocab) — usually (B, vocab) for one decode step
        return: (...,)  sampled token ids

    Behaviour:
      temperature == 0  → greedy (argmax), top_k/top_p ignored
      otherwise         → divide by T, apply top-k, then top-p, then sample

    The (top_k, top_p) order matters: top-k bounds the candidate set first
    (a hard cap), top-p then adaptively prunes within that set.
    """
    if temperature == 0:
        return greedy(logits)

    logits = apply_temperature(logits, temperature)

    if top_k > 0:
        logits = filter_top_k(logits, top_k)
    if top_p < 1.0:
        logits = filter_top_p(logits, top_p)

    # Cast to float32 for the softmax — bf16 softmax of a 128k-vocab row
    # accumulates noticeable error in the tail.
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
