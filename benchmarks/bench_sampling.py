"""
Section 12 — sampling correctness checks + latency benchmark.

Three quick sanity tests (deterministic where possible):

  1. greedy always picks the highest logit
  2. top-k filter zeroes out all but k tokens after softmax
  3. top-p with p=1.0 is a no-op (equivalent to plain multinomial)
     top-p with p≈0 collapses to greedy
     top-p actually trims the tail (peaky distributions keep few tokens)

Then a latency micro-benchmark at realistic shapes:
  - vocab = 128,256 (Llama 3 vocab)
  - batch = 1, 8, 32
  - sample_top_k(k=50), sample_top_p(p=0.9), sample(T=0.7, k=50, p=0.9)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from sampling import (
    apply_temperature,
    greedy,
    filter_top_k,
    filter_top_p,
    sample_top_k,
    sample_top_p,
    sample,
)
from benchmarks.bench_utils import bench_fn, record, print_results

DEVICE = "cuda"
DTYPE  = torch.float32   # sampling runs in fp32; the model casts logits at the call site
VOCAB  = 128_256


# ── tests ────────────────────────────────────────────────────────────────────

def test_greedy_picks_max():
    torch.manual_seed(0)
    logits = torch.randn(8, VOCAB, device=DEVICE, dtype=DTYPE)
    out = greedy(logits)
    expected = logits.argmax(dim=-1)
    assert torch.equal(out, expected), "greedy did not return argmax"
    print("  [PASS] greedy(logits) == argmax(logits)")


def test_temperature_no_op():
    logits = torch.randn(4, VOCAB, device=DEVICE, dtype=DTYPE)
    out = apply_temperature(logits, 1.0)
    assert out is logits or torch.equal(out, logits), "T=1.0 should be a no-op"
    print("  [PASS] apply_temperature(logits, 1.0) is a no-op")


def test_top_k_filter_keeps_k():
    torch.manual_seed(1)
    logits = torch.randn(4, VOCAB, device=DEVICE, dtype=DTYPE)
    k = 50
    filtered = filter_top_k(logits, k)
    # After filtering, exactly k entries per row should be finite (the rest -inf)
    finite_count = torch.isfinite(filtered).sum(dim=-1)
    assert torch.all(finite_count == k), f"expected exactly {k} finite per row, got {finite_count.tolist()}"
    # And they must be the k largest from the original logits
    top_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = top_vals[..., -1:]
    expected_finite_mask = logits >= threshold
    actual_finite_mask = torch.isfinite(filtered)
    # Could differ on exact ties, but with random fp32 logits ties are vanishingly rare
    assert torch.equal(expected_finite_mask, actual_finite_mask), "top-k mask mismatch"
    print(f"  [PASS] filter_top_k keeps exactly {k} largest per row")


def test_top_p_p1_is_noop():
    logits = torch.randn(4, VOCAB, device=DEVICE, dtype=DTYPE)
    out = filter_top_p(logits, 1.0)
    assert torch.equal(out, logits), "top_p with p=1.0 should leave logits untouched"
    print("  [PASS] filter_top_p(logits, 1.0) is a no-op")


def test_top_p_keeps_at_least_one():
    # Construct a peaky distribution where the top token alone has prob > 0.99
    logits = torch.full((1, 16), -1e4, device=DEVICE, dtype=DTYPE)
    logits[0, 7] = 0.0  # huge gap → softmax puts ~1.0 on this token
    filtered = filter_top_p(logits, 0.5)
    # The top token must survive even though its prob already exceeds p
    assert torch.isfinite(filtered[0, 7]), "top-p removed the single highest-prob token"
    print("  [PASS] filter_top_p always retains the top token (peaky case)")


def test_top_p_trims_tail():
    # Uniform-ish distribution: top-p should keep roughly p*V tokens
    torch.manual_seed(2)
    V = 1000
    logits = torch.randn(1, V, device=DEVICE, dtype=DTYPE) * 0.01  # nearly uniform
    p = 0.5
    filtered = filter_top_p(logits, p)
    n_finite = torch.isfinite(filtered).sum().item()
    # Very loose bound: should keep "around half" the vocab for p=0.5 + near-uniform
    assert 0.3 * V < n_finite < 0.7 * V, (
        f"top-p with p={p} on near-uniform kept {n_finite}/{V}, expected ~{int(p*V)}"
    )
    print(f"  [PASS] filter_top_p trims tail: kept {n_finite}/{V} at p={p} (near-uniform)")


def test_sample_temperature_zero_is_greedy():
    torch.manual_seed(3)
    logits = torch.randn(8, VOCAB, device=DEVICE, dtype=DTYPE)
    out = sample(logits, temperature=0)
    assert torch.equal(out, greedy(logits)), "sample(T=0) must equal greedy"
    print("  [PASS] sample(logits, temperature=0) == greedy(logits)")


def test_sample_returns_valid_tokens():
    torch.manual_seed(4)
    logits = torch.randn(8, VOCAB, device=DEVICE, dtype=DTYPE)
    for cfg in [
        dict(temperature=1.0,  top_k=0,   top_p=1.0),  # plain multinomial
        dict(temperature=0.7,  top_k=50,  top_p=1.0),  # top-k only
        dict(temperature=0.7,  top_k=0,   top_p=0.9),  # top-p only
        dict(temperature=0.7,  top_k=50,  top_p=0.9),  # combined
    ]:
        out = sample(logits, **cfg)
        assert out.shape == (8,), f"bad shape for {cfg}: {out.shape}"
        assert out.dtype == torch.long
        assert (out >= 0).all() and (out < VOCAB).all(), "out-of-range token id"
    print("  [PASS] sample(...) returns valid (B,) token ids for all configs")


# ── benchmark ────────────────────────────────────────────────────────────────

def run_benchmarks():
    print("\n--- Sampling latency (vocab = 128,256) ---")
    print("  Sampling cost is trivial vs a forward pass, but we measure to confirm.\n")

    print(f"  {'Config':<42} {'B':>3}  {'Latency':>10}")
    print(f"  {'-'*42} {'-'*3}  {'-'*10}")

    cases = [
        ("greedy",                     dict(temperature=0)),
        ("sample T=1.0  (pure multinomial)", dict(temperature=1.0)),
        ("sample T=0.7  top_k=50",     dict(temperature=0.7, top_k=50)),
        ("sample T=0.7  top_p=0.9",    dict(temperature=0.7, top_p=0.9)),
        ("sample T=0.7  k=50  p=0.9",  dict(temperature=0.7, top_k=50, top_p=0.9)),
    ]
    for label, kwargs in cases:
        for B in (1, 8, 32):
            torch.manual_seed(0)
            logits = torch.randn(B, VOCAB, device=DEVICE, dtype=DTYPE)
            lat = bench_fn(lambda: sample(logits, **kwargs))
            print(f"  {label:<42} {B:>3}  {lat:>9.4f}ms")
            record("sampling", "pytorch", f"{label} B={B}", lat,
                   extra={"batch": B, **{k: v for k, v in kwargs.items()}})


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("  Section 12 — Sampling correctness + benchmark")
    print(f"{'='*70}\n")

    print("--- Correctness ---")
    test_greedy_picks_max()
    test_temperature_no_op()
    test_top_k_filter_keeps_k()
    test_top_p_p1_is_noop()
    test_top_p_keeps_at_least_one()
    test_top_p_trims_tail()
    test_sample_temperature_zero_is_greedy()
    test_sample_returns_valid_tokens()

    run_benchmarks()
    print_results("sampling")
    print(f"\n  Results saved to benchmarks/results_baseline.json")
