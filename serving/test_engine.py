"""
Parity / correctness test for the continuous-batching engine (Section 15).

What we actually need to prove
------------------------------
Continuous batching is correct iff a request's output is INDEPENDENT of whatever
else happens to share its batches. The thing that can silently break it is
cross-request contamination: wrong KV slot, wrong per-row position/RoPE angle, or
a mask that lets one request peek at another's tokens. So the strongest test is
batch invariance:

    a request decoded SOLO must produce byte-identical tokens to the same
    request decoded CONCURRENTLY with others, and even under slot-reuse
    pressure (more requests than slots → queueing + eviction + slot recycling).

We use greedy decoding (temperature=0) so "identical" is a hard, exact check.

We additionally sanity-check the math against the proven single-stream path
(model.forward prefill + model.decode_step): the engine's first token must equal
that path's argmax for every prompt. (We only assert the first token because the
reference path may run the Triton kernels while the engine runs PyTorch SDPA;
greedy sequences can drift after many tokens from tiny numeric differences. The
batch-invariance check above is what actually guards decode-loop correctness.)

Run:
    XDG_CONFIG_HOME=~/.cache/xdgconfig UV_CACHE_DIR=~/.cache/uv \
    PATH=~/.local/bin:$PATH uv run python -m serving.test_engine
"""

from __future__ import annotations

import env_loader  # noqa: F401  loads .env (HF_TOKEN)
import torch

from config import ModelConfig
from loader import WeightLoader
from model.kv_cache import KVCache
from model.llama import LlamaModel
from tokenizer import Tokenizer
from serving.engine import InferenceEngine

MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda"
DTYPE = torch.bfloat16

MAX_NEW = 24
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a land far away,",
    "def add(a, b):\n    return",
    "The three primary colors are red,",
]


def build_model() -> tuple[LlamaModel, Tokenizer]:
    print("Loading tokenizer + model...")
    tok = Tokenizer.from_pretrained(MODEL_ID)
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()
    return model, tok


@torch.no_grad()
def reference_greedy(model: LlamaModel, prompt_ids: list[int], max_new: int, max_seq: int) -> list[int]:
    """Ground-truth greedy decode using the proven forward / decode_step path."""
    eos = model.cfg.eos_token_id
    kv = KVCache(
        n_layers=model.cfg.num_hidden_layers,
        max_batch=1,
        max_seq_len=max_seq,
        n_heads_kv=model.cfg.num_key_value_heads,
        head_dim=model.cfg.head_dim,
        dtype=DTYPE,
        device=DEVICE,
    )
    pt = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    logits = model(pt, start_pos=0, kv_cache=kv)
    g = int(logits[:, -1, :].argmax(dim=-1).item())
    out = [g]
    pos = len(prompt_ids)
    while not (g == eos or len(out) >= max_new):
        tok = torch.tensor([[g]], dtype=torch.long, device=DEVICE)
        logits = model.decode_step(tok, pos, kv)
        pos += 1
        g = int(logits[:, -1, :].argmax(dim=-1).item())
        out.append(g)
    return out


def engine_outputs(model, prompt_id_lists, max_running, max_seq) -> dict[int, list[int]]:
    """Run all prompts through one engine; return {prompt_index: token_ids}."""
    engine = InferenceEngine(
        model=model,
        max_running=max_running,
        max_seq_len=max_seq,
        temperature=0.0,  # greedy → deterministic
        warmup=False,     # correctness test: keep construction state pristine
    )
    reqs = []
    for ids in prompt_id_lists:
        reqs.append(engine.add_request(ids, max_new_tokens=MAX_NEW))
    engine.run()
    return {i: r.generated for i, r in enumerate(reqs)}


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    model, tok = build_model()

    prompt_ids = [tok.encode(p, add_bos=True) for p in PROMPTS]
    max_prompt = max(len(ids) for ids in prompt_ids)
    max_seq = max_prompt + MAX_NEW + 1

    print(f"\n{len(PROMPTS)} prompts, max_new={MAX_NEW}, max_seq={max_seq}\n")

    # Reference (proven single-stream path).
    print("Reference greedy (forward + decode_step)...")
    reference = [reference_greedy(model, ids, MAX_NEW, max_seq) for ids in prompt_ids]

    # Engine, three batching regimes.
    print("Engine: solo (max_running=1, one prompt at a time)...")
    solo = {}
    for i, ids in enumerate(prompt_ids):
        solo[i] = engine_outputs(model, [ids], max_running=1, max_seq=max_seq)[0]

    print("Engine: concurrent-all (max_running=len(prompts))...")
    concurrent = engine_outputs(model, prompt_ids, max_running=len(PROMPTS), max_seq=max_seq)

    print("Engine: concurrent-limited (max_running=2 → queueing + slot reuse)...")
    limited = engine_outputs(model, prompt_ids, max_running=2, max_seq=max_seq)

    print("Engine: warmed (warmup=True → must leave pristine state)...")
    warm_engine = InferenceEngine(
        model=model, max_running=len(PROMPTS), max_seq_len=max_seq,
        temperature=0.0, warmup=True,
    )
    warm_reqs = [warm_engine.add_request(ids, max_new_tokens=MAX_NEW) for ids in prompt_ids]
    warm_engine.run()
    warmed = {i: r.generated for i, r in enumerate(warm_reqs)}

    # ── checks ────────────────────────────────────────────────────────────
    failures = 0
    for i, p in enumerate(PROMPTS):
        ref, so, co, li, wa = reference[i], solo[i], concurrent[i], limited[i], warmed[i]

        inv_ok = (so == co == li == wa)
        first_ok = (so[0] == ref[0])
        # how far the engine's solo greedy matches the reference path
        pref = 0
        for a, b in zip(so, ref):
            if a != b:
                break
            pref += 1

        status = "OK" if (inv_ok and first_ok) else "FAIL"
        if not (inv_ok and first_ok):
            failures += 1

        print(f"\n[{status}] prompt {i}: {p!r}")
        print(f"   batch-invariant (solo==concurrent==limited==warmed): {inv_ok}")
        print(f"   first-token matches reference:                       {first_ok}")
        print(f"   reference prefix match:                              {pref}/{len(so)} tokens")
        print(f"   engine text: {tok.decode(so, skip_special=True)!r}")
        if not inv_ok:
            print(f"     solo:       {so}")
            print(f"     concurrent: {co}")
            print(f"     limited:    {li}")
            print(f"     warmed:     {wa}")

    print("\n" + "=" * 60)
    if failures == 0:
        print(f"PASS — all {len(PROMPTS)} prompts batch-invariant and math-correct.")
    else:
        print(f"FAIL — {failures}/{len(PROMPTS)} prompts failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
