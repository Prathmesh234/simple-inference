"""
generate.py — Section 13

Streaming generation loop: encode → prefill → decode → yield.

Usage
-----
    from generate import generate
    from model.llama import LlamaModel
    from model.kv_cache import KVCache
    from tokenizer import Tokenizer

    for token_str in generate(
        prompt        = "The theory of relativity states that",
        model         = model,
        tokenizer     = tokenizer,
        kv_cache      = kv_cache,
        max_new_tokens = 200,
        temperature   = 1.0,
        top_k         = 50,
        top_p         = 0.9,
    ):
        print(token_str, end="", flush=True)

Design notes
------------
- `generate` is a Python generator — it yields one decoded token string per
  step so the caller can stream output to a terminal, websocket, etc.
- The KV cache is reset at the start of each call so the same cache object
  can be reused across multiple independent prompts.
- EOS detection: stop as soon as the model produces `tokenizer.eos_id`.
- Prefill timing and decode timing are optionally returned via a stats dict
  if `return_stats=True` is passed.
"""

from __future__ import annotations

from typing import Generator, Optional
import time

import torch

from model.llama import LlamaModel
from model.kv_cache import KVCache
from tokenizer import Tokenizer
from sampling import sample
from utilities import write_run_metrics, print_metrics_table


def generate(
    prompt: str,
    model: LlamaModel,
    tokenizer: Tokenizer,
    kv_cache: KVCache,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """
    Encode `prompt`, run prefill, then decode up to `max_new_tokens` tokens,
    yielding each decoded token string as it is produced.

    Stops early if the model generates the EOS token.

    Args:
        prompt:         raw text to condition on
        model:          loaded LlamaModel (eval mode, on device)
        tokenizer:      Tokenizer wrapping the HF fast tokenizer
        kv_cache:       pre-allocated KVCache (will be reset before use)
        max_new_tokens: hard cap on generated tokens
        temperature:    sampling temperature (0 = greedy)
        top_k:          top-k filter (0 = disabled)
        top_p:          nucleus filter (1.0 = disabled)

    Yields:
        Decoded string for each new token (not including the prompt).
    """
    device = next(model.parameters()).device

    # 1. Encode
    input_ids = tokenizer.encode(prompt, add_bos=True)
    prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    T = prompt_tensor.shape[1]

    if T >= kv_cache.max_seq_len:
        raise ValueError(f"Prompt length {T} exceeds KVCache capacity {kv_cache.max_seq_len}")

    kv_cache.reset()

    # 2. Prefill
    with torch.no_grad():
        logits = model(prompt_tensor, start_pos=0, kv_cache=kv_cache)

    next_tok_id = sample(
        logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
    ).item()

    if next_tok_id == tokenizer.eos_id:
        return


    # 3. Decode loop
    pos = T
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if pos >= kv_cache.max_seq_len:
                break
            tok_tensor = torch.tensor([[next_tok_id]], dtype=torch.long, device=device)
            logits = model.decode_step(tok_tensor, pos, kv_cache)
            next_tok_id = sample(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            ).item()
            pos += 1

            if next_tok_id == tokenizer.eos_id:
                break

            yield tokenizer.decode([next_tok_id], skip_special=True)


def generate_with_stats(
    prompt: str,
    model: LlamaModel,
    tokenizer: Tokenizer,
    kv_cache: KVCache,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    on_token=None,
) -> tuple[str, dict]:
    """
    Non-streaming version that returns the full generated text and timing stats.

    Returns:
        (generated_text, stats)  where stats = {
            "prompt_tokens":  int,    # prompt length fed to prefill
            "new_tokens":     int,    # tokens actually generated
            "ttft_ms":        float,  # Time To First Token: request start →
                                      #   first generated token (prefill + 1 sample)
            "tpot_ms":        float,  # Time Per Output Token: avg inter-token
                                      #   latency over decode steps (excludes TTFT)
            "prefill_ms":     float,  # prefill forward pass only
            "decode_tok_s":   float,  # steady-state decode throughput = 1000/tpot
            "total_wall_ms":  float,  # whole call wall time
            "total_tok_s":    float,  # (prompt+new) / total_wall
            "peak_vram_gb":   float,
        }
    """
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, add_bos=True)
    prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    T = prompt_tensor.shape[1]

    if T >= kv_cache.max_seq_len:
        raise ValueError(f"Prompt length {T} exceeds KVCache capacity {kv_cache.max_seq_len}")

    kv_cache.reset()
    tokens_out = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    wall_start = time.perf_counter()

    # Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(prompt_tensor, start_pos=0, kv_cache=kv_cache)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    next_tok_id = sample(
        logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
    ).item()

    # TTFT: request start → first token ready (prefill + first sample).
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - wall_start) * 1000

    decode_times = []
    pos = T

    if next_tok_id != tokenizer.eos_id:
        tokens_out.append(next_tok_id)
        if on_token is not None:
            on_token(tokenizer.decode([next_tok_id], skip_special=True))

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens - 1):
                    if pos >= kv_cache.max_seq_len:
                        break
                    tok_tensor = torch.tensor([[next_tok_id]], dtype=torch.long, device=device)
                    torch.cuda.synchronize()
                    ts = time.perf_counter()
                    logits = model.decode_step(tok_tensor, pos, kv_cache)
                    torch.cuda.synchronize()
                    decode_times.append((time.perf_counter() - ts) * 1000)

                    next_tok_id = sample(
                        logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
                    ).item()
                    pos += 1

                    if next_tok_id == tokenizer.eos_id:
                        break
                    tokens_out.append(next_tok_id)
                    if on_token is not None:
                        on_token(tokenizer.decode([next_tok_id], skip_special=True))
        except KeyboardInterrupt:
            # Stop cleanly so partial timing stats are still returned/logged.
            pass

    total_wall_ms = (time.perf_counter() - wall_start) * 1000
    # TPOT (Time Per Output Token) = average inter-token latency across decode
    # steps, i.e. every token AFTER the first. This is the steady-state cost.
    tpot_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0
    generated_text = tokenizer.decode(tokens_out, skip_special=True)
    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
    )
    total_new = len(tokens_out)

    stats = {
        "prompt_tokens": T,
        "new_tokens":    total_new,
        "ttft_ms":       round(ttft_ms, 2),
        "tpot_ms":       round(tpot_ms, 3),
        "prefill_ms":    round(prefill_ms, 2),
        "decode_tok_s":  round(1000.0 / tpot_ms, 1) if tpot_ms > 0 else 0.0,
        "total_wall_ms": round(total_wall_ms, 2),
        "total_tok_s":   round((T + total_new) / (total_wall_ms * 1e-3), 1) if total_wall_ms > 0 else 0.0,
        "peak_vram_gb":  round(peak_vram_gb, 2),
    }
    return generated_text, stats


# ---------------------------------------------------------------------------
# Quick demo (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import env_loader  # noqa: F401  loads .env for HF_TOKEN
    from config import ModelConfig
    from loader import WeightLoader
    import model.llama as llama_mod

    DEVICE   = "cuda"
    DTYPE    = torch.bfloat16
    MODEL_ID = "meta-llama/Llama-3.2-3B"
    PROMPT   = (
        "Question: Design a real-time chat application that can scale to millions of concurrent users.\n"
        "Answer: To design a real-time chat application at scale, we use a microservices architecture with WebSockets for persistent connections, a message broker like Redis Pub/Sub for routing messages, and a NoSQL database like Cassandra for storing message history. The key components are:\n"
        "1. Connection Gateway: Handles WebSocket handshakes and maintains active connection states.\n"
        "2. Message Broker: Distributes incoming messages to the correct gateway servers.\n"
        "3. Database Cluster: Writes chat history asynchronously to avoid blocking the hot path.\n\n"
        "Question: Design a distributed, fault-tolerant key-value store that supports strong consistency and automatic partitioning.\n"
        "Answer: A consistent key-value store can be designed using the Raft consensus protocol for replication and consistent hashing for partitioning. The system consists of:\n"
        "1. Consensus Group (Raft): Ensures all write operations are committed in a linearizable log across a quorum of nodes.\n"
        "2. Partition Router: Maps keys to specific node rings using consistent hashing to minimize data movement during scaling.\n"
        "3. Storage Engine: Uses Log-Structured Merge (LSM) trees (like RocksDB) for high-performance write throughput.\n\n"
        "Question: Design a system that can download, parse, and index billions of web pages daily for a search engine. Explain the crawling pipeline, duplicate detection, and scaling strategies.\n"
        "Answer:"
    )

    print("Loading tokenizer...")
    tok = Tokenizer.from_pretrained(MODEL_ID)

    print("Loading model...")
    cfg    = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model  = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()
    if model.maybe_compile():
        print(f"  torch.compile: enabled (mode={llama_mod.COMPILE_MODE})")

    kv = KVCache(
        n_layers    = cfg.num_hidden_layers,
        max_batch   = 1,
        max_seq_len = 4096,
        n_heads_kv  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
        device      = DEVICE,
    )

    print(f"\nPrompt: {PROMPT!r}\n")
    print("Output: ", end="", flush=True)

    SAMPLING = {"temperature": 0.7, "top_k": 50, "top_p": 0.9}

    # Single measured run: stream tokens live AND collect TTFT/TPOT in one pass.
    # generate_with_stats swallows KeyboardInterrupt internally and returns the
    # partial stats, so the metrics file is written even if you Ctrl-C early.
    generated_text, stats = generate_with_stats(
        PROMPT, model, tok, kv, max_new_tokens=2048,
        on_token=lambda s: print(s, end="", flush=True),
        **SAMPLING,
    )

    # Which attention/RMSNorm/etc. path actually ran this generation.
    import ops.attention as attn_mod
    backend = "triton (fused kernels)" if attn_mod.USE_TRITON else "pytorch (SDPA reference)"

    print_metrics_table(stats, backend)
    run_path = write_run_metrics(
        PROMPT, generated_text, stats,
        backend=backend, model_id=MODEL_ID, sampling=SAMPLING,
    )
    print(f"\n  Metrics saved to {run_path}")
