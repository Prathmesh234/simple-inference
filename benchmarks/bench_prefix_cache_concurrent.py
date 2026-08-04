"""Concurrent chat-prefix benchmark with an explicit eviction/recompute phase.

The benchmark models a chat service where every request starts with the same
system prompt and then diverges into a user-specific query:

1. Seed the radix cache with the shared system prompt.
2. Prefill several user requests concurrently and verify they all reuse it.
3. Insert a newer unrelated prompt large enough to evict the shared prefix.
4. Re-run the same concurrent requests and verify they miss and recompute it.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import env_loader  # noqa: F401
import torch

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from serving.engine import InferenceEngine
from tokenizer import Tokenizer


MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE = "cuda"
DTYPE = torch.bfloat16
SYSTEM_TEXT = (
    "You are the shared assistant for a large enterprise chat application. "
    "Every conversation begins with these operating instructions. Protect user "
    "privacy, never expose credentials or internal identifiers, distinguish "
    "verified facts from assumptions, and ask for clarification when required. "
    "Give accurate, concise, actionable answers suitable for production use. "
    "When diagnosing a problem, summarize the symptoms, identify the most likely "
    "causes, propose safe troubleshooting steps in priority order, and explain "
    "how the user can verify that the issue is resolved. Do not invent account "
    "state, message contents, audit events, or administrative permissions. "
    "For security-sensitive requests, prefer reversible actions, preserve an "
    "audit trail, and recommend escalation to an authorized administrator when "
    "the requested operation exceeds the user's role. For data export, retention, "
    "or deletion questions, explain the distinction between workspace policy, "
    "legal retention, backups, and the user's visible conversation history. "
    "For collaboration questions, account for channels, direct messages, threads, "
    "mentions, notifications, file attachments, bots, integrations, and regional "
    "availability. Format multi-step procedures as ordered steps and mention any "
    "important prerequisites before the procedure begins. "
)
SEED_QUERY = (
    "Seed request: prepare an architecture review of how the service stores "
    "conversation metadata, routes websocket messages between regions, retries "
    "temporarily failed deliveries, maintains ordering inside a conversation, "
    "and records audit events without exposing private message contents. Include "
    "the expected behavior during a regional failover and explain which state "
    "must remain strongly consistent. "
)
USER_QUERIES = (
    (
        "I replaced my phone and no longer have access to the authenticator app. "
        "Explain how I can recover my account and reset the second factor without "
        "losing conversation history, workspace memberships, saved messages, or "
        "notification preferences. Include the checks support should perform "
        "before approving recovery and the steps I should take afterward. "
    ),
    (
        "Our incident-response channel accumulated several hundred messages "
        "overnight. Describe a safe workflow for summarizing unread discussions "
        "by thread, identifying unresolved action items, separating confirmed "
        "facts from speculation, and preserving links to the original messages "
        "so responders can verify every conclusion. "
    ),
    (
        "A large diagnostic archive failed during upload after reaching almost "
        "one hundred percent. Help me determine whether the cause is file size, "
        "network interruption, malware scanning, storage quota, or an expired "
        "upload session. Give retry steps that avoid duplicate attachments and "
        "explain what information should be collected for support. "
    ),
    (
        "Draft a response to an enterprise customer who is asking how message "
        "encryption, administrator access, retention policies, legal holds, and "
        "backup deletion interact. Keep the answer technically precise, avoid "
        "making unsupported compliance claims, and clearly identify which "
        "settings are controlled by the workspace administrator. "
    ),
    (
        "Explain the operational difference between muting a channel, disabling "
        "mentions, archiving it, and leaving it. Cover what happens to unread "
        "state, search visibility, notification delivery, membership history, "
        "thread replies, and the ability to rejoin later. Recommend the least "
        "disruptive option for a channel that is useful only during incidents. "
    ),
    (
        "Mobile notifications are delayed by several minutes while desktop "
        "notifications arrive immediately. Provide a prioritized diagnostic plan "
        "covering device power management, push-notification permissions, quiet "
        "hours, notification routing, multiple signed-in devices, connectivity, "
        "and server-side delivery health. Include a way to isolate each cause. "
    ),
    (
        "We need to export selected conversations for an internal compliance "
        "review. Explain how to define the scope, preserve message edits and "
        "deletions, include thread context and attachments, validate export "
        "integrity, restrict access to the archive, document chain of custody, "
        "and dispose of the export after the review is complete. "
    ),
    (
        "Our company has grown to several thousand employees across engineering, "
        "sales, support, and external partner organizations. Propose a channel "
        "and permission structure that limits sensitive access, prevents channel "
        "sprawl, supports discoverability, gives incident teams temporary access, "
        "and defines ownership and archival rules for inactive projects. "
    ),
)
COLD_TEXT = (
    "This unrelated eviction workload is a technical handbook about distributed "
    "database replication, consensus terms, write-ahead logs, storage compaction, "
    "index maintenance, network partitions, quorum reads, leader elections, "
    "recovery checkpoints, snapshot installation, backup verification, schema "
    "migration, and capacity planning. It intentionally uses a different opening "
    "and vocabulary so it shares no complete token page with the chat assistant "
    "system prompt. "
)


@dataclass
class ScenarioResult:
    hit_ms: float
    miss_ms: float
    hit_matched: list[int]
    miss_matched: list[int]
    evicted_match: int
    evictions: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--shared-tokens",
        type=int,
        default=4096,
        help="token count in the shared chat system prompt",
    )
    parser.add_argument(
        "--user-tail-tokens",
        type=int,
        default=256,
        help="distinct user-specific tokens appended after the shared prefix",
    )
    parser.add_argument("--concurrent-requests", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def build_model(model_id: str) -> LlamaModel:
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(model_id)
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    return model.eval()


def repeat_tokens(tokenizer: Tokenizer, text: str, count: int) -> list[int]:
    if count <= 0:
        return []
    body = tokenizer.encode(text, add_bos=False)
    if not body:
        raise RuntimeError("benchmark text produced no tokens")
    copies = (count + len(body) - 1) // len(body)
    return (body * copies)[:count]


def build_prompts(
    tokenizer: Tokenizer,
    *,
    shared_tokens: int,
    tail_tokens: int,
    concurrent_requests: int,
) -> tuple[list[int], list[list[int]]]:
    system_prefix = [tokenizer.bos_id] + repeat_tokens(
        tokenizer, SYSTEM_TEXT, shared_tokens - 1
    )
    seed = system_prefix + repeat_tokens(tokenizer, SEED_QUERY, tail_tokens)
    users = [
        system_prefix
        + repeat_tokens(
            tokenizer,
            f"User {index + 1}: {USER_QUERIES[index % len(USER_QUERIES)]}",
            tail_tokens,
        )
        for index in range(concurrent_requests)
    ]
    return seed, users


def build_cold_prompt(tokenizer: Tokenizer, token_count: int) -> list[int]:
    return [tokenizer.bos_id] + repeat_tokens(tokenizer, COLD_TEXT, token_count - 1)


def timed_prefill(
    engine: InferenceEngine,
    prompts: list[list[int]],
) -> tuple[float, list]:
    requests = [engine.add_request(prompt, max_new_tokens=1) for prompt in prompts]
    torch.cuda.synchronize()
    start = time.perf_counter()
    emitted = engine.step()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if len(emitted) != len(requests) or not all(req.reached_limit() for req in requests):
        raise RuntimeError("all one-token requests must finish in one prefill step")
    return elapsed_ms, requests


def run_scenario(
    engine: InferenceEngine,
    *,
    seed_prompt: list[int],
    user_prompts: list[list[int]],
    cold_prompt: list[int],
    shared_tokens: int,
) -> ScenarioResult:
    engine.reset()

    _, seed_requests = timed_prefill(engine, [seed_prompt])
    if seed_requests[0].cached_prefix_len != 0:
        raise AssertionError("the seed request must populate a cold cache")

    hit_ms, hit_requests = timed_prefill(engine, user_prompts)
    hit_matched = [req.cached_prefix_len for req in hit_requests]
    if hit_matched != [shared_tokens] * len(user_prompts):
        raise AssertionError(
            f"expected every concurrent request to reuse {shared_tokens} tokens, "
            f"got {hit_matched}"
        )
    hit_tokens = [req.generated for req in hit_requests]

    _, cold_requests = timed_prefill(engine, [cold_prompt])
    if cold_requests[0].cached_prefix_len != 0:
        raise AssertionError("the unrelated eviction request unexpectedly hit the cache")

    evicted_match = engine.prefix_cache.match(user_prompts[0]).token_count
    if evicted_match != 0:
        raise AssertionError(
            f"shared system prefix survived forced eviction ({evicted_match} tokens remain)"
        )
    evictions = engine.prefix_cache.stats()["evictions"]

    miss_ms, miss_requests = timed_prefill(engine, user_prompts)
    miss_matched = [req.cached_prefix_len for req in miss_requests]
    if miss_matched != [0] * len(user_prompts):
        raise AssertionError(
            f"expected every post-eviction request to miss, got {miss_matched}"
        )
    miss_tokens = [req.generated for req in miss_requests]
    if miss_tokens != hit_tokens:
        raise AssertionError("cache hit and recomputed miss produced different greedy tokens")

    return ScenarioResult(
        hit_ms=hit_ms,
        miss_ms=miss_ms,
        hit_matched=hit_matched,
        miss_matched=miss_matched,
        evicted_match=evicted_match,
        evictions=evictions,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(
        args.shared_tokens,
        args.user_tail_tokens,
        args.concurrent_requests,
        args.block_size,
        args.repetitions,
    ) <= 0:
        raise ValueError("all numeric arguments must be positive")
    if args.shared_tokens % args.block_size != 0:
        raise ValueError("shared-tokens must be a multiple of block-size")
    if args.user_tail_tokens % args.block_size != 0:
        raise ValueError("user-tail-tokens must be a multiple of block-size")

    tokenizer = Tokenizer.from_pretrained(args.model_id)
    model = build_model(args.model_id)
    seed_prompt, user_prompts = build_prompts(
        tokenizer,
        shared_tokens=args.shared_tokens,
        tail_tokens=args.user_tail_tokens,
        concurrent_requests=args.concurrent_requests,
    )

    shared_blocks = args.shared_tokens // args.block_size
    tail_blocks = args.user_tail_tokens // args.block_size
    # This upper bound holds the shared path plus the seed and every user tail.
    # A cold path of exactly this many newer blocks deterministically makes all
    # older leaves, including the system prompt, the LRU eviction victims.
    prefix_cache_blocks = shared_blocks + (args.concurrent_requests + 1) * tail_blocks
    cold_prompt = build_cold_prompt(
        tokenizer,
        prefix_cache_blocks * args.block_size,
    )
    max_seq_len = max(len(cold_prompt), len(seed_prompt)) + 1
    engine = InferenceEngine(
        model=model,
        max_running=args.concurrent_requests,
        max_seq_len=max_seq_len,
        block_size=args.block_size,
        token_budget=args.concurrent_requests * max_seq_len,
        eos_id=-1,
        temperature=0.0,
        warmup=False,
        use_cuda_graphs=False,
        use_prefix_cache=True,
        prefix_cache_blocks=prefix_cache_blocks,
    )

    # Prime the full-prefill and cached-suffix shapes before measured runs.
    run_scenario(
        engine,
        seed_prompt=seed_prompt,
        user_prompts=user_prompts,
        cold_prompt=cold_prompt,
        shared_tokens=args.shared_tokens,
    )

    results = [
        run_scenario(
            engine,
            seed_prompt=seed_prompt,
            user_prompts=user_prompts,
            cold_prompt=cold_prompt,
            shared_tokens=args.shared_tokens,
        )
        for _ in range(args.repetitions)
    ]
    hit_ms = statistics.median(result.hit_ms for result in results)
    miss_ms = statistics.median(result.miss_ms for result in results)
    evictions = int(statistics.median(result.evictions for result in results))

    print("\nConcurrent shared-system-prefix cache benchmark")
    print(f"  requests             : {args.concurrent_requests}")
    print(f"  prompt tokens/request: {len(user_prompts[0])}")
    print(f"  shared prefix        : {args.shared_tokens} tokens ({shared_blocks} blocks)")
    print(f"  prefix-cache capacity: {prefix_cache_blocks} blocks")
    print(f"  eviction prompt      : {len(cold_prompt)} tokens")
    print()
    print(
        f"  HIT batch            : {hit_ms:.3f} ms, "
        f"{results[0].hit_matched[0]} tokens reused/request"
    )
    print(
        f"  EVICTION             : {evictions} leaves evicted, "
        f"original prefix match={results[0].evicted_match} tokens"
    )
    print(
        f"  MISS batch           : {miss_ms:.3f} ms, "
        f"{results[0].miss_matched[0]} tokens reused/request"
    )
    print(f"  recompute / hit      : {miss_ms / hit_ms:.2f}x slower")
    print("  output parity        : PASS")


if __name__ == "__main__":
    main()
