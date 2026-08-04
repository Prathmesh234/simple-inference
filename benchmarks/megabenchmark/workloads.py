"""Deterministic workloads shared by every compatible benchmark state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


PROMPT_TEXTS = (
    "Explain how an inference service schedules variable length language model requests while preserving low latency and high GPU utilization.",
    "Analyze a distributed system incident, separate observed facts from hypotheses, and propose reversible diagnostic steps in priority order.",
    "Describe how paged key value memory maps logical token positions to physical blocks and why this reduces fragmentation for mixed workloads.",
    "Compare eager kernel launches with CUDA graph replay for autoregressive decoding, including CPU launch overhead and shape bucketing.",
    "Explain radix prefix caching for a chat service where many users share a long system prompt but ask independent questions.",
    "Provide a detailed code review of a concurrent queue, focusing on ownership, synchronization, failure handling, and measurable correctness.",
    "Derive the memory footprint of grouped query attention caches as a function of layers, sequence length, KV heads, and head dimension.",
    "Design a production benchmark that reports time to first token, inter-token latency, throughput, memory use, and output parity.",
)

SYSTEM_TEXT = (
    "You are the shared assistant for an enterprise chat application. Protect "
    "user privacy, distinguish verified facts from assumptions, preserve audit "
    "trails, prefer reversible troubleshooting steps, and never invent account "
    "state. Give concise but technically complete production guidance. For "
    "multi-step procedures, state prerequisites first, order actions by risk, "
    "and explain how to verify the result. Account for authentication, regional "
    "failover, message ordering, retention, legal holds, backups, attachments, "
    "notifications, channels, direct messages, threads, integrations, and "
    "administrator permissions. "
)

PREFIX_TAILS = (
    "Help a user recover access after replacing a phone without losing conversation history.",
    "Summarize an incident channel while preserving links and separating facts from speculation.",
    "Diagnose a large attachment upload that failed near completion without creating duplicates.",
    "Explain how encryption, retention, legal holds, and backup deletion interact.",
    "Compare muting, archiving, and leaving a channel for an incident-response workflow.",
    "Diagnose delayed mobile notifications when desktop delivery is immediate.",
    "Plan a scoped compliance export with integrity checks and chain of custody.",
    "Design channel permissions for a company with employees and external partners.",
)

COLD_TEXT = (
    "This unrelated eviction document covers database replication, quorum "
    "protocols, write ahead logging, compaction, index maintenance, leader "
    "election, snapshot installation, schema migration, backup verification, "
    "network partitions, capacity planning, storage engines, and recovery. "
)


@dataclass(frozen=True)
class Workload:
    id: str
    label: str
    description: str
    prompt_lengths: tuple[int, ...]
    max_new_tokens: int
    max_running: int
    single_stream: bool = False

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["prompt_lengths"] = list(self.prompt_lengths)
        return result


@dataclass(frozen=True)
class PrefixWorkload:
    shared_tokens: int
    tail_tokens: int
    concurrent_requests: int
    block_size: int
    repetitions: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def profile_workloads(profile: str) -> tuple[list[Workload], PrefixWorkload]:
    if profile == "quick":
        return (
            [
                Workload(
                    "single_request",
                    "Single request",
                    "Foundational latency and parity workload.",
                    (128,),
                    32,
                    1,
                    True,
                ),
                Workload(
                    "batch8",
                    "Eight concurrent requests",
                    "Equal-length throughput workload.",
                    (256,) * 8,
                    32,
                    8,
                    True,
                ),
                Workload(
                    "continuous_mixed",
                    "Mixed continuous batch",
                    "Queueing and admission with heterogeneous prompt lengths.",
                    (64, 128, 256, 512) * 3,
                    32,
                    4,
                ),
                Workload(
                    "decode_heavy",
                    "Decode-heavy batch",
                    "Measures steady-state decode throughput and graph replay.",
                    (128,) * 8,
                    96,
                    8,
                    True,
                ),
            ],
            PrefixWorkload(2048, 512, 16, 16, 1),
        )

    if profile == "full":
        return (
            [
                Workload(
                    "single_request",
                    "Single request",
                    "Longer foundational latency and parity workload.",
                    (512,),
                    128,
                    1,
                    True,
                ),
                Workload(
                    "batch8",
                    "Eight concurrent requests",
                    "Equal-length serving throughput workload.",
                    (1024,) * 8,
                    64,
                    8,
                    True,
                ),
                Workload(
                    "continuous_mixed",
                    "Mixed continuous batch",
                    "Thirty-two requests admitted through eight running slots.",
                    (128, 256, 512, 1024, 2048, 768, 384, 1536) * 4,
                    64,
                    8,
                ),
                Workload(
                    "long_prefill",
                    "Long-prompt prefill",
                    "Compute-heavy long-context prefill workload.",
                    (4096,) * 4,
                    16,
                    4,
                ),
                Workload(
                    "decode_heavy",
                    "Decode-heavy batch",
                    "Long decode run for stable TPOT and CUDA-graph measurements.",
                    (256,) * 16,
                    256,
                    16,
                    True,
                ),
            ],
            PrefixWorkload(2048, 512, 16, 16, 3),
        )

    if profile == "stress":
        workloads, prefix = profile_workloads("full")
        workloads.append(
            Workload(
                "capacity_mixed",
                "Paged-capacity stress",
                "Sixty-four variable-length requests through thirty-two slots.",
                (
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    768,
                    1536,
                )
                * 8,
                128,
                32,
            )
        )
        return workloads, PrefixWorkload(4096, 512, 32, 16, 3)

    raise ValueError(f"unknown profile: {profile}")


def repeat_tokens(tokenizer, text: str, count: int) -> list[int]:
    if count <= 0:
        return []
    body = tokenizer.encode(text, add_bos=False)
    if not body:
        raise RuntimeError("benchmark text produced no tokens")
    return (body * math.ceil(count / len(body)))[:count]


def build_prompts(tokenizer, workload: Workload) -> list[list[int]]:
    prompts: list[list[int]] = []
    for index, length in enumerate(workload.prompt_lengths):
        if length < 1:
            raise ValueError(f"prompt length must be positive, got {length}")
        text = f"Request {index + 1}. {PROMPT_TEXTS[index % len(PROMPT_TEXTS)]}"
        prompts.append(
            [tokenizer.bos_id] + repeat_tokens(tokenizer, text, length - 1)
        )
    return prompts


def build_prefix_prompts(
    tokenizer,
    workload: PrefixWorkload,
) -> tuple[list[int], list[list[int]], list[int], int]:
    shared = [tokenizer.bos_id] + repeat_tokens(
        tokenizer,
        SYSTEM_TEXT,
        workload.shared_tokens - 1,
    )
    seed = shared + repeat_tokens(
        tokenizer,
        "Seed the shared assistant cache with an architecture and reliability review.",
        workload.tail_tokens,
    )
    users = [
        shared
        + repeat_tokens(
            tokenizer,
            f"User {index + 1}. {PREFIX_TAILS[index % len(PREFIX_TAILS)]}",
            workload.tail_tokens,
        )
        for index in range(workload.concurrent_requests)
    ]
    shared_blocks = workload.shared_tokens // workload.block_size
    tail_blocks = workload.tail_tokens // workload.block_size
    cache_blocks = shared_blocks + (workload.concurrent_requests + 1) * tail_blocks
    cold = [tokenizer.bos_id] + repeat_tokens(
        tokenizer,
        COLD_TEXT,
        cache_blocks * workload.block_size - 1,
    )
    return seed, users, cold, cache_blocks
