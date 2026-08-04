"""Run one isolated megabenchmark state and write detailed JSON results."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.megabenchmark.states import STATE_BY_ID
from benchmarks.megabenchmark.workloads import (
    PrefixWorkload,
    Workload,
    build_prefix_prompts,
    build_prompts,
    profile_workloads,
)

MODEL_ID = "meta-llama/Llama-3.2-3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, choices=sorted(STATE_BY_ID))
    parser.add_argument("--profile", choices=("quick", "full", "stress"), default="full")
    parser.add_argument("--model-id", choices=(MODEL_ID,), default=MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def digest_outputs(outputs: list[list[int]]) -> str:
    payload = json.dumps(outputs, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def system_metadata(torch) -> dict[str, Any]:
    gpu: dict[str, Any] = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "multiprocessor_count": props.multi_processor_count,
        }
    smi = command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,pstate,temperature.gpu,power.limit,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "nvidia_smi": smi,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
    }


def memory_snapshot(torch) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def kv_snapshot(engine) -> dict[str, Any]:
    kv = getattr(engine, "kv", None)
    if kv is None:
        return {}
    result: dict[str, Any] = {
        "class": f"{type(kv).__module__}.{type(kv).__name__}",
    }
    for name in (
        "num_blocks",
        "total_blocks",
        "block_size",
        "num_scratch_blocks",
        "max_batch",
        "max_seq_len",
    ):
        value = getattr(kv, name, None)
        if value is not None:
            result[name] = value
    tensors = [
        tensor
        for tensor in (getattr(kv, "k_cache", None), getattr(kv, "v_cache", None))
        if tensor is not None
    ]
    result["storage_bytes"] = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors
    )
    allocator = getattr(kv, "allocator", None)
    if allocator is not None:
        result["allocator"] = {
            "free_blocks": allocator.free_blocks,
            "reserved_blocks": allocator.reserved_blocks,
            "available_reservation_blocks": allocator.available_reservation_blocks,
        }
    prefix_cache = getattr(engine, "prefix_cache", None)
    if prefix_cache is not None:
        result["prefix_cache"] = prefix_cache.stats()
    return result


def engine_snapshot(engine) -> dict[str, Any]:
    result = {
        "class": f"{type(engine).__module__}.{type(engine).__name__}",
        "max_running": engine.max_running,
        "max_seq_len": engine.max_seq_len,
        "decode_backend": getattr(engine, "decode_backend", None),
        "cuda_graphs_active": getattr(engine, "cuda_graphs_active", None),
        "use_cuda_graphs": getattr(engine, "use_cuda_graphs", None),
        "use_triton": getattr(engine, "use_triton", None),
        "use_prefix_cache": getattr(engine, "use_prefix_cache", False),
    }
    result["kv"] = kv_snapshot(engine)
    return result


def per_request_results(requests, starts: dict[int, float], emissions: dict[int, list[float]], tokenizer) -> list[dict[str, Any]]:
    results = []
    for index, request in enumerate(requests):
        times = emissions[request.id]
        intervals = [
            (right - left) * 1000.0 for left, right in zip(times, times[1:])
        ]
        output = list(request.generated)
        results.append(
            {
                "index": index,
                "request_id": request.id,
                "prompt_tokens": request.prompt_len,
                "generated_tokens": len(output),
                "cached_prefix_tokens": getattr(request, "cached_prefix_len", 0),
                "ttft_ms": (times[0] - starts[request.id]) * 1000.0,
                "e2e_ms": (times[-1] - starts[request.id]) * 1000.0,
                "inter_token_ms": distribution(intervals),
                "output_token_ids": output,
                "output_sha256": digest_outputs([output]),
                "output_text": tokenizer.decode(output)[:500],
            }
        )
    return results


def summarize_run(
    workload: Workload,
    requests,
    starts: dict[int, float],
    emissions: dict[int, list[float]],
    step_trace: list[dict[str, Any]],
    wall_ms: float,
    memory: dict[str, int],
    engine,
    tokenizer,
) -> dict[str, Any]:
    request_results = per_request_results(requests, starts, emissions, tokenizer)
    ttft = [item["ttft_ms"] for item in request_results]
    e2e = [item["e2e_ms"] for item in request_results]
    itl = [
        interval
        for request in requests
        for interval in [
            (right - left) * 1000.0
            for left, right in zip(emissions[request.id], emissions[request.id][1:])
        ]
    ]
    total_prompt = sum(request.prompt_len for request in requests)
    total_generated = sum(request.num_generated for request in requests)
    prefill_ms = sum(
        step["latency_ms"] for step in step_trace if step["first_token_emissions"]
    )
    decode_ms = sum(
        step["latency_ms"] for step in step_trace if step["decode_emissions"]
    )
    decode_tokens = max(0, total_generated - len(requests))
    outputs = [list(request.generated) for request in requests]
    return {
        "id": workload.id,
        "label": workload.label,
        "description": workload.description,
        "status": "ok",
        "request_count": len(requests),
        "max_running": workload.max_running,
        "prompt_tokens": {
            "total": total_prompt,
            "min": min(request.prompt_len for request in requests),
            "mean": statistics.fmean(request.prompt_len for request in requests),
            "max": max(request.prompt_len for request in requests),
        },
        "generated_tokens": total_generated,
        "wall_ms": wall_ms,
        "request_per_second": len(requests) / (wall_ms / 1000.0),
        "output_tokens_per_second": total_generated / (wall_ms / 1000.0),
        "all_tokens_per_second": (total_prompt + total_generated) / (wall_ms / 1000.0),
        "ttft_ms": distribution(ttft),
        "e2e_ms": distribution(e2e),
        "inter_token_ms": distribution(itl),
        "prefill_bearing_step_ms": prefill_ms,
        "first_tokens_per_second": len(requests) / (prefill_ms / 1000.0),
        "decode_bearing_step_ms": decode_ms,
        "steady_decode_tokens_per_second": (
            decode_tokens / (decode_ms / 1000.0) if decode_ms else None
        ),
        "step_count": len(step_trace),
        "step_latency_ms": distribution([step["latency_ms"] for step in step_trace]),
        "step_trace": step_trace,
        "memory": memory,
        "engine": engine_snapshot(engine),
        "output_sha256": digest_outputs(outputs),
        "requests": request_results,
    }


def warm_continuous_engine(torch, engine, prompts: list[list[int]]) -> None:
    warm_requests = [
        engine.add_request(prompt, max_new_tokens=2) for prompt in prompts
    ]
    while not all(request.reached_limit() for request in warm_requests):
        engine.step()
    while engine.has_work():
        engine.step()
    torch.cuda.synchronize()
    engine.reset()


def paged_block_count(prompts: list[list[int]], max_new_tokens: int, max_running: int, block_size: int) -> int:
    per_request = sorted(
        (
            math.ceil((len(prompt) + max_new_tokens) / block_size)
            for prompt in prompts
        ),
        reverse=True,
    )
    return sum(per_request[:max_running])


def build_continuous_engine(state, model, workload: Workload, prompts: list[list[int]]):
    module = importlib.import_module(state.engine_module)
    engine_cls = getattr(module, state.engine_class)
    max_seq_len = max(len(prompt) for prompt in prompts) + workload.max_new_tokens
    kwargs: dict[str, Any] = {
        "model": model,
        "max_running": workload.max_running,
        "max_seq_len": max_seq_len,
        "token_budget": workload.max_running * max_seq_len,
        "eos_id": -1,
        "temperature": 0.0,
        "warmup": True,
        **state.engine_kwargs,
    }
    if state.paged:
        block_size = 16
        kwargs["block_size"] = block_size
        kwargs["num_kv_blocks"] = paged_block_count(
            prompts,
            workload.max_new_tokens,
            workload.max_running,
            block_size,
        )
    return engine_cls(**kwargs)


def run_continuous_workload(torch, state, model, tokenizer, workload: Workload) -> dict[str, Any]:
    prompts = build_prompts(tokenizer, workload)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    init_start = time.perf_counter()
    engine = build_continuous_engine(state, model, workload, prompts)
    torch.cuda.synchronize()
    engine_init_ms = (time.perf_counter() - init_start) * 1000.0
    engine_memory = memory_snapshot(torch)

    warm_continuous_engine(torch, engine, prompts)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    requests = [
        engine.add_request(prompt, workload.max_new_tokens) for prompt in prompts
    ]
    run_start = time.perf_counter()
    starts = {request.id: run_start for request in requests}
    emissions = {request.id: [] for request in requests}
    step_trace: list[dict[str, Any]] = []

    while not all(request.reached_limit() for request in requests):
        waiting_before = len(engine.scheduler.waiting)
        running_before = len(engine.scheduler.running)
        generated_before = {
            request.id: request.num_generated for request in requests
        }
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        emitted = engine.step()
        torch.cuda.synchronize()
        step_end = time.perf_counter()
        first_count = 0
        decode_count = 0
        for request_id in emitted:
            emissions[request_id].append(step_end)
            if generated_before[request_id] == 0:
                first_count += 1
            else:
                decode_count += 1
        step_trace.append(
            {
                "index": len(step_trace),
                "latency_ms": (step_end - step_start) * 1000.0,
                "waiting_before": waiting_before,
                "running_before": running_before,
                "waiting_after": len(engine.scheduler.waiting),
                "running_after": len(engine.scheduler.running),
                "emitted_tokens": len(emitted),
                "first_token_emissions": first_count,
                "decode_emissions": decode_count,
            }
        )

    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - run_start) * 1000.0
    measured_memory = memory_snapshot(torch)
    while engine.has_work():
        engine.step()
    torch.cuda.synchronize()

    result = summarize_run(
        workload,
        requests,
        starts,
        emissions,
        step_trace,
        wall_ms,
        measured_memory,
        engine,
        tokenizer,
    )
    result["engine_init_ms"] = engine_init_ms
    result["engine_init_memory"] = engine_memory
    del engine
    torch.cuda.empty_cache()
    return result


def single_forward_run(torch, model, prompts: list[list[int]], max_new_tokens: int, use_kv: bool):
    from model.kv_cache import KVCache

    lengths = {len(prompt) for prompt in prompts}
    if len(lengths) != 1:
        raise ValueError("static single-stream states require equal prompt lengths")
    prompt_tensor = torch.tensor(prompts, dtype=torch.long, device="cuda")
    batch_size, prompt_len = prompt_tensor.shape
    kv_cache = None
    if use_kv:
        cfg = model.cfg
        kv_cache = KVCache(
            n_layers=cfg.num_hidden_layers,
            max_batch=batch_size,
            max_seq_len=prompt_len + max_new_tokens,
            n_heads_kv=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            dtype=next(model.parameters()).dtype,
            device="cuda",
        )

    torch.cuda.synchronize()
    start = time.perf_counter()
    step_start = start
    logits = model(prompt_tensor, start_pos=0, kv_cache=kv_cache)
    next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    now = time.perf_counter()
    generated = [[int(token)] for token in next_tokens[:, 0].tolist()]
    emission_times = [[now] for _ in range(batch_size)]
    trace = [
        {
            "index": 0,
            "latency_ms": (now - step_start) * 1000.0,
            "emitted_tokens": batch_size,
            "first_token_emissions": batch_size,
            "decode_emissions": 0,
        }
    ]
    sequence = torch.cat((prompt_tensor, next_tokens), dim=1)
    position = prompt_len

    for step in range(1, max_new_tokens):
        step_start = time.perf_counter()
        if use_kv:
            logits = model.decode_step(next_tokens, position, kv_cache)
        else:
            logits = model(sequence, start_pos=0)
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        now = time.perf_counter()
        for index, token in enumerate(next_tokens[:, 0].tolist()):
            generated[index].append(int(token))
            emission_times[index].append(now)
        trace.append(
            {
                "index": step,
                "latency_ms": (now - step_start) * 1000.0,
                "emitted_tokens": batch_size,
                "first_token_emissions": 0,
                "decode_emissions": batch_size,
            }
        )
        if not use_kv:
            sequence = torch.cat((sequence, next_tokens), dim=1)
        position += 1

    return {
        "start_time": start,
        "wall_ms": (time.perf_counter() - start) * 1000.0,
        "generated": generated,
        "emission_times": emission_times,
        "step_trace": trace,
        "kv_cache": kv_cache,
    }


def run_single_workload(torch, state, model, tokenizer, workload: Workload) -> dict[str, Any]:
    prompts = build_prompts(tokenizer, workload)
    use_kv = state.kind == "single_kv"
    single_forward_run(torch, model, prompts, 2, use_kv)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    raw = single_forward_run(
        torch,
        model,
        prompts,
        workload.max_new_tokens,
        use_kv,
    )
    memory = memory_snapshot(torch)

    output_requests = []
    ttft = []
    e2e = []
    intervals: list[float] = []
    for index, (prompt, output, times) in enumerate(
        zip(prompts, raw["generated"], raw["emission_times"])
    ):
        request_intervals = [
            (right - left) * 1000.0 for left, right in zip(times, times[1:])
        ]
        intervals.extend(request_intervals)
        ttft_ms = (times[0] - raw["start_time"]) * 1000.0
        e2e_ms = (times[-1] - raw["start_time"]) * 1000.0
        ttft.append(ttft_ms)
        e2e.append(e2e_ms)
        output_requests.append(
            {
                "index": index,
                "prompt_tokens": len(prompt),
                "generated_tokens": len(output),
                "cached_prefix_tokens": 0,
                "ttft_ms": ttft_ms,
                "e2e_ms": e2e_ms,
                "inter_token_ms": distribution(request_intervals),
                "output_token_ids": output,
                "output_sha256": digest_outputs([output]),
                "output_text": tokenizer.decode(output)[:500],
            }
        )

    total_prompt = sum(map(len, prompts))
    total_generated = sum(map(len, raw["generated"]))
    decode_ms = sum(item["latency_ms"] for item in raw["step_trace"][1:])
    result = {
        "id": workload.id,
        "label": workload.label,
        "description": workload.description,
        "status": "ok",
        "request_count": len(prompts),
        "max_running": len(prompts),
        "prompt_tokens": {
            "total": total_prompt,
            "min": min(map(len, prompts)),
            "mean": statistics.fmean(map(len, prompts)),
            "max": max(map(len, prompts)),
        },
        "generated_tokens": total_generated,
        "wall_ms": raw["wall_ms"],
        "request_per_second": len(prompts) / (raw["wall_ms"] / 1000.0),
        "output_tokens_per_second": total_generated / (raw["wall_ms"] / 1000.0),
        "all_tokens_per_second": (total_prompt + total_generated) / (raw["wall_ms"] / 1000.0),
        "ttft_ms": distribution(ttft),
        "e2e_ms": distribution(e2e),
        "inter_token_ms": distribution(intervals),
        "prefill_bearing_step_ms": raw["step_trace"][0]["latency_ms"],
        "first_tokens_per_second": len(prompts) / (raw["step_trace"][0]["latency_ms"] / 1000.0),
        "decode_bearing_step_ms": decode_ms,
        "steady_decode_tokens_per_second": (
            (total_generated - len(prompts)) / (decode_ms / 1000.0)
            if decode_ms
            else None
        ),
        "step_count": len(raw["step_trace"]),
        "step_latency_ms": distribution(
            [item["latency_ms"] for item in raw["step_trace"]]
        ),
        "step_trace": raw["step_trace"],
        "memory": memory,
        "engine": {
            "class": "model.llama.LlamaModel",
            "kv": {
                "class": (
                    f"{type(raw['kv_cache']).__module__}.{type(raw['kv_cache']).__name__}"
                    if raw["kv_cache"] is not None
                    else None
                ),
                "storage_bytes": (
                    sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in (
                            raw["kv_cache"].k_cache,
                            raw["kv_cache"].v_cache,
                        )
                    )
                    if raw["kv_cache"] is not None
                    else 0
                ),
            },
        },
        "output_sha256": digest_outputs(raw["generated"]),
        "requests": output_requests,
    }
    del raw
    torch.cuda.empty_cache()
    return result


def timed_prefix_phase(torch, engine, prompts: list[list[int]]) -> dict[str, Any]:
    requests = [engine.add_request(prompt, max_new_tokens=1) for prompt in prompts]
    torch.cuda.synchronize()
    start = time.perf_counter()
    emitted = engine.step()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if len(emitted) != len(requests):
        raise RuntimeError("prefix phase did not emit one token for every request")
    while engine.has_work():
        engine.step()
    torch.cuda.synchronize()
    return {
        "latency_ms": elapsed_ms,
        "matched_tokens": [
            getattr(request, "cached_prefix_len", 0) for request in requests
        ],
        "outputs": [list(request.generated) for request in requests],
    }


def one_prefix_repetition(torch, engine, seed, users, cold) -> dict[str, Any]:
    engine.reset()
    seed_result = timed_prefix_phase(torch, engine, [seed])
    hit_result = timed_prefix_phase(torch, engine, users)
    cold_result = timed_prefix_phase(torch, engine, [cold])
    prefix_cache = getattr(engine, "prefix_cache", None)
    evicted_match = (
        prefix_cache.match(users[0]).token_count if prefix_cache is not None else 0
    )
    miss_result = timed_prefix_phase(torch, engine, users)
    return {
        "seed": seed_result,
        "hit": hit_result,
        "eviction": {
            **cold_result,
            "original_prefix_match_tokens": evicted_match,
            "cache_stats": prefix_cache.stats() if prefix_cache is not None else None,
        },
        "miss": miss_result,
        "output_parity": hit_result["outputs"] == miss_result["outputs"],
    }


def run_prefix_scenario(torch, state, model, tokenizer, workload: PrefixWorkload) -> dict[str, Any]:
    if workload.shared_tokens % workload.block_size:
        raise ValueError("shared prefix must be block aligned")
    if workload.tail_tokens % workload.block_size:
        raise ValueError("prefix tail must be block aligned")
    seed, users, cold, cache_blocks = build_prefix_prompts(tokenizer, workload)
    module = importlib.import_module(state.engine_module)
    engine_cls = getattr(module, state.engine_class)
    max_seq_len = max(len(seed), len(cold)) + 1
    concurrent_blocks = workload.concurrent_requests * math.ceil(
        (len(users[0]) + 1) / workload.block_size
    )
    engine = engine_cls(
        model=model,
        max_running=workload.concurrent_requests,
        max_seq_len=max_seq_len,
        block_size=workload.block_size,
        num_kv_blocks=max(cache_blocks, concurrent_blocks),
        token_budget=workload.concurrent_requests * max_seq_len,
        eos_id=-1,
        temperature=0.0,
        warmup=True,
        prefix_cache_blocks=cache_blocks,
        **state.engine_kwargs,
    )

    one_prefix_repetition(torch, engine, seed, users, cold)
    repetitions = [
        one_prefix_repetition(torch, engine, seed, users, cold)
        for _ in range(workload.repetitions)
    ]
    hit_ms = [item["hit"]["latency_ms"] for item in repetitions]
    miss_ms = [item["miss"]["latency_ms"] for item in repetitions]
    result = {
        "status": "ok",
        "config": workload.to_dict(),
        "prompt_tokens_per_request": len(users[0]),
        "prefix_cache_blocks": cache_blocks,
        "eviction_prompt_tokens": len(cold),
        "hit_latency_ms": distribution(hit_ms),
        "miss_latency_ms": distribution(miss_ms),
        "recompute_over_hit": statistics.median(miss_ms) / statistics.median(hit_ms),
        "matched_tokens_hit": repetitions[0]["hit"]["matched_tokens"],
        "matched_tokens_miss": repetitions[0]["miss"]["matched_tokens"],
        "evicted_prefix_match_tokens": repetitions[0]["eviction"][
            "original_prefix_match_tokens"
        ],
        "cache_stats": repetitions[-1]["eviction"]["cache_stats"],
        "output_parity": all(item["output_parity"] for item in repetitions),
        "repetitions": repetitions,
        "engine": engine_snapshot(engine),
    }
    del engine
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    state = STATE_BY_ID[args.state]
    for key, value in state.env.items():
        os.environ[key] = value

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the megabenchmark")

    from config import ModelConfig
    from loader import WeightLoader
    from model.llama import LlamaModel
    from tokenizer import Tokenizer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    print(f"\n[{state.id}] {state.label}", flush=True)
    print(f"  flags: {state.env}", flush=True)

    metadata = system_metadata(torch)
    load_start = time.perf_counter()
    loader = WeightLoader.from_pretrained(args.model_id)
    tokenizer = Tokenizer(loader.model_dir)
    cfg = ModelConfig.llama_3_2_3b()
    model = LlamaModel(cfg, torch.device("cuda"))
    model.load_weights(loader)
    model.to("cuda", torch.bfloat16)
    model.eval()
    compiled = model.maybe_compile()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    model_memory = memory_snapshot(torch)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    workloads, prefix_workload = profile_workloads(args.profile)
    selected_workloads = [
        workload
        for workload in workloads
        if (
            (state.kind == "continuous" or workload.single_stream)
            and (
                state.workload_ids is None
                or workload.id in state.workload_ids
            )
        )
    ]
    workload_results = []
    for workload in selected_workloads:
        print(f"  running {workload.id}: {workload.label}", flush=True)
        if state.kind == "continuous":
            result = run_continuous_workload(
                torch, state, model, tokenizer, workload
            )
        else:
            result = run_single_workload(
                torch, state, model, tokenizer, workload
            )
        workload_results.append(result)
        print(
            f"    wall={result['wall_ms']:.3f} ms "
            f"ttft_p50={result['ttft_ms']['p50']:.3f} ms "
            f"output={result['output_tokens_per_second']:.1f} tok/s "
            f"peak={result['memory']['max_allocated_bytes'] / 1e9:.2f} GB",
            flush=True,
        )

    prefix_result = None
    if state.prefix_scenario:
        print("  running shared-prefix hit/eviction/miss scenario", flush=True)
        prefix_result = run_prefix_scenario(
            torch,
            state,
            model,
            tokenizer,
            prefix_workload,
        )
        print(
            f"    hit={prefix_result['hit_latency_ms']['p50']:.3f} ms "
            f"miss={prefix_result['miss_latency_ms']['p50']:.3f} ms "
            f"ratio={prefix_result['recompute_over_hit']:.2f}x",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "status": "ok",
        "state": state.to_dict(),
        "profile": args.profile,
        "model": {
            "id": args.model_id,
            "parameter_count": parameter_count,
            "parameter_bytes": parameter_bytes,
            "dtype": str(next(model.parameters()).dtype),
            "compiled": compiled,
            "load_ms": load_ms,
            "memory_after_load": model_memory,
        },
        "environment": metadata,
        "workload_definitions": [workload.to_dict() for workload in selected_workloads],
        "workloads": workload_results,
        "prefix_cache": prefix_result,
        "duration_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
