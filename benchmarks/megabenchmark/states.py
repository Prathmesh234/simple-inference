"""Optimization states exercised by the megabenchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


BASE_ENV = {
    "USE_TRITON": "false",
    "USE_AUTOTUNE": "false",
    "FUSE": "false",
    "USE_COMPILE": "false",
    "COMPILE_MODE": "default",
    "USE_CUDA_GRAPHS": "false",
    "SERVE_USE_CUDA_GRAPHS": "false",
    "SERVE_USE_PREFIX_CACHE": "false",
}


def _env(**overrides: str) -> dict[str, str]:
    values = dict(BASE_ENV)
    values.update(overrides)
    return values


@dataclass(frozen=True)
class StateSpec:
    id: str
    label: str
    description: str
    kind: str
    env: dict[str, str]
    optimizations: tuple[str, ...]
    engine_module: str | None = None
    engine_class: str = "InferenceEngine"
    engine_kwargs: dict[str, object] = field(default_factory=dict)
    paged: bool = False
    prefix_scenario: bool = False
    core: bool = True
    workload_ids: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["optimizations"] = list(self.optimizations)
        if self.workload_ids is not None:
            result["workload_ids"] = list(self.workload_ids)
        return result


STATES = (
    StateSpec(
        id="00_naive_no_kv_pytorch",
        label="Naive PyTorch, no KV cache",
        description="Recomputes the full growing sequence for every decode token.",
        kind="single_no_kv",
        env=_env(),
        optimizations=(),
        workload_ids=("single_request",),
    ),
    StateSpec(
        id="01_static_kv_pytorch",
        label="Static KV cache, PyTorch",
        description="Prefills once and reuses contiguous K/V during greedy decode.",
        kind="single_kv",
        env=_env(),
        optimizations=("static KV cache",),
    ),
    StateSpec(
        id="02_static_kv_triton_first_config",
        label="Static KV + Triton",
        description="Uses the existing Triton kernels without autotuning or fused transpose.",
        kind="single_kv",
        env=_env(USE_TRITON="true"),
        optimizations=("static KV cache", "Triton kernels"),
    ),
    StateSpec(
        id="03_static_kv_triton_autotuned",
        label="Static KV + autotuned Triton",
        description="Enables the existing Triton autotuning configurations.",
        kind="single_kv",
        env=_env(USE_TRITON="true", USE_AUTOTUNE="true"),
        optimizations=("static KV cache", "Triton kernels", "Triton autotuning"),
    ),
    StateSpec(
        id="04_static_kv_triton_fused",
        label="Static KV + autotuned/fused Triton",
        description="Also enables the existing fused attention transpose path.",
        kind="single_kv",
        env=_env(
            USE_TRITON="true",
            USE_AUTOTUNE="true",
            FUSE="true",
        ),
        optimizations=(
            "static KV cache",
            "Triton kernels",
            "Triton autotuning",
            "fused attention transpose",
        ),
    ),
    StateSpec(
        id="05_contiguous_continuous_batching",
        label="Contiguous continuous batching",
        description="Adds iteration-level scheduling and dynamic request admission.",
        kind="continuous",
        env=_env(
            USE_TRITON="true",
            USE_AUTOTUNE="true",
            FUSE="true",
        ),
        optimizations=(
            "static KV cache",
            "optimized Triton kernels",
            "continuous batching",
        ),
        engine_module="iterations.inference_01_contiguous_eager.engine",
    ),
    StateSpec(
        id="06_contiguous_cuda_graphs",
        label="Contiguous CUDA graphs",
        description="Replays contiguous batched decode with the preserved graph milestone.",
        kind="continuous",
        env=_env(
            USE_TRITON="true",
            USE_AUTOTUNE="true",
            FUSE="true",
            USE_CUDA_GRAPHS="true",
        ),
        optimizations=(
            "static KV cache",
            "optimized Triton kernels",
            "continuous batching",
            "CUDA graphs",
        ),
        engine_module="iterations.inference_02_contiguous_cuda_graphs.engine",
        engine_kwargs={"use_cuda_graphs": True},
    ),
    StateSpec(
        id="07_paged_gathered",
        label="Paged KV + gathered SDPA",
        description="Uses physical pages and block tables, then gathers K/V for SDPA.",
        kind="continuous",
        env=_env(
            USE_TRITON="true",
            USE_AUTOTUNE="true",
            FUSE="true",
        ),
        optimizations=(
            "optimized Triton kernels",
            "continuous batching",
            "paged KV allocation",
            "gathered SDPA decode",
        ),
        engine_module="iterations.inference_03_paged_gathered.engine",
        paged=True,
    ),
    StateSpec(
        id="08_paged_direct_triton_eager",
        label="Direct paged Triton, eager",
        description="Reads page tables directly in the production Triton decode kernel.",
        kind="continuous",
        env=_env(
            USE_TRITON="true",
            USE_AUTOTUNE="true",
            FUSE="true",
        ),
        optimizations=(
            "optimized Triton kernels",
            "continuous batching",
            "paged KV allocation",
            "direct paged Triton decode",
        ),
        engine_module="serving.engine",
        engine_kwargs={
            "use_triton": True,
            "use_cuda_graphs": False,
            "use_prefix_cache": False,
        },
        paged=True,
    ),
    StateSpec(
        id="09_paged_direct_triton_cuda_graphs",
        label="Direct paged Triton + CUDA graphs",
        description="Replays the same direct paged Triton forward through CUDA graphs.",
        kind="continuous",
        env=_env(
            USE_TRITON="true",
            USE_AUTOTUNE="true",
            FUSE="true",
            USE_CUDA_GRAPHS="true",
            SERVE_USE_CUDA_GRAPHS="true",
        ),
        optimizations=(
            "optimized Triton kernels",
            "continuous batching",
            "paged KV allocation",
            "direct paged Triton decode",
            "CUDA graphs",
        ),
        engine_module="serving.engine",
        engine_kwargs={
            "use_triton": True,
            "use_cuda_graphs": True,
            "use_prefix_cache": False,
        },
        paged=True,
    ),
    StateSpec(
        id="10_prefix_cache",
        label="Paged graphs + radix prefix cache",
        description="Adds block-aligned longest-prefix matching, sharing, and LRU eviction.",
        kind="continuous",
        env=_env(
            USE_TRITON="true",
            USE_AUTOTUNE="true",
            FUSE="true",
            USE_CUDA_GRAPHS="true",
            SERVE_USE_CUDA_GRAPHS="true",
            SERVE_USE_PREFIX_CACHE="true",
        ),
        optimizations=(
            "optimized Triton kernels",
            "continuous batching",
            "paged KV allocation",
            "direct paged Triton decode",
            "CUDA graphs",
            "radix prefix caching",
        ),
        engine_module="serving.engine",
        engine_kwargs={
            "use_triton": True,
            "use_cuda_graphs": True,
            "use_prefix_cache": True,
        },
        paged=True,
        prefix_scenario=True,
    ),
    StateSpec(
        id="alt_torch_compile_pytorch_kv",
        label="Alternative: torch.compile + PyTorch KV",
        description="Existing Inductor alternative for the pure-PyTorch static-KV path.",
        kind="single_kv",
        env=_env(
            USE_COMPILE="true",
            COMPILE_MODE="reduce-overhead",
        ),
        optimizations=("static KV cache", "torch.compile reduce-overhead"),
        core=False,
    ),
)

STATE_BY_ID = {state.id: state for state in STATES}


def select_states(selector: str) -> list[StateSpec]:
    if selector == "core":
        return [state for state in STATES if state.core]
    if selector == "all":
        return list(STATES)
    ids = [item.strip() for item in selector.split(",") if item.strip()]
    unknown = [state_id for state_id in ids if state_id not in STATE_BY_ID]
    if unknown:
        raise ValueError(f"unknown states: {', '.join(unknown)}")
    return [STATE_BY_ID[state_id] for state_id in ids]
