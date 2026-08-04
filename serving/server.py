"""
FastAPI inference server — wraps the continuous-batching engine (Section 15).

Production shape (vLLM-style)
-----------------------------
A single background WORKER THREAD owns the model, KV cache and engine, and is the
only thread that touches the GPU. HTTP handlers never run model code — they just
encode the prompt, drop a job on a thread-safe queue, and block on a per-request
result queue. The worker continuously batches whatever requests are in flight:
every iteration it admits new submissions, runs one engine.step() (batched
prefill + batched ragged decode), and streams the emitted tokens back to each
request's queue. So N concurrent HTTP requests are served as ONE rolling batch.

Startup order (exactly what was asked)
--------------------------------------
The FastAPI lifespan does, before any endpoint accepts traffic:
  1. load the tokenizer + model weights onto the GPU,
  2. build the engine and run a warmup pass (dummy batched prefill + decode) so
     all kernels/allocations are primed — the first real request is hot,
  3. start the worker thread and only THEN yield, opening the endpoints.

Run
---
    XDG_CONFIG_HOME=~/.cache/xdgconfig UV_CACHE_DIR=~/.cache/uv PATH=~/.local/bin:$PATH \
    uv run uvicorn serving.server:app --host 0.0.0.0 --port 8000

  PagedAttention and CUDA-graph decode are enabled by default. Set
  `SERVE_USE_CUDA_GRAPHS=false` to run the same direct paged Triton forward
  eagerly instead of through graph replay.

  (single worker process only — the model is loaded once in-process; do not run
   uvicorn with --workers > 1.)

Endpoints
---------
  GET  /health           liveness + engine config + live queue depths
  POST /generate         {prompt, max_new_tokens?} -> full completion (blocking)
  POST /generate/stream  same body -> text/plain stream of tokens as generated
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import env_loader  # noqa: F401  loads .env (HF_TOKEN, USE_* toggles)
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from torch.profiler import record_function

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from serving.engine import InferenceEngine
from serving.profiler import ServerProfiler
from serving.request import Request, RequestState
from tokenizer import Tokenizer

# ── server configuration (env-overridable) ─────────────────────────────────


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name, "")
    return int(value) if value else None


@dataclass(frozen=True)
class ServerConfig:
    model_id: str = os.environ.get(
        "SERVE_MODEL_ID",
        "meta-llama/Llama-3.2-3B",
    )
    device: str = os.environ.get("SERVE_DEVICE", "cuda")
    max_running: int = int(os.environ.get("SERVE_MAX_RUNNING", "8"))
    max_seq_len: int = int(os.environ.get("SERVE_MAX_SEQ_LEN", "4096"))
    block_size: int = int(os.environ.get("SERVE_PAGED_BLOCK_SIZE", "16"))
    num_kv_blocks: int | None = _env_optional_int("SERVE_KV_BLOCKS")
    token_budget: int | None = _env_optional_int("SERVE_TOKEN_BUDGET")
    temperature: float = float(os.environ.get("SERVE_TEMPERATURE", "0.7"))
    top_k: int = int(os.environ.get("SERVE_TOP_K", "50"))
    top_p: float = float(os.environ.get("SERVE_TOP_P", "0.9"))
    default_max_new: int = int(os.environ.get("SERVE_DEFAULT_MAX_NEW", "128"))
    use_cuda_graphs: bool = _env_bool("SERVE_USE_CUDA_GRAPHS", "true")
    use_prefix_cache: bool = _env_bool("SERVE_USE_PREFIX_CACHE", "true")
    prefix_cache_blocks: int | None = _env_optional_int(
        "SERVE_PREFIX_CACHE_BLOCKS"
    )


CONFIG = ServerConfig()
DTYPE = torch.bfloat16


# ── worker job plumbing ─────────────────────────────────────────────────────

# Items the worker pushes onto a job's result queue.
TOKEN = "token"    # payload: int token id
DONE = "done"      # payload: finish reason ("stop" | "length")
ERROR = "error"    # payload: error message str


@dataclass
class _Job:
    """One in-flight HTTP request, handed to the worker thread."""
    prompt_ids: list[int]
    max_new_tokens: int
    out: "queue.Queue[tuple[str, object]]" = field(default_factory=queue.Queue)
    req: Request | None = None


class _Worker:
    """Owns the engine and runs the single GPU-bound continuous-batching loop."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.profiler = ServerProfiler(engine)
        self.submit_q: "queue.Queue[_Job]" = queue.Queue()
        self.tracked: dict[int, _Job] = {}   # req_id -> job (worker-thread only)
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="engine-worker",
        )
        self._future: Future | None = None

    def start(self) -> None:
        if self._future is not None:
            raise RuntimeError("engine worker already started")
        self._future = self._executor.submit(self._loop)

    def stop(self) -> None:
        self._stop.set()
        if self._future is None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            return
        try:
            self._future.result(timeout=10)
        except TimeoutError:
            print("[server] engine worker did not stop within 10 seconds")
            self._executor.shutdown(wait=False, cancel_futures=True)
        else:
            self._executor.shutdown(wait=True)

    def submit(self, job: _Job) -> None:
        self.submit_q.put(job)

    # ── the loop (runs only in the worker thread) ──────────────────────────

    def _admit(self, job: _Job) -> None:
        try:
            job.req = self.engine.add_request(job.prompt_ids, job.max_new_tokens)
            self.tracked[job.req.id] = job
        except Exception as e:  # bad request (e.g. too long) — fail just this job
            job.out.put((ERROR, str(e)))

    def _drain_submissions(self, block: bool) -> None:
        if block:
            try:
                self._admit(self.submit_q.get(timeout=0.1))
            except queue.Empty:
                return
        while True:
            try:
                self._admit(self.submit_q.get_nowait())
            except queue.Empty:
                return

    def _loop(self) -> None:
        self.profiler.start()
        try:
            while not self._stop.is_set():
                idle = not self.engine.has_work() and not self.tracked
                # When idle, block briefly for the first job; otherwise drain quickly.
                with record_function("server.admit_submissions"):
                    self._drain_submissions(block=idle)
                if not self.engine.has_work():
                    continue

                try:
                    self.profiler.before_step()
                    with record_function("server.engine_step"):
                        emitted = self.engine.step()
                except Exception as e:  # a forward-pass failure dooms the whole batch
                    for job in self.tracked.values():
                        job.out.put((ERROR, f"engine step failed: {e}"))
                    self.tracked.clear()
                    self.engine.reset()
                    continue

                with record_function("server.dispatch_outputs"):
                    for req_id, tok in emitted.items():
                        job = self.tracked.get(req_id)
                        if job is not None:
                            job.out.put((TOKEN, tok))

                    # Retire finished requests (state set inside engine.step()).
                    for req_id, job in list(self.tracked.items()):
                        if job.req is not None and job.req.state is RequestState.FINISHED:
                            reason = "stop" if job.req.eos_hit else "length"
                            job.out.put((DONE, reason))
                            self.tracked.pop(req_id)
                self.profiler.step()
        finally:
            self.profiler.stop()


# ── shared server state ─────────────────────────────────────────────────────

class _State:
    tokenizer: Tokenizer
    engine: InferenceEngine
    worker: _Worker


state = _State()


def _load() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to serve this model")
    print(f"[server] loading tokenizer + model: {CONFIG.model_id}")
    state.tokenizer = Tokenizer.from_pretrained(CONFIG.model_id)
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(CONFIG.model_id)
    model = LlamaModel(cfg, torch.device(CONFIG.device))
    model.load_weights(loader)
    model.to(CONFIG.device, DTYPE)
    model.eval()

    print(
        "[server] building engine "
        f"(max_running={CONFIG.max_running}, max_seq_len={CONFIG.max_seq_len}) "
        "+ warmup"
    )
    # warmup=True runs a dummy batched prefill+decode before we serve traffic.
    state.engine = InferenceEngine(
        model=model,
        max_running=CONFIG.max_running,
        max_seq_len=CONFIG.max_seq_len,
        block_size=CONFIG.block_size,
        num_kv_blocks=CONFIG.num_kv_blocks,
        token_budget=CONFIG.token_budget,
        temperature=CONFIG.temperature,
        top_k=CONFIG.top_k,
        top_p=CONFIG.top_p,
        warmup=True,
        use_cuda_graphs=CONFIG.use_cuda_graphs,
        use_prefix_cache=CONFIG.use_prefix_cache,
        prefix_cache_blocks=CONFIG.prefix_cache_blocks,
    )
    state.worker = _Worker(state.engine)
    state.worker.start()
    print("[server] ready — endpoints open")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    try:
        yield
    finally:
        state.worker.stop()


app = FastAPI(title="simple-inference server", lifespan=lifespan)


# ── request/response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=CONFIG.default_max_new, ge=1)


class GenerateResponse(BaseModel):
    request_id: int
    prompt: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str


def _prepare(body: GenerateRequest) -> _Job:
    """Encode + clamp, returning a submitted-ready job (raises HTTP 400 on bad input)."""
    ids = state.tokenizer.encode(body.prompt, add_bos=True)
    room = CONFIG.max_seq_len - len(ids)
    if room < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"prompt has {len(ids)} tokens; "
                f"max_seq_len is {CONFIG.max_seq_len}"
            ),
        )
    max_new = min(body.max_new_tokens, room)
    return _Job(prompt_ids=ids, max_new_tokens=max_new)


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    eng = getattr(state, "engine", None)
    ready = eng is not None
    sched = eng.scheduler if ready else None
    return {
        "status": "ok" if ready else "loading",
        "model_id": CONFIG.model_id,
        "max_running": CONFIG.max_running,
        "max_seq_len": CONFIG.max_seq_len,
        "paged_attention": True,
        "cuda_graphs": (
            eng.cuda_graphs_active if ready else CONFIG.use_cuda_graphs
        ),
        "decode_backend": eng.decode_backend if ready else None,
        "prefix_cache": eng.prefix_cache.stats() if ready and eng.prefix_cache else None,
        "paged_block_size": CONFIG.block_size,
        "kv_blocks": eng.num_kv_blocks if ready else CONFIG.num_kv_blocks,
        "sampling": {
            "temperature": CONFIG.temperature,
            "top_k": CONFIG.top_k,
            "top_p": CONFIG.top_p,
        },
        "running": len(sched.running) if sched else 0,
        "waiting": len(sched.waiting) if sched else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest) -> GenerateResponse:
    job = _prepare(body)
    state.worker.submit(job)

    tokens: list[int] = []
    finish_reason = "length"
    while True:
        kind, payload = job.out.get()
        if kind == TOKEN:
            tokens.append(int(payload))
        elif kind == DONE:
            finish_reason = str(payload)
            break
        elif kind == ERROR:
            raise HTTPException(status_code=400, detail=str(payload))

    text = state.tokenizer.decode(tokens, skip_special=True)
    return GenerateResponse(
        request_id=job.req.id if job.req else -1,
        prompt=body.prompt,
        text=text,
        prompt_tokens=len(job.prompt_ids),
        generated_tokens=len(tokens),
        finish_reason=finish_reason,
    )


@app.post("/generate/stream")
def generate_stream(body: GenerateRequest) -> StreamingResponse:
    job = _prepare(body)
    state.worker.submit(job)

    def token_stream():
        while True:
            kind, payload = job.out.get()
            if kind == TOKEN:
                yield state.tokenizer.decode([int(payload)], skip_special=True)
            elif kind == DONE:
                break
            elif kind == ERROR:
                yield f"\n[error] {payload}"
                break

    return StreamingResponse(token_stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SERVE_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVE_PORT", "8000")),
    )
