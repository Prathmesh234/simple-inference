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
    USE_CUDA_GRAPHS=false uv run uvicorn serving.server:app --host 0.0.0.0 --port 8000

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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import env_loader  # noqa: F401  loads .env (HF_TOKEN, USE_* toggles)
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from serving.engine import InferenceEngine
from serving.request import Request, RequestState
from tokenizer import Tokenizer

# ── server configuration (env-overridable) ─────────────────────────────────

MODEL_ID = os.environ.get("SERVE_MODEL_ID", "meta-llama/Llama-3.2-3B")
DEVICE = os.environ.get("SERVE_DEVICE", "cuda")
MAX_RUNNING = int(os.environ.get("SERVE_MAX_RUNNING", "8"))
MAX_SEQ_LEN = int(os.environ.get("SERVE_MAX_SEQ_LEN", "4096"))
_TB = os.environ.get("SERVE_TOKEN_BUDGET", "")
TOKEN_BUDGET = int(_TB) if _TB else None
TEMPERATURE = float(os.environ.get("SERVE_TEMPERATURE", "0.7"))
TOP_K = int(os.environ.get("SERVE_TOP_K", "50"))
TOP_P = float(os.environ.get("SERVE_TOP_P", "0.9"))
DEFAULT_MAX_NEW = int(os.environ.get("SERVE_DEFAULT_MAX_NEW", "128"))
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
    req: Optional[Request] = None


class _Worker:
    """Owns the engine and runs the single GPU-bound continuous-batching loop."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.submit_q: "queue.Queue[_Job]" = queue.Queue()
        self.tracked: dict[int, _Job] = {}   # req_id -> job (worker-thread only)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="engine-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

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
        while not self._stop.is_set():
            idle = not self.engine.has_work() and not self.tracked
            # When idle, block briefly for the first job; otherwise drain quickly.
            self._drain_submissions(block=idle)
            if not self.engine.has_work():
                continue

            try:
                emitted = self.engine.step()
            except Exception as e:  # a forward-pass failure dooms the whole batch
                for job in self.tracked.values():
                    job.out.put((ERROR, f"engine step failed: {e}"))
                self.tracked.clear()
                self.engine.reset()
                continue

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


# ── shared server state ─────────────────────────────────────────────────────

class _State:
    tokenizer: Tokenizer
    engine: InferenceEngine
    worker: _Worker


state = _State()


def _load() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to serve this model")
    print(f"[server] loading tokenizer + model: {MODEL_ID}")
    state.tokenizer = Tokenizer.from_pretrained(MODEL_ID)
    cfg = ModelConfig.llama_3_2_3b()
    loader = WeightLoader.from_pretrained(MODEL_ID)
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()

    print(f"[server] building engine (max_running={MAX_RUNNING}, max_seq_len={MAX_SEQ_LEN}) + warmup")
    # warmup=True runs a dummy batched prefill+decode before we serve traffic.
    state.engine = InferenceEngine(
        model=model,
        max_running=MAX_RUNNING,
        max_seq_len=MAX_SEQ_LEN,
        token_budget=TOKEN_BUDGET,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        warmup=True,
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
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW, ge=1)


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
    room = MAX_SEQ_LEN - len(ids)
    if room < 1:
        raise HTTPException(
            status_code=400,
            detail=f"prompt has {len(ids)} tokens; max_seq_len is {MAX_SEQ_LEN}",
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
        "model_id": MODEL_ID,
        "max_running": MAX_RUNNING,
        "max_seq_len": MAX_SEQ_LEN,
        "sampling": {"temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P},
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
