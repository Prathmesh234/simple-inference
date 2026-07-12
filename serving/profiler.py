"""Opt-in torch.profiler capture for the live server worker."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule


def _enabled(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


class ServerProfiler:
    """Capture real continuous-batching steps driven by HTTP requests."""

    def __init__(self, engine):
        self.enabled = _enabled("SERVE_PROFILE")
        self.wait = int(os.environ.get("SERVE_PROFILE_WAIT", "1"))
        self.warmup = int(os.environ.get("SERVE_PROFILE_WARMUP", "1"))
        self.active = int(os.environ.get("SERVE_PROFILE_ACTIVE", "5"))
        self.delay = int(os.environ.get("SERVE_PROFILE_DELAY_STEPS", "0"))
        if min(self.wait, self.warmup, self.delay) < 0 or self.active < 1:
            raise ValueError(
                "SERVE_PROFILE_WAIT/WARMUP/DELAY_STEPS must be >= 0 and "
                "SERVE_PROFILE_ACTIVE must be >= 1"
            )

        default_out = (
            Path(__file__).resolve().parents[1]
            / "profiling"
            / "profile-engine"
            / "torch-profiler"
            / "out"
        )
        self.out_dir = Path(os.environ.get("SERVE_PROFILE_OUT", default_out))
        self.total_steps = self.wait + self.warmup + self.active
        self.engine = engine
        self.prof = None
        self.observed_steps = 0
        self.steps = 0
        self.written = False
        self.completed = False

    def start(self) -> None:
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[server-profiler] armed — "
            f"delay={self.delay}, wait={self.wait}, warmup={self.warmup}, "
            f"active={self.active} engine steps"
        )
        if self.delay == 0:
            self._start_capture()

    def before_step(self) -> None:
        if (
            self.enabled
            and not self.completed
            and self.prof is None
            and self.observed_steps >= self.delay
        ):
            self._start_capture()

    def _start_capture(self) -> None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self.prof = profile(
            activities=activities,
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=1,
            ),
            on_trace_ready=self._write,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        )
        self.prof.__enter__()
        print("[server-profiler] capture started")

    def step(self) -> None:
        if self.prof is None:
            self.observed_steps += 1
            return
        self.prof.step()
        self.steps += 1
        if self.steps >= self.total_steps:
            self.stop()

    def stop(self) -> None:
        if self.prof is None:
            return
        prof = self.prof
        self.prof = None
        prof.__exit__(None, None, None)
        self.completed = True

    def _write(self, prof) -> None:
        if self.written:
            return
        self.written = True
        trace_path = self.out_dir / "server_engine_trace.json"
        report_path = self.out_dir / "profiler_server_engine.txt"

        prof.export_chrome_trace(str(trace_path))
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=100)
        report = (
            "LIVE SERVER ENGINE PROFILE\n"
            f"max_running={self.engine.max_running}\n"
            f"max_seq_len={self.engine.max_seq_len}\n"
            f"token_budget={self.engine.scheduler.token_budget}\n"
            f"cuda_graphs={self.engine.use_cuda_graphs}\n"
            f"profile_steps={self.active}\n\n"
            f"{table}\n"
        )
        report_path.write_text(report)
        print(f"[server-profiler] report: {report_path}")
        print(f"[server-profiler] trace : {trace_path}")
