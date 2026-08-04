"""Orchestrate isolated megabenchmark states and generate aggregate reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.megabenchmark.report import generate_reports
from benchmarks.megabenchmark.states import STATES, select_states
from benchmarks.megabenchmark.workloads import profile_workloads


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
MODEL_ID = "meta-llama/Llama-3.2-3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full", "stress"), default="full")
    parser.add_argument(
        "--states",
        default="core",
        help="'core', 'all', or a comma-separated list of state IDs",
    )
    parser.add_argument("--model-id", choices=(MODEL_ID,), default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-states", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def print_states() -> None:
    for state in STATES:
        tier = "core" if state.core else "alternative"
        print(f"{state.id:<42} [{tier}] {state.label}")


def stream_process(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def failure_result(state, returncode: int, log_path: Path) -> dict:
    tail = ""
    if log_path.exists():
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
    return {
        "schema_version": 1,
        "status": "failed",
        "state": state.to_dict(),
        "error": f"worker exited with code {returncode}",
        "log_tail": tail,
    }


def main() -> None:
    args = parse_args()
    if args.list_states:
        print_states()
        return

    states = select_states(args.states)
    workloads, prefix = profile_workloads(args.profile)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = args.output_dir or DEFAULT_RESULTS_ROOT / timestamp
    results_dir = results_dir.resolve()

    print("Inference engine megabenchmark")
    print(f"  profile : {args.profile}")
    print(f"  states  : {len(states)}")
    print(f"  output  : {results_dir}")
    print("  matrix  :")
    for state in states:
        print(f"    {state.id}: {state.label}")
    print("  workloads:")
    for workload in workloads:
        print(
            f"    {workload.id}: requests={len(workload.prompt_lengths)}, "
            f"max_running={workload.max_running}, "
            f"prompt={min(workload.prompt_lengths)}..{max(workload.prompt_lengths)}, "
            f"decode={workload.max_new_tokens}"
        )
    if any(state.prefix_scenario for state in states):
        print(
            "    prefix_cache: "
            f"shared={prefix.shared_tokens}, tail={prefix.tail_tokens}, "
            f"requests={prefix.concurrent_requests}, repetitions={prefix.repetitions}"
        )
    if args.dry_run:
        return
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {results_dir}. "
            "Choose a new directory so results from separate runs cannot mix."
        )

    states_dir = results_dir / "states"
    logs_dir = results_dir / "logs"
    cache_dir = results_dir / "compiler-cache"
    states_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "model_id": args.model_id,
        "states": [state.to_dict() for state in states],
        "workloads": [workload.to_dict() for workload in workloads],
        "prefix_workload": prefix.to_dict(),
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for index, state in enumerate(states, start=1):
        print(f"\n{'=' * 100}")
        print(f"STATE {index}/{len(states)}: {state.id} — {state.label}")
        print(f"{'=' * 100}")
        output_path = states_dir / f"{state.id}.json"
        log_path = logs_dir / f"{state.id}.log"
        env = os.environ.copy()
        env.update(state.env)
        env["PYTHONUNBUFFERED"] = "1"
        env["TRITON_CACHE_DIR"] = str(cache_dir / "triton" / state.id)
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir / "inductor" / state.id)
        command = [
            sys.executable,
            "-m",
            "benchmarks.megabenchmark.worker",
            "--state",
            state.id,
            "--profile",
            args.profile,
            "--model-id",
            args.model_id,
            "--output",
            str(output_path),
        ]
        returncode = stream_process(command, env, log_path)
        if returncode != 0 or not output_path.exists():
            output_path.write_text(
                json.dumps(failure_result(state, returncode, log_path), indent=2)
                + "\n"
            )
            if args.fail_fast:
                break

    paths = generate_reports(results_dir)
    print("\nArtifacts")
    print(f"  raw states : {states_dir}")
    print(f"  logs       : {logs_dir}")
    print(f"  combined   : {paths['combined']}")
    print(f"  CSV        : {paths['csv']}")
    print(f"  plan report: {paths['markdown']}")


if __name__ == "__main__":
    main()
