"""
serve_trace.py — open an exported engine trace in the Perfetto UI.

The HF "Profiling in PyTorch" blog views traces in the Perfetto UI
(https://ui.perfetto.dev). profile_engine_torch.py already exports a
Perfetto-compatible Chrome trace per flavor to torch-profiler/out/
(engine_<flavor>_trace.json), so the only missing piece is a convenient way to
load one into Perfetto from this (often remote) machine WITHOUT uploading it to
any third party (the blog's `trace-util` pushes traces to a Hugging Face
bucket).

This helper serves a single trace file over localhost with the CORS handshake
Perfetto UI expects, then prints the deep link:

    https://ui.perfetto.dev/#!/?url=http://127.0.0.1:9001/<trace>

It mirrors Perfetto's own tools/open_trace_in_ui script: Perfetto UI fetches the
`url` you give it, so the server must answer that fetch with an
`Access-Control-Allow-Origin: https://ui.perfetto.dev` header.

USAGE
-----
    # serve the newest engine_*_trace.json in out/
    uv run python profiling/profile-engine/torch-profiler/serve_trace.py

    # serve a specific flavor
    uv run python profiling/profile-engine/torch-profiler/serve_trace.py --flavor short

    # serve an explicit file / pick a port
    uv run python profiling/profile-engine/torch-profiler/serve_trace.py --trace path/to.json --port 9001

REMOTE MACHINES
---------------
Perfetto runs in YOUR browser and fetches http://127.0.0.1:<port>, so when the
trace lives on a remote box, forward the port first, e.g.:

    ssh -L 9001:localhost:9001 <user>@<host>

then open the printed link locally.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "out"

PERFETTO_ORIGIN = "https://ui.perfetto.dev"


def _resolve_trace(args) -> Path:
    if args.trace:
        p = Path(args.trace).resolve()
        if not p.is_file():
            raise SystemExit(f"trace not found: {p}")
        return p
    if args.flavor:
        p = OUT_DIR / f"engine_{args.flavor}_trace.json"
        if not p.is_file():
            raise SystemExit(
                f"no trace for flavor '{args.flavor}' at {p} — run "
                "profile_engine_torch.py first."
            )
        return p
    traces = sorted(OUT_DIR.glob("engine_*_trace.json"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit(
            f"no engine_*_trace.json in {OUT_DIR} — run profile_engine_torch.py first."
        )
    return traces[-1].resolve()


class _PerfettoCORSHandler(SimpleHTTPRequestHandler):
    """Static file handler that lets the Perfetto UI fetch the trace cross-origin."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", PERFETTO_ORIGIN)
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *_args):  # quiet: one request per load is enough noise
        pass


def main():
    ap = argparse.ArgumentParser(description="Open an engine trace in the Perfetto UI.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--flavor", help="serve out/engine_<flavor>_trace.json")
    g.add_argument("--trace", help="serve an explicit trace .json path")
    ap.add_argument("--port", type=int, default=9001, help="localhost port (default 9001)")
    args = ap.parse_args()

    trace = _resolve_trace(args)
    serve_dir = trace.parent
    fname = trace.name

    handler = partial(_PerfettoCORSHandler, directory=str(serve_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    link = f"{PERFETTO_ORIGIN}/#!/?url=http://127.0.0.1:{args.port}/{fname}"
    bar = "=" * 78
    print(bar)
    print("  Serving trace for the Perfetto UI")
    print(bar)
    print(f"  trace : {trace}  ({trace.stat().st_size / 1e6:.1f} MB)")
    print(f"  open  : {link}")
    print("  (remote box? forward the port first: "
          f"ssh -L {args.port}:localhost:{args.port} <user>@<host>)")
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
