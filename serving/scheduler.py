"""
Scheduler — iteration-level admission control (Section 15).

The whole point of continuous batching
--------------------------------------
Static batching locks a batch in place: every sequence is padded to the longest
one and the GPU can't take a new request until the *entire* batch finishes. A
64-token request stuck behind a 2048-token request idles for ~30x longer than
its own work needs.

Iteration-level scheduling fixes this by re-deciding the batch composition every
single decode step:

    each engine.step():
      1. evict requests that finished this iteration  → frees their KV slots
      2. admit waiting requests into any free slots, up to a token budget
      3. the engine runs ONE iteration over whatever is now running

Because admission happens every step, a short request can join, run, and leave
while a long request is still going — they no longer block each other.

What the scheduler owns
-----------------------
  - `waiting`:  FCFS queue of not-yet-admitted requests
  - `running`:  requests currently holding a KV slot
  - `free_slots`: the pool of available KV-cache rows (0 .. max_running-1)

Two budgets bound a batch:
  - `max_running`   : hard cap = number of KV-cache slots (rows) we allocated
  - `token_budget`  : soft cap on Σ context length across running requests, the
                      knob that keeps a single step's compute/memory bounded
"""

from __future__ import annotations

from collections import deque

from serving.request import Request, RequestState


class Scheduler:
    def __init__(self, max_running: int, token_budget: int):
        """
        Args:
            max_running:  max concurrent requests = number of KV-cache slots.
            token_budget: soft cap on total context tokens across the running
                          set; admission stops once admitting the next waiting
                          request would exceed it.
        """
        assert max_running > 0
        self.max_running = max_running
        self.token_budget = token_budget

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # Slots handed out lowest-first for deterministic, easy-to-debug behaviour.
        self.free_slots: list[int] = list(range(max_running))

    # ── public queue API ──────────────────────────────────────────────────

    def add(self, req: Request) -> None:
        """Enqueue a brand-new request (FCFS)."""
        req.state = RequestState.WAITING
        self.waiting.append(req)

    def has_work(self) -> bool:
        """True while any request is still waiting or running."""
        return bool(self.waiting or self.running)

    # ── per-iteration scheduling ──────────────────────────────────────────

    def evict_finished(self) -> list[Request]:
        """
        Remove finished requests from the running set and return their slots to
        the pool. Returns the evicted requests (for the engine to report out).
        """
        evicted = [r for r in self.running if r.state is RequestState.FINISHED]
        if not evicted:
            return []
        for r in evicted:
            self.free_slots.append(r.slot)
            r.slot = None
        self.free_slots.sort()
        self.running = [r for r in self.running if r.state is not RequestState.FINISHED]
        return evicted

    def _running_tokens(self) -> int:
        """Σ context length over the running set (prompt + already generated)."""
        return sum(r.prompt_len + r.num_generated for r in self.running)

    def admit(self) -> list[Request]:
        """
        Move WAITING → PREFILL for as many head-of-queue requests as fit in the
        free slots and the token budget. Returns the newly admitted requests so
        the engine knows which ones still need a prefill forward this iteration.
        """
        admitted: list[Request] = []
        budget_used = self._running_tokens()

        while self.waiting and self.free_slots:
            nxt = self.waiting[0]
            # Respect the token budget, but always allow at least one request in
            # (an empty running set must make progress even on a huge prompt).
            if self.running or admitted:
                if budget_used + nxt.prompt_len > self.token_budget:
                    break

            self.waiting.popleft()
            nxt.slot = self.free_slots.pop(0)
            nxt.pos = 0
            nxt.state = RequestState.PREFILL
            self.running.append(nxt)
            admitted.append(nxt)
            budget_used += nxt.prompt_len

        return admitted

    def step_schedule(self) -> tuple[list[Request], list[Request]]:
        """
        Run one scheduling round.

        Returns:
            (evicted, admitted) — evicted finished requests (slots freed) and
            newly admitted requests (now in PREFILL). After this call,
            `self.running` is the exact set the engine will run this iteration.
        """
        evicted = self.evict_finished()
        admitted = self.admit()
        return evicted, admitted

    def __repr__(self) -> str:
        return (
            f"Scheduler(waiting={len(self.waiting)}, running={len(self.running)}, "
            f"free_slots={len(self.free_slots)}/{self.max_running})"
        )
