"""
Request — the unit of work for the continuous-batching engine (Section 15).

Why this exists
---------------
Up to Section 14 the "unit of work" was a single forward pass over one prompt.
The serving engine flips that: the unit of work becomes a *request* that lives
across many forward passes, and every decode iteration the scheduler decides
which requests share the next batch. A request therefore needs an explicit
lifecycle and enough bookkeeping to be paused/resumed between iterations.

Lifecycle
---------
    WAITING   queued, not yet admitted (no KV-cache slot assigned)
       │  admit(): a slot frees up and the token budget allows it
       ▼
    PREFILL   admitted; its prompt still needs the one-shot prefill forward
       │  after the prefill forward runs and the first token is sampled
       ▼
    DECODE    steady state; one token produced per engine.step()
       │  EOS sampled OR len(generated) == max_new_tokens
       ▼
    FINISHED  done; its slot is released back to the pool

The Request carries everything needed to resume it next iteration:
  - `slot`: which row of the KV-cache pool holds this request's K/V (None until
    admitted). This is the contiguous-cache analogue of vLLM's block table.
  - `pos`:  absolute position of the NEXT token to write — equals prompt_len
    after prefill, then increments by one per decoded token. Drives both the
    KV write index and the RoPE angle for this request.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field


class RequestState(enum.Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


_id_counter = itertools.count(1)


@dataclass
class Request:
    prompt_tokens: list[int]
    max_new_tokens: int

    # assigned by the engine / scheduler ------------------------------------
    id: int = field(default_factory=lambda: next(_id_counter))
    state: RequestState = RequestState.WAITING
    slot: int | None = None          # KV-cache row, assigned on admission
    pos: int = 0                     # absolute index of the NEXT token to write
    generated: list[int] = field(default_factory=list)
    eos_hit: bool = False

    # ── derived helpers ────────────────────────────────────────────────────

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_generated(self) -> int:
        return len(self.generated)

    @property
    def last_token(self) -> int:
        """The most recent token (last generated, or last prompt token)."""
        return self.generated[-1] if self.generated else self.prompt_tokens[-1]

    def reached_limit(self) -> bool:
        """True once this request has produced its full token budget."""
        return self.num_generated >= self.max_new_tokens

    def should_finish(self) -> bool:
        """Terminal condition check (EOS already sampled, or hit the limit)."""
        return self.eos_hit or self.reached_limit()

    def __repr__(self) -> str:
        return (
            f"Request(id={self.id}, state={self.state.value}, slot={self.slot}, "
            f"pos={self.pos}, prompt={self.prompt_len}, gen={self.num_generated})"
        )
