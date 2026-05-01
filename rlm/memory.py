"""
memory.py
---------
ExternalMemory — the document lives here, NOT in the LLM's context window.

This is the core of what makes RLM different from a full-context baseline or
a large context call.  The model never sees the full document.  It interacts with
ExternalMemory via explicit method calls, retrieving only what it asks for.

State tracked per run
---------------------
  visited        : set[int]     — chunk ids the model has read this run
  partial_results: list[str]    — sub-call answers accumulated so far
  token_counts   : list[int]    — tokens used per model call
  sub_call_count : int          — number of recursive sub-calls made

All mutations are protected by asyncio.Lock so parallel sub-calls
(gathered via asyncio.gather) never corrupt shared state.

Public API (called by RLMEngine)
---------------------------------
  get_toc()                  -> str        # model reads this first
  get_chunk(chunk_id)        -> str        # fetch one chunk by id
  record_result(text, tokens)             # store a sub-call result
  summary()                  -> RunSummary # benchmark metrics at end
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set

from rlm.chunker import Chunk, Chunker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run summary — what the benchmark records per query
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    run_id:           str
    question:         str
    chunks_visited:   int
    unique_chunks:    int          # len(visited set) — no double-counting
    total_tokens:     int          # sum across all sub-calls
    sub_call_count:   int
    elapsed_seconds:  float
    partial_results:  List[str]


# ---------------------------------------------------------------------------
# ExternalMemory
# ---------------------------------------------------------------------------

class ExternalMemory:
    """
    Holds the full chunked document and all per-run mutable state.

    One ExternalMemory instance is created per agent run (per question).
    It is passed into RLMEngine and lives for the lifetime of that run.

    Parameters
    ----------
    chunks   : list of Chunk objects produced by Chunker
    run_id   : identifier for this run (used in logs and RunSummary)
    question : the user question for this run (stored for RunSummary)
    """

    def __init__(
        self,
        chunks:   List[Chunk],
        run_id:   str,
        question: str,
    ) -> None:
        # Immutable document store — keyed by chunk_id for O(1) lookup
        self._chunks: Dict[int, Chunk] = {c.chunk_id: c for c in chunks}
        self._toc: str = Chunker.toc(chunks)

        self._run_id   = run_id
        self._question = question

        # Per-run mutable state — all protected by _lock
        self._lock:            asyncio.Lock  = asyncio.Lock()
        self._visited:         Set[int]      = set()
        self._partial_results: List[str]     = []
        self._token_counts:    List[int]     = []
        self._sub_call_count:  int           = 0
        self._start_time:      float         = time.perf_counter()

    # ------------------------------------------------------------------
    # Read operations (safe to call without lock — read-only)
    # ------------------------------------------------------------------

    def get_toc(self) -> str:
        """Return the table of contents."""
        return self._toc

    def get_chunk(self, chunk_id: int) -> str:
        """
        Fetch a single chunk by id.

        Use fetch_and_mark() when the visit should also be recorded.
        Raises KeyError if chunk_id is out of range.
        """
        if chunk_id not in self._chunks:
            available = sorted(self._chunks.keys())
            raise KeyError(
                f"chunk_id {chunk_id} not found. "
                f"Available ids: {available[0]}–{available[-1]}"
            )

        chunk = self._chunks[chunk_id]
        logger.debug(
            "chunk_fetched",
            extra={
                "run_id":   self._run_id,
                "chunk_id": chunk_id,
                "section":  chunk.section_title,
                "tokens":   chunk.token_estimate,
            },
        )
        return chunk.text

    def chunk_count(self) -> int:
        """Total number of chunks in the document."""
        return len(self._chunks)

    def get_chunk_metadata(self, chunk_id: int) -> str:
        """Return a one-line description of a chunk without its full text."""
        if chunk_id not in self._chunks:
            return f"chunk_id {chunk_id} not found"
        return self._chunks[chunk_id].short_repr()

    # ------------------------------------------------------------------
    # Write operations (acquire lock before mutating)
    # ------------------------------------------------------------------

    async def mark_visited(self, chunk_id: int) -> None:
        """Record that a chunk was read during this run."""
        async with self._lock:
            self._visited.add(chunk_id)

    async def record_result(self, text: str, tokens: int) -> None:
        """
        Store the output of one recursive sub-call.

        Parameters
        ----------
        text   : the answer text produced by the sub-call
        tokens : total tokens consumed (prompt + completion)
        """
        async with self._lock:
            self._partial_results.append(text)
            self._token_counts.append(tokens)
            self._sub_call_count += 1
            logger.info(
                "sub_call_recorded",
                extra={
                    "run_id":       self._run_id,
                    "call_number":  self._sub_call_count,
                    "tokens":       tokens,
                    "total_tokens": sum(self._token_counts),
                },
            )

    # ------------------------------------------------------------------
    # Convenience: mark_visited + get_chunk in one atomic step
    # ------------------------------------------------------------------

    async def fetch_and_mark(self, chunk_id: int) -> str:
        """Fetch chunk text AND mark it visited atomically."""
        text = self.get_chunk(chunk_id)       # raises KeyError if bad id
        await self.mark_visited(chunk_id)
        return text

    # ------------------------------------------------------------------
    # Run summary — called by RLMEngine at the end of a run
    # ------------------------------------------------------------------

    async def summary(self) -> RunSummary:
        """Return benchmark metrics for this run."""
        async with self._lock:
            return RunSummary(
                run_id          = self._run_id,
                question        = self._question,
                chunks_visited  = len(self._visited),   # unique chunks read
                unique_chunks   = len(self._visited),
                total_tokens    = sum(self._token_counts),
                sub_call_count  = self._sub_call_count,
                elapsed_seconds = time.perf_counter() - self._start_time,
                partial_results = list(self._partial_results),
            )

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ExternalMemory("
            f"run_id={self._run_id!r}, "
            f"chunks={len(self._chunks)}, "
            f"visited={len(self._visited)}, "
            f"sub_calls={self._sub_call_count}"
            f")"
        )
