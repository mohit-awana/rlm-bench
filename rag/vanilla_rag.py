"""
vanilla_rag.py — full-document long-context baseline (module name kept for runner.py compatibility).

FullContextBaseline — the natural comparison for RLM.

This is NOT RAG. This is the full-document long-context baseline:
the entire document is stuffed into a single prompt and the model
answers in one pass. No retrieval, no chunking for retrieval,
no vector embeddings.

This mirrors what the original MIT RLM paper (Zhang et al. 2025)
compares against: vanilla long-context LLM calls vs recursive
document navigation.

Why this is the right baseline
-------------------------------
The MIT paper's core claim is that RLM outperforms direct long-context
calls because:
  1. Lost-in-the-middle effect — facts buried in long contexts are missed
  2. Context degradation — model quality drops as context grows
  3. Cost — RLM uses targeted retrieval over only the relevant files;
     full-context uses the entire corpus regardless of what is relevant

Our contribution: implementing this comparison via a ToolContract-
compatible interface so any agent can switch between strategies.

Compatibility note
------------------
This module is imported as vanilla_rag for runner.py compatibility.
The pipeline label in results is "full_context" not "rag".
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI

from rlm.chunker import Chunk, Chunker
from rlm.config  import cfg

logger = logging.getLogger(__name__)

FULL_CONTEXT_SYSTEM = """You are a precise Python code analyst.

You have been given the COMPLETE source code of the httpx library below.
Answer the question using ONLY information from this source code.
Be specific. Cite file names and function names.
Trace call chains carefully — follow imports and function calls across files.
If the code does not contain enough information, say so clearly."""


@dataclass
class RAGResult:
    """Named RAGResult for runner.py compatibility."""
    run_id:           str
    question:         str
    answer:           str
    chunks_retrieved: int    # always = total chunks (full doc)
    retrieved_ids:    List[int]
    total_tokens:     int
    elapsed_seconds:  float


class VanillaRAG:
    """
    Full-document long-context baseline.
    Named VanillaRAG for runner.py import compatibility.

    Stuffs the entire document into a single prompt and generates
    one answer in one API call. No retrieval step.
    """

    def __init__(
        self,
        chunks:         List[Chunk],
        top_k:          int = 5,           # ignored — kept for interface compatibility
        generate_model: str = None,
        embed_model:    str = None,        # ignored — no embeddings needed
    ) -> None:
        self._chunks        = chunks
        self._generate_model = generate_model or cfg.model
        self._client        = AsyncOpenAI(api_key=cfg.api_key)

        # Build full document text once at startup
        self._full_document = self._build_full_document(chunks)
        self._total_tokens_estimate = sum(c.token_estimate for c in chunks)

        logger.info(
            "FullContextBaseline ready: %d chunks, ~%d tokens total document",
            len(chunks), self._total_tokens_estimate,
        )

    async def build_index(self) -> None:
        """No-op — full context needs no index. Kept for interface compatibility."""
        pass

    @classmethod
    async def from_dir(
        cls,
        src_dir:        str,
        generate_model: str = None,
    ) -> "VanillaRAG":
        chunks = Chunker.from_dir(src_dir)
        return cls(chunks=chunks, generate_model=generate_model)

    async def run(self, question: str) -> RAGResult:
        """
        Full-context single-pass generation.
        Entire document → single prompt → one API call → answer.
        """
        run_id = str(uuid.uuid4())[:8]
        start  = time.perf_counter()

        prompt = (
            f"COMPLETE DOCUMENT TEXT:\n\n"
            f"{self._full_document}\n\n"
            f"---\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer based only on the document above."
        )

        logger.info(
            "full_context_run run_id=%s doc_tokens=~%d question=%s",
            run_id, self._total_tokens_estimate, question[:60],
        )

        response = await self._client.chat.completions.create(
            model    = self._generate_model,
            messages = [
                {"role": "system", "content": FULL_CONTEXT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )

        answer  = response.choices[0].message.content or ""
        tokens  = (
            response.usage.prompt_tokens +
            response.usage.completion_tokens
        )
        elapsed = time.perf_counter() - start

        logger.info(
            "full_context_done run_id=%s tokens=%d elapsed=%.2fs",
            run_id, tokens, elapsed,
        )

        return RAGResult(
            run_id           = run_id,
            question         = question,
            answer           = answer,
            chunks_retrieved = len(self._chunks),  # all chunks = full doc
            retrieved_ids    = list(range(len(self._chunks))),
            total_tokens     = tokens,
            elapsed_seconds  = elapsed,
        )

    def _build_full_document(self, chunks: List[Chunk]) -> str:
        """
        Concatenate all chunks into one document string.
        Preserves section structure with clear headers.
        """
        parts = []
        for chunk in chunks:
            header = (
                f"\n{'='*60}\n"
                f"Section: {chunk.section_title} "
                f"(Pages {chunk.page_start}–{chunk.page_end})\n"
                f"{'='*60}\n"
            )
            parts.append(header + chunk.text)
        return "\n".join(parts)
