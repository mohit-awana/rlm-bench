"""
rlm_tool.py — RLMTool exposed to the agent via ToolContract.

RLMTool wraps RLMEngine behind a standard ToolContract interface.
All recursive reasoning, memory management, and token tracking happen
inside the engine — the agent sees only a question-in, answer-out call.

Usage:
    registry = ToolRegistry(pdf_path="data/httpx_src")
    answer   = await agent("Trace the call chain for `httpx.get(url)`.", registry=registry)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from typing import List, Type

from pydantic import BaseModel, ConfigDict

from rlm.chunker import Chunker
from rlm.config  import cfg
from rlm.engine  import RLMEngine
from rlm.memory  import ExternalMemory, RunSummary
from tools.contract import ToolContract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# trace_tool decorator — logs latency and errors for every tool execution
# ---------------------------------------------------------------------------

def trace_tool(func):
    @functools.wraps(func)
    async def wrapper(self, inputs):
        start     = time.perf_counter()
        tool_name = self.__class__.__name__
        logger.info("tool_start", extra={"tool": tool_name, "inputs": inputs.model_dump()})
        try:
            result   = await func(self, inputs)
            duration = time.perf_counter() - start
            logger.info("tool_success", extra={"tool": tool_name, "latency": duration})
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error("tool_failure", extra={"tool": tool_name, "latency": duration, "error": str(e)})
            raise
    return wrapper


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RLMInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question:    str
    document_id: str = "httpx_src"


class RLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer:         str
    chunks_visited: int
    total_tokens:   int
    sub_call_count: int
    elapsed_seconds: float
    run_id:         str


# ---------------------------------------------------------------------------
# RLMEngine wrapper — created once, reused across calls
# ---------------------------------------------------------------------------

class RLMEngineWrapper:
    """
    Holds the pre-chunked document and a reusable RLMEngine instance.
    Created once at startup via ToolRegistry — never re-chunked per query.
    """

    def __init__(self, pdf_path: str, model: str = None, chunk_size: int = None) -> None:
        model      = model      or cfg.model
        chunk_size = chunk_size or cfg.chunk_size
        logger.info("Loading codebase: %s", pdf_path)
        self._chunks  = Chunker.from_dir(pdf_path)
        self._engine  = RLMEngine(model=model)
        logger.info("Codebase ready: %d files", len(self._chunks))

    async def run(self, question: str) -> tuple[str, RunSummary]:
        run_id = str(uuid.uuid4())[:8]
        memory = ExternalMemory(
            chunks   = self._chunks,
            run_id   = run_id,
            question = question,
        )
        return await self._engine.run(question, memory)


# ---------------------------------------------------------------------------
# RLMTool
# ---------------------------------------------------------------------------

class RLMTool:
    """Answer questions over the configured httpx source tree using recursive
    document reasoning. Use this for questions that require
    cross-section analysis, comparisons, or multi-hop reasoning over the
    source material."""

    input_schema  = RLMInput
    output_schema = RLMOutput

    def __init__(self, engine: RLMEngineWrapper | None = None) -> None:
        self.engine = engine  # injected by ToolRegistry

    @classmethod
    def schema(cls) -> dict:
        return {
            "type": "function",
            "function": {
                "name":        cls.__name__,
                "description": cls.__doc__.strip(),
                "parameters":  cls.input_schema.model_json_schema(),
            },
        }

    @trace_tool
    async def execute(self, inputs: RLMInput) -> RLMOutput:
        if self.engine is None:
            raise RuntimeError("RLMTool has no engine — inject via ToolRegistry")

        answer, summary = await self.engine.run(inputs.question)

        return RLMOutput(
            answer          = answer,
            chunks_visited  = summary.chunks_visited,
            total_tokens    = summary.total_tokens,
            sub_call_count  = summary.sub_call_count,
            elapsed_seconds = summary.elapsed_seconds,
            run_id          = summary.run_id,
        )


# ---------------------------------------------------------------------------
# ToolRegistry — single instance, engines created once at startup
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Builds and holds tool instances at startup — never re-created per query.
    """

    def __init__(
        self,
        pdf_path:   str  = None,
        model:      str  = None,
        chunk_size: int  = None,
    ) -> None:
        pdf_path   = pdf_path   or cfg.pdf_path
        model      = model      or cfg.model
        chunk_size = chunk_size or cfg.chunk_size
        _engine = RLMEngineWrapper(
            pdf_path   = pdf_path,
            model      = model,
            chunk_size = chunk_size,
        )

        self._tools: dict[str, ToolContract] = {
            RLMTool.__name__: RLMTool(engine=_engine),
        }

    def schemas(self) -> List[dict]:
        return [t.schema() for t in self._tools.values()]

    def get(self, name: str) -> ToolContract | None:
        return self._tools.get(name)

    def input_schema_for(self, name: str) -> Type[BaseModel] | None:
        tool = self._tools.get(name)
        return tool.input_schema if tool else None
