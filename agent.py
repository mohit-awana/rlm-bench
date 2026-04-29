"""
agent.py
--------

  - RLMTool is the only tool
  - ToolRegistry is configured with the source directory via cfg.pdf_path

Usage
-----
    python agent.py
    > Ask question: Trace the complete call chain when a user calls `httpx.get(url)`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import List, Type

import ollama
from pydantic import BaseModel

from tools.rlm_tool import RLMTool, RLMInput, ToolRegistry
from tools.contract  import ToolContract
from rlm.config      import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL  = cfg.model
SYSTEM = (
    "You are a precise Python code analyst. "
    "Use RLMTool to answer any question about the httpx Python library source code. "
    "Always use the tool — do not answer from memory."
)

# ---------------------------------------------------------------------------
# Guarded execution — unchanged from original
# ---------------------------------------------------------------------------

async def guarded_execute(
    tool:   ToolContract,
    inputs: BaseModel,
    *,
    sem:    asyncio.Semaphore,
    run_id: str,
) -> BaseModel:
    """Acquire semaphore slot then execute with hard timeout."""
    tool_name = tool.__class__.__name__
    async with sem:
        logger.info("sem_acquired", extra={"run_id": run_id, "tool": tool_name})
        try:
            return await asyncio.wait_for(tool.execute(inputs), timeout=120)
        finally:
            logger.info("sem_released", extra={"run_id": run_id, "tool": tool_name})


# ---------------------------------------------------------------------------
# _append_tool_result — unchanged from original
# ---------------------------------------------------------------------------

async def _append_tool_result(
    name:         str,
    result:       BaseModel | Exception,
    *,
    messages:     List[dict],
    history_lock: asyncio.Lock,
    run_id:       str,
) -> None:
    if isinstance(result, Exception):
        logger.exception(f"{name} failed", exc_info=result, extra={"run_id": run_id})
        content = f"Tool error: {result}"
    else:
        content = str(result.model_dump())

    async with history_lock:
        messages.append({"role": "tool", "name": name, "content": content})


# ---------------------------------------------------------------------------
# Turn warning — unchanged from original
# ---------------------------------------------------------------------------

def _maybe_inject_turn_warning(
    messages:  List[dict],
    turn:      int,
    max_turns: int,
) -> None:
    if turn == max_turns - 2:
        messages.append({
            "role":    "system",
            "content": (
                "You have 1 turn remaining. "
                "If you have enough information, produce your final answer now "
                "without calling any more tools."
            ),
        })
        logger.info("turn_warning_injected", extra={"turn": turn, "max_turns": max_turns})


# ---------------------------------------------------------------------------
# Agent — unchanged from original except registry type
# ---------------------------------------------------------------------------

async def agent(
    user_input:      str,
    *,
    registry:        ToolRegistry,
    max_concurrency: int = 3,
    max_turns:       int = 5,
) -> str:
    run_id       = str(uuid.uuid4())[:8]
    sem          = asyncio.Semaphore(max_concurrency)
    history_lock = asyncio.Lock()

    messages: List[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_input},
    ]

    logger.info("agent_start", extra={"run_id": run_id})

    for turn in range(max_turns):
        _maybe_inject_turn_warning(messages, turn, max_turns)

        response = await asyncio.to_thread(
            ollama.chat,
            model    = MODEL,
            messages = messages,
            tools    = registry.schemas(),
        )

        msg = response.message

        async with history_lock:
            messages.append({"role": "assistant", "content": msg.content or ""})

        if not msg.tool_calls:
            logger.info("agent_done", extra={"run_id": run_id, "turns": turn + 1})
            return msg.content or ""

        tasks, tool_names = [], []

        for call in msg.tool_calls:
            tool = registry.get(call.function.name)
            if tool is None:
                raise KeyError(f"Unknown tool: {call.function.name}")

            inputs = tool.input_schema.model_validate(call.function.arguments)
            tasks.append(asyncio.create_task(
                guarded_execute(tool, inputs, sem=sem, run_id=run_id)
            ))
            tool_names.append(call.function.name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        await asyncio.gather(*[
            _append_tool_result(
                name, result,
                messages     = messages,
                history_lock = history_lock,
                run_id       = run_id,
            )
            for name, result in zip(tool_names, results)
        ])

    logger.warning("agent_max_turns_exceeded", extra={"run_id": run_id})
    return (
        f"[Agent stopped after {max_turns} turns without a final answer. "
        "Try rephrasing your question.]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    registry = ToolRegistry(
        pdf_path   = cfg.pdf_path,
        model      = MODEL,
        chunk_size = 500,
    )
    q = input("Ask question: ")
    print(await agent(q, registry=registry))


if __name__ == "__main__":
    asyncio.run(main())
