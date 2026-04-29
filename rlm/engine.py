"""
engine.py
---------
RLMEngine — recursive inference loop using OpenAI gpt-4.1-nano.

Implementation notes
--------------------
- Uses openai.AsyncOpenAI for all model calls (native async, no to_thread needed)
- Token counting via response.usage.prompt_tokens + completion_tokens
- Tool calls via OpenAI function-calling protocol (reliable, not text-parsed)
- Auto-TOC injection on Turn 0 if model skips get_toc()
- chunk_id parser handles float strings e.g. "2.1" → 2
- Turn warning softened to avoid premature termination

Tools exposed to the model
--------------------------
  get_toc()              → compact table of contents
  get_chunk(chunk_id)    → one section by integer id
  record_partial(finding)→ store intermediate finding
"""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any, Dict, List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from rlm.config import cfg
from rlm.memory import ExternalMemory, RunSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_toc",
            "description": (
                "Get the table of contents for the document. "
                "ALWAYS call this first before fetching any chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": (
                "Fetch the full text of one document chunk by its integer chunk_id. "
                "Use the TOC to identify which chunk_id is relevant. "
                "Call multiple times to read different sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "integer",
                        "description": "Integer chunk_id from the TOC.",
                    }
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_partial",
            "description": (
                "Store an intermediate finding before reading more chunks. "
                "Use when you have a partial answer but need to check other sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "finding": {
                        "type": "string",
                        "description": "The intermediate finding to record.",
                    }
                },
                "required": ["finding"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise Python code analyst with access to an external document store.

You CANNOT see the full document. You MUST use tools to navigate it.

STRICT WORKFLOW — follow this exactly:
1. ALWAYS call get_toc() first. This is mandatory.
2. Identify which chunk_ids from the TOC are relevant to the question.
3. Call get_chunk(chunk_id) for each relevant section. Read at least 2 chunks.
4. If you find a partial answer, call record_partial(finding) to save it.
5. After reading ALL relevant sections, produce your final answer.

RULES:
- Never answer from memory or training data alone — always read the document.
- Always call get_toc() before anything else — no exceptions.
- Never guess section content — fetch it.
- Cite file names and chunk ids in your final answer.
- Your final answer must be grounded only in what you read from the document."""


# ---------------------------------------------------------------------------
# RLMEngine
# ---------------------------------------------------------------------------

class RLMEngine:
    """
    Drives recursive document reasoning over ExternalMemory using OpenAI.

    Parameters
    ----------
    model     : OpenAI model name (default from cfg)
    max_turns : hard cap on conversation turns
    """

    def __init__(
        self,
        model:     str = None,
        max_turns: int = 20,
    ) -> None:
        self.model     = model or cfg.model
        self.max_turns = max_turns
        self._client   = AsyncOpenAI(api_key=cfg.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        question: str,
        memory:   ExternalMemory,
    ) -> tuple[str, RunSummary]:
        """
        Run the recursive inference loop for one question.
        Returns (final_answer, RunSummary).
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": question},
        ]

        logger.info(
            "rlm_run_start run_id=%s question=%s",
            memory._run_id, question[:80]
        )

        # Auto-inject TOC on turn 0 — ensures model always starts with context
        toc_text = memory.get_toc()
        messages.append({
            "role":    "system",
            "content": f"Document Table of Contents (auto-loaded):\n\n{toc_text}",
        })
        await memory.record_result("[TOC auto-loaded]", 0)

        final_answer = "[RLM did not produce a final answer]"

        for turn in range(self.max_turns):

            # Soft turn warning — don't force termination, just nudge
            if turn == self.max_turns - 3:
                messages.append({
                    "role":    "system",
                    "content": (
                        f"You have {self.max_turns - turn} turns remaining. "
                        "If you have read all relevant sections, produce your final answer."
                    ),
                })

            # OpenAI async call
            response: ChatCompletion = await self._client.chat.completions.create(
                model    = self.model,
                messages = messages,
                tools    = TOOLS,
                tool_choice = "auto",
            )

            choice     = response.choices[0]
            msg        = choice.message
            tokens_used = (
                response.usage.prompt_tokens +
                response.usage.completion_tokens
            )

            # Append assistant message
            messages.append(msg.model_dump(exclude_none=True))

            # No tool calls → model is done
            if not msg.tool_calls:
                final_answer = msg.content or final_answer
                logger.info(
                    "rlm_run_done run_id=%s turns=%d tokens=%d",
                    memory._run_id, turn + 1, tokens_used
                )
                await memory.record_result(final_answer, tokens_used)
                break

            # Execute tool calls
            tool_results = await self._execute_tools(msg.tool_calls, memory)
            await memory.record_result(msg.content or "", tokens_used)

            # Append tool results
            for tool_call_id, tool_name, result_text in tool_results:
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call_id,
                    "name":         tool_name,
                    "content":      result_text,
                })

        else:
            logger.warning(
                "rlm_max_turns_exceeded run_id=%s max_turns=%d",
                memory._run_id, self.max_turns
            )

        summary = await memory.summary()
        return final_answer, summary

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    async def _execute_tools(
        self,
        tool_calls: list,
        memory:     ExternalMemory,
    ) -> List[tuple[str, str, str]]:
        """
        Execute all tool calls from one turn.
        Returns list of (tool_call_id, tool_name, result_text).
        """
        results = []
        for call in tool_calls:
            tool_call_id = call.id
            name         = call.function.name
            args         = self._parse_args(call.function.arguments)
            result       = await self._dispatch(name, args, memory)
            results.append((tool_call_id, name, result))
            logger.debug(
                "tool_executed run_id=%s tool=%s args=%s",
                memory._run_id, name, args
            )
        return results

    @staticmethod
    def _parse_args(arguments: str | dict) -> Dict[str, Any]:
        """Parse tool arguments — handles both string and dict."""
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}

    async def _dispatch(
        self,
        name:   str,
        args:   Dict[str, Any],
        memory: ExternalMemory,
    ) -> str:
        """Route a tool call to the correct memory method."""

        if name == "get_toc":
            return memory.get_toc()

        elif name == "get_chunk":
            raw = args.get("chunk_id", -1)
            # Robustly handle int, float, float-string e.g. "2.1" → 2
            try:
                chunk_id = int(float(str(raw)))
            except (ValueError, TypeError):
                return f"Error: chunk_id must be an integer, got {raw!r}"
            try:
                return await memory.fetch_and_mark(chunk_id)
            except KeyError as e:
                return f"Error: {e}"

        elif name == "record_partial":
            finding = str(args.get("finding", ""))
            await memory.record_result(finding, 0)
            return f"Recorded: {finding[:80]}..."

        else:
            return f"Unknown tool: {name}"
