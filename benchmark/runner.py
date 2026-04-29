"""
runner.py
---------
Benchmark runner — executes both RLM and full-context baseline pipelines against all 20
questions and records structured results for DeepEval scoring.

Design principles
-----------------
1. Both pipelines use the SAME chunks from the SAME source directory.
   VanillaRAG and RLMTool are built from identical Chunker output.
   Any quality difference is purely due to the retrieval/reasoning strategy.

2. Questions run sequentially per pipeline to keep runs reproducible and
   avoid concurrent API spikes. Parallelism within a single RLM run
   (sub-calls) is handled by engine.py.

3. Every result is recorded atomically to results/benchmark_results.json
   after each question so partial runs are recoverable.

4. Failures are caught and recorded — a failed question gets a special
   result entry rather than crashing the whole run.

5. Cost is measured as total_tokens (prompt + completion across all calls).
   For API-based runs, actual dollar cost depends on the provider and
   model pricing, so token count is the comparison proxy.

6. Latency is wall-clock elapsed_seconds per question. The summary table
   reports avg, p50 (median), and p95 per pipeline x tier.

Output schema (per question, per pipeline)
-------------------------------------------
{
  "question_id"     : int,
  "tier"            : int,
  "tier_label"      : str,
  "question"        : str,
  "ground_truth"    : str,
  "source_files"    : list[str],
  "pipeline"        : "rlm" | "full_context",
  "answer"          : str,
  "total_tokens"    : int,
  "chunks_accessed" : int,     # chunks_visited for RLM, chunks_retrieved for the full-context baseline
  "sub_call_count"  : int,     # RLM: tool calls made; full-context: always 1
  "elapsed_seconds" : float,
  "error"           : str | null
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmark.questions import ALL_QUESTIONS, BenchmarkQuestion
from rag.vanilla_rag      import VanillaRAG
from rlm.chunker          import Chunker
from rlm.config           import cfg
from tools.rlm_tool       import RLMEngineWrapper

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — driven by cfg singleton (loaded from .env)
# ---------------------------------------------------------------------------

PDF_PATH    = cfg.pdf_path   # legacy field name retained for compatibility; points at the httpx source tree
MODEL       = cfg.model
CHUNK_SIZE  = cfg.chunk_size
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

def make_result(
    question:        BenchmarkQuestion,
    pipeline:        str,
    answer:          str,
    total_tokens:    int,
    chunks_accessed: int,
    sub_call_count:  int,
    elapsed_seconds: float,
    error:           Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "question_id":     question.id,
        "tier":            question.tier,
        "tier_label":      question.tier_label,
        "question":        question.question,
        "ground_truth":    question.ground_truth,
        "source_files":    question.source_files,
        "pipeline":        pipeline,
        "answer":          answer,
        "total_tokens":    total_tokens,
        "chunks_accessed": chunks_accessed,
        "sub_call_count":  sub_call_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "error":           error,
    }


# ---------------------------------------------------------------------------
# Single question runners
# ---------------------------------------------------------------------------

async def run_rlm_question(
    question: BenchmarkQuestion,
    engine:   RLMEngineWrapper,
) -> Dict[str, Any]:
    """Run one question through the RLM pipeline."""
    logger.info(
        "RLM  Q%02d T%d  %s",
        question.id, question.tier, question.question[:60]
    )
    start = time.perf_counter()
    try:
        answer, summary = await engine.run(question.question)
        elapsed = time.perf_counter() - start
        return make_result(
            question        = question,
            pipeline        = "rlm",
            answer          = answer,
            total_tokens    = summary.total_tokens,
            chunks_accessed = summary.chunks_visited,
            sub_call_count  = summary.sub_call_count,
            elapsed_seconds = elapsed,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("RLM Q%02d failed: %s", question.id, e)
        return make_result(
            question        = question,
            pipeline        = "rlm",
            answer          = "",
            total_tokens    = 0,
            chunks_accessed = 0,
            sub_call_count  = 0,
            elapsed_seconds = elapsed,
            error           = str(e),
        )


async def run_full_context_question(
    question: BenchmarkQuestion,
    full_context: VanillaRAG,
) -> Dict[str, Any]:
    """Run one question through the full-context baseline."""
    logger.info(
        "FULL_CONTEXT  Q%02d T%d  %s",
        question.id, question.tier, question.question[:60]
    )
    start = time.perf_counter()
    try:
        result  = await full_context.run(question.question)
        elapsed = time.perf_counter() - start
        return make_result(
            question        = question,
            pipeline        = "full_context",
            answer          = result.answer,
            total_tokens    = result.total_tokens,
            chunks_accessed = result.chunks_retrieved,
            sub_call_count  = 1,          # full context always makes exactly 1 generation call
            elapsed_seconds = elapsed,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("FULL_CONTEXT Q%02d failed: %s", question.id, e)
        return make_result(
            question        = question,
            pipeline        = "full_context",
            answer          = "",
            total_tokens    = 0,
            chunks_accessed = 0,
            sub_call_count  = 0,
            elapsed_seconds = elapsed,
            error           = str(e),
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> List[Dict[str, Any]]:
    """Load existing results if the run was interrupted."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_checkpoint(results: List[Dict[str, Any]], path: Path) -> None:
    """Write results atomically after every question."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


def already_done(
    results:     List[Dict[str, Any]],
    question_id: int,
    pipeline:    str,
) -> bool:
    """Check if this (question_id, pipeline) pair already has a result."""
    return any(
        r["question_id"] == question_id and r["pipeline"] == pipeline
        for r in results
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    questions:   List[BenchmarkQuestion] = ALL_QUESTIONS,
    run_rlm:     bool = True,
    run_full_context: bool = True,
    output_file: str  = "benchmark_results.json",
) -> List[Dict[str, Any]]:
    """
    Run the full benchmark.

    Parameters
    ----------
    questions   : which questions to run (default: all 20 questions)
    run_rlm     : whether to run the RLM pipeline
    run_full_context : whether to run the full-context baseline
    output_file : filename in results/ directory

    Returns
    -------
    List of result dicts (one per question per pipeline)
    """
    output_path = RESULTS_DIR / output_file
    results     = load_checkpoint(output_path)

    if results:
        logger.info(
            "Resuming from checkpoint: %d results already recorded",
            len(results)
        )

    # ------------------------------------------------------------------
    # Build pipelines ONCE — chunking happens here, not per question
    # ------------------------------------------------------------------
    logger.info("Building pipelines from %s ...", PDF_PATH)
    build_start = time.perf_counter()

    chunks = Chunker.from_dir(PDF_PATH)
    logger.info(
        "Loaded codebase: %d files | ~%d tokens total",
        len(chunks),
        sum(c.token_estimate for c in chunks),
    )

    rlm_engine = RLMEngineWrapper(
        pdf_path   = PDF_PATH,
        model      = MODEL,
        chunk_size = CHUNK_SIZE,
    ) if run_rlm else None

    full_context_pipeline = None
    if run_full_context:
        full_context_pipeline = VanillaRAG(
            chunks         = chunks,
            generate_model = MODEL,
        )
        await full_context_pipeline.build_index()

    logger.info("Pipelines ready in %.1fs", time.perf_counter() - build_start)

    # ------------------------------------------------------------------
    # Run questions sequentially — one at a time to avoid concurrent API spikes
    # ------------------------------------------------------------------
    total     = len(questions) * (int(run_rlm) + int(run_full_context))
    done      = 0
    run_start = time.perf_counter()

    for q in questions:
        # RLM first
        if run_rlm and not already_done(results, q.id, "rlm"):
            result = await run_rlm_question(q, rlm_engine)
            results.append(result)
            save_checkpoint(results, output_path)
            done += 1
            _log_progress(done, total, run_start, q, "rlm", result)

        # Full context second
        if run_full_context and not already_done(results, q.id, "full_context"):
            result = await run_full_context_question(q, full_context_pipeline)
            results.append(result)
            save_checkpoint(results, output_path)
            done += 1
            _log_progress(done, total, run_start, q, "full_context", result)

    logger.info(
        "Benchmark complete: %d results saved to %s",
        len(results), output_path,
    )

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Progress logger
# ---------------------------------------------------------------------------

def _log_progress(
    done:     int,
    total:    int,
    start:    float,
    question: BenchmarkQuestion,
    pipeline: str,
    result:   Dict[str, Any],
) -> None:
    pct    = done / total * 100
    status = "ERROR" if result["error"] else "OK"
    logger.info(
        "[%d/%d %.0f%%]  %s  Q%02d T%d  "
        "tokens=%-6d  chunks=%-3d  sub_calls=%-3d  elapsed=%.2fs  %s",
        done, total, pct,
        pipeline.upper(),
        question.id, question.tier,
        result["total_tokens"],
        result["chunks_accessed"],
        result["sub_call_count"],
        result["elapsed_seconds"],
        status,
    )


# ---------------------------------------------------------------------------
# Summary table — printed after run completes
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print summary grouped by pipeline x tier.
    Reports: N, avg tokens, avg chunks, avg sub-calls, avg/p50/p95 latency.
    """
    from collections import defaultdict

    buckets: Dict[str, Dict[int, List]] = {
        "rlm":          defaultdict(list),
        "full_context": defaultdict(list),
    }
    for r in results:
        if not r.get("error"):
            buckets[r["pipeline"]][r["tier"]].append(r)

    W = 100
    print("\n" + "=" * W)
    print("BENCHMARK RUN SUMMARY  (20 questions, Tier 3, RLM vs Full-Context Baseline)")
    print("=" * W)
    print(
        f"{'Pipeline':<16} {'Tier':<10} {'N':<5} "
        f"{'Avg Tokens':<13} {'Avg Chunks':<12} {'Avg Calls':<11} "
        f"{'Avg(s)':<9} {'p50(s)':<9} {'p95(s)'}"
    )
    print("-" * W)

    for pipeline in ("rlm", "full_context"):
        for tier in (1, 2, 3, 0):
            if tier:
                items = buckets[pipeline].get(tier, [])
                label = f"Tier {tier}"
                prefix = "  "
            else:
                items  = [r for lst in buckets[pipeline].values() for r in lst]
                label  = "OVERALL"
                prefix = "→ "

            # Skip empty tiers — only print rows with actual results
            if not items:
                continue

            times = np.array([r["elapsed_seconds"] for r in items])
            print(
                f"{prefix}{pipeline.upper():<14} {label:<10} {len(items):<5} "
                f"{sum(r['total_tokens'] for r in items)/len(items):<13.0f} "
                f"{sum(r['chunks_accessed'] for r in items)/len(items):<12.1f} "
                f"{sum(r['sub_call_count'] for r in items)/len(items):<11.1f} "
                f"{float(times.mean()):<9.2f} "
                f"{float(np.percentile(times,50)):<9.2f} "
                f"{float(np.percentile(times,95)):.2f}"
            )
        print()

    print("=" * W)
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n  Failed: {len(errors)} question(s)")
        for e in errors:
            print(f"    Q{e['question_id']:02d} [{e['pipeline']}]: {str(e['error'])[:70]}")
    else:
        print("  No errors")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run RLM vs Full-Context benchmark (20 questions, httpx codebase).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        choices=["both", "rlm", "full_context"],
        default="both",
        help="Which pipeline(s) to run (default: both)",
    )
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        choices=[3],
        default=[3],
        metavar="TIER",
        help="Which tier(s) to run (default: 3). Example: --tiers 3",
    )
    parser.add_argument(
        "--question",
        type=int,
        default=None,
        help="Run a single question by ID (useful for debugging)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output filename in results/ (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and start fresh",
    )
    args = parser.parse_args()

    # Select questions
    if args.question:
        questions = [q for q in ALL_QUESTIONS if q.id == args.question]
        if not questions:
            print(f"ERROR: Question ID {args.question} not found.")
            raise SystemExit(1)
        logger.info("Single-question mode: Q%d", args.question)
    else:
        questions = [q for q in ALL_QUESTIONS if q.tier in args.tiers]
        logger.info(
            "Running tiers %s: %d questions",
            sorted(args.tiers), len(questions)
        )

    if args.fresh:
        p = RESULTS_DIR / args.output
        if p.exists():
            p.unlink()
            logger.info("Cleared checkpoint: %s", p)

    asyncio.run(run_benchmark(
        questions   = questions,
        run_rlm     = args.pipeline in ("both", "rlm"),
        run_full_context = args.pipeline in ("both", "full_context"),
        output_file = args.output,
    ))
