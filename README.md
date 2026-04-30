# RLM-Bench

**Recursive Language Model Navigation as a Drop-in Tool for Code Repository QA**

> A benchmark and reference implementation inspired by Zhang, Kraska, and Khattab (2025). It packages RLM as a ToolContract-compatible tool and compares it against full-context generation on the `httpx` Python library.

[![arXiv](https://img.shields.io/badge/arXiv-2512.24601-b31b1b.svg)](https://arxiv.org/abs/2512.24601)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

RLM-Bench is a code-repository QA benchmark and a reference implementation of RLM as a standard agent tool. The repo packages the recursive navigation loop behind a `ToolContract` interface so it can be used like any other tool in an existing agent stack.

The benchmark compares two strategies over the same `httpx` source tree:

- RLMTool, which reads a table of contents, fetches relevant chunks, stores partial findings, and reasons over external memory.
- A full-context baseline, implemented in `rag/vanilla_rag.py` for historical compatibility, which stuffs the entire corpus into one prompt and answers in a single model call.

---

## Reported Results

The manuscript reports the following aggregate results on 20 cross-file questions:

| Metric | RLM | Full-context |
|---|---:|---:|
| Composite score | 0.677 | 0.668 |
| Pass rate | 80% | 84% |
| Tokens per query | 18,530 | 66,305 |
| Average latency | 7.03 s | 12.67 s |

Additional case-study result from the manuscript:

- On Q16, full-context faithfulness dropped to 0.00 and composite score to 0.334.
- On the same question, RLM reached faithfulness 1.00 and composite score 0.593.

The headline result is that RLM is statistically comparable in accuracy while being materially cheaper and faster.

---

## Why This Matters

The benchmark questions are designed so the answer is not usually in the obvious file.

For example, a question about redirect handling can require `_client.py` for the redirect logic and `_urls.py` for the URL-origin checks. A similarity-based retriever can easily hit the right vocabulary but miss the actual rule. Full-context generation can also fail when the relevant files sit in the middle of a long context.

RLM avoids both failure modes by treating the repository as an external environment:

- read the TOC first
- fetch only the relevant files
- record partial findings as it goes
- synthesize the final answer from the retrieved evidence

---

## Architecture

```text
Agent
  └── RLMTool.execute()
        └── RLMEngine  (recursive OpenAI tool-call loop)
              └── ExternalMemory  (TOC, chunks, findings, token counts)
                    ├── get_toc()
                    ├── get_chunk(chunk_id)
                    ├── record_result()
                    └── summary()
```

Key implementation details:

- `tools/rlm_tool.py` exposes `RLMTool` and the `ToolRegistry` wrapper that injects a shared engine instance.
- `rlm/engine.py` runs the recursive OpenAI tool-call loop and auto-loads the TOC on turn 0.
- `rlm/memory.py` stores the corpus outside the model context and tracks visited chunks, partial results, and token usage.
- `rlm/chunker.py` makes one Python file into one chunk and builds a deterministic table of contents.
- `rag/vanilla_rag.py` is the full-context baseline, kept under the historical module name `vanilla_rag.py` for compatibility.
- `benchmark/runner.py` executes both pipelines and checkpoints results after each question.
- `benchmark/evaluator.py` computes the DeepEval cascade summary.

---

## Corpus

- `httpx` Python HTTP library, version `0.28.1`
- Source tree copied into `data/httpx_src/` by `scripts/download_data.sh`
- Paper-reported corpus size: 23 Python source files and about 71,087 tokens
- License for the corpus: BSD 3-Clause

The corpus is intentionally code-shaped rather than document-shaped: one file is one chunk, and the TOC is the file index.

---

## Benchmark

The current benchmark contains 20 manually written Tier 3 questions:

- all questions require cross-file call-chain reasoning
- answers depend on connecting 2 to 4 modules
- ground truths are original and traceable to file names and function signatures
- examples cover redirects, auth, digest auth, response decoding, streaming, exceptions, and URL parsing

The question set lives in `benchmark/questions.py`.

---

## Evaluation

The evaluator uses three signals:

- Faithfulness, judged with `gpt-4.1-nano`
- Code QA Accuracy, judged with `gpt-4o-mini`
- Symbol Recall, computed deterministically from the question and ground truth text

The composite score in `benchmark/evaluator.py` is:

```text
0.35 * Faithfulness
0.45 * Code QA Accuracy
0.20 * Symbol Recall
```

The evaluator checkpoints progress to `results/cascade_checkpoint.json` and prints a summary at the end.

---

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Copy the httpx source tree into data/httpx_src/
bash scripts/download_data.sh

# 3. Configure OpenAI access
cp .env.example .env
# Add OPENAI_API_KEY to .env

# 4. Run the benchmark
uv run python -m benchmark.runner --fresh

# 5. Score the results
uv run python -m benchmark.evaluator --input benchmark_results.json
```

The benchmark runner writes `results/benchmark_results.json`. The evaluator reads that file and writes its checkpoint to `results/cascade_checkpoint.json`.

Useful runner variants:

```bash
uv run python -m benchmark.runner --pipeline rlm
uv run python -m benchmark.runner --pipeline full_context --question 16
```

Optional local demo:

- `agent.py` is a small OpenAI API agent that uses `RLMTool`.

---

## Project Structure

```text
rlm-bench/
├── data/httpx_src/           # copied httpx source tree
├── rlm/
│   ├── chunker.py            # one file per chunk + TOC
│   ├── memory.py             # external memory, token tracking, run summary
│   └── engine.py             # recursive OpenAI loop
├── tools/
│   ├── contract.py           # ToolContract protocol
│   └── rlm_tool.py           # RLMTool + ToolRegistry
├── rag/
│   └── vanilla_rag.py        # full-context baseline (legacy module name)
├── benchmark/
│   ├── questions.py          # 20 Tier 3 cross-file questions
│   ├── runner.py             # benchmark harness with checkpointing
│   └── evaluator.py          # DeepEval scoring and summary
├── agent.py                  # optional Ollama demo agent
├── scripts/
│   └── download_data.sh      # copies httpx source from the installed package
├── results/                  # benchmark outputs and checkpoints
└── paper/
    ├── main.tex              # arXiv-ready manuscript source
    ├── references.bib        # bibliography
    └── arxiv_submission_metadata.txt
```

---

## Relation to the Original RLM Paper

The original RLM paper shows that recursive navigation can outperform naive long-context prompting on long-document benchmarks. This repository applies the same idea to code repository QA:

- the model sees the repository as an external environment
- retrieval is explicit rather than similarity-only
- the tool interface makes RLM usable inside a normal agent loop

The result is a drop-in pattern for existing codebases that need cross-file reasoning.

---

## Citation

```bibtex
@misc{awana2026rlmbench,
  author = {Mohit Awana},
  title = {RLM-Bench: Recursive Language Model Navigation as a Drop-in Tool for Code Repository QA},
  year = {2026},
  doi = {https://doi.org/10.5281/zenodo.19892314}
}
```

Original RLM paper:

```bibtex
@misc{zhang2025recursive,
  title  = {Recursive Language Models},
  author = {Alex L. Zhang and Tim Kraska and Omar Khattab},
  year   = {2025},
  eprint = {2512.24601},
  url    = {https://arxiv.org/abs/2512.24601}
}
```

---

## License

This repository is MIT licensed.

The `httpx` corpus is BSD 3-Clause licensed.
The benchmark questions and ground truth answers are original work and are intended to be reproducible from the public `httpx` repository.
