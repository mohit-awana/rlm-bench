"""
chunker.py
----------
Deterministic chunker for Python source code corpora.

Strategy
--------
One chunk = one Python file. This is the natural unit for code:
  - Each file has a clear identity (module name)
  - Import statements in the file reveal cross-file dependencies
  - The TOC is the list of files with their module paths
  - chunk_id maps to a specific file — stable across runs

Why file-level chunking beats sub-file chunking for this benchmark
-----------------------------------------------------------------
The benchmark tests cross-file reasoning: "function A in file X calls
function B in file Y." If we split files into sub-chunks, the import
statements and function definitions may land in different chunks,
losing the within-file call structure. One file = one chunk preserves
the complete function-to-function relationships within each file.

For large files (>chunk_size tokens), we keep them as one chunk anyway
— a function definition should never be split across chunks.

Typical usage
-------------
    from rlm.chunker import Chunker
    chunks = Chunker.from_dir("data/httpx_src")
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id:       int
    page_start:     int          # file index (1-based) — kept for interface compat
    page_end:       int          # same as page_start for file chunks
    section_title:  str          # module label e.g. "_client"
    text:           str          # full file content
    file_path:      str = ""     # relative path e.g. "_client.py"
    token_estimate: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_estimate = len(self.text.split()) * 4 // 3  # word * 4/3 ≈ tokens


class Chunker:
    """
    Produces one Chunk per Python source file in a directory.
    """

    @staticmethod
    def from_dir(
        src_dir: str,
        chunk_size: int = 500,   # kept for interface compat, not used
        extensions: tuple = (".py",),
        exclude: tuple = ("__pycache__", ".pyc", "test_", "_test.py"),
    ) -> List[Chunk]:
        """
        Walk src_dir and return one Chunk per .py file.

        Parameters
        ----------
        src_dir    : root directory of the source code
        chunk_size : ignored (kept for interface compat with runner.py)
        extensions : file extensions to include
        exclude    : path fragments to exclude
        """
        src_path = pathlib.Path(src_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"Source directory not found: {src_dir}")

        # Collect files deterministically
        py_files = sorted(
            f for f in src_path.rglob("*")
            if f.suffix in extensions
            and not any(ex in str(f) for ex in exclude)
            and f.is_file()
        )

        if not py_files:
            raise ValueError(f"No Python files found in {src_dir}")

        chunks: List[Chunk] = []
        for idx, filepath in enumerate(py_files):
            rel_path = filepath.relative_to(src_path)

            # Build module-style section title
            # e.g. "_client.py" -> "_client"
            # e.g. "_transports/default.py" -> "_transports.default"
            parts = list(rel_path.parts)
            if parts[-1] == "__init__.py":
                module = ".".join(parts[:-1]) if len(parts) > 1 else "root"
            else:
                parts[-1] = parts[-1].replace(".py", "")
                module = ".".join(parts)

            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning("Could not read %s: %s", filepath, e)
                continue

            chunks.append(Chunk(
                chunk_id      = idx,
                page_start    = idx + 1,
                page_end      = idx + 1,
                section_title = module,
                text          = text,
                file_path     = str(rel_path),
            ))

        logger.info(
            "Chunked %d Python files from %s (~%d tokens total)",
            len(chunks),
            src_dir,
            sum(c.token_estimate for c in chunks),
        )
        return chunks

    @staticmethod
    def from_pdf(pdf_path: str, chunk_size: int = 500) -> List[Chunk]:
        """
        Legacy PDF chunker — raises a clear error directing to from_dir().
        Kept so runner.py imports don't break; will not be called in
        the codebase benchmark.
        """
        raise NotImplementedError(
            "PDF chunking is no longer used. "
            "Use Chunker.from_dir('data/httpx_src') instead. "
            "Update config.py: pdf_path -> src_dir."
        )

    @staticmethod
    def toc(chunks: List[Chunk]) -> str:
        """
        Build a compact table of contents string for the RLM engine.
        Format: chunk_id | file_path | module | ~tokens
        """
        lines = ["Python Source File Index", "=" * 60]
        for c in chunks:
            lines.append(
                f"  [{c.chunk_id:02d}] {c.file_path:<45} "
                f"({c.token_estimate:,} tokens)"
            )
        lines.append("=" * 60)
        lines.append(f"Total: {len(chunks)} files")
        return "\n".join(lines)


if __name__ == "__main__":
    chunks = Chunker.from_dir("data/httpx_src")
    print(Chunker.toc(chunks))
