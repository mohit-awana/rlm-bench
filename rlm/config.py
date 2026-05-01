"""
config.py
---------
Central configuration — loads from .env via python-dotenv.
All model names and API settings live here.
Import this module anywhere instead of hardcoding model strings.

Usage
-----
    from rlm.config import cfg
    client = cfg.openai_client()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env if present — must happen before any os.getenv calls
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass   # python-dotenv optional — env vars may be set externally


@dataclass(frozen=True)
class Config:
    model:        str = "gpt-4.1-nano"

    # Embedding model (compatibility setting)
    embed_model:  str = "text-embedding-3-small"

    # Retrieval depth (compatibility setting)
    top_k:        int = 5

    chunk_size:   int = 500

    # Source code directory (legacy field name retained as pdf_path)
    pdf_path:     str = "data/httpx_src"

    judge_model:  str = "gpt-4o"

    # OpenAI API key (read from env)
    @property
    def api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. "
                "Copy .env.example to .env and add your key."
            )
        return key

    def openai_client(self):
        """Return a configured synchronous OpenAI client."""
        from openai import OpenAI
        return OpenAI(api_key=self.api_key)

    def async_openai_client(self):
        """Return a configured async OpenAI client."""
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self.api_key)


# Singleton — override via env vars at runtime
cfg = Config(
    model       = os.getenv("RLM_MODEL",    "gpt-4.1-nano"),
    embed_model = os.getenv("EMBED_MODEL",  "text-embedding-3-small"),
    top_k       = int(os.getenv("RAG_TOP_K", "5")),
    chunk_size  = int(os.getenv("CHUNK_SIZE", "300")),
    pdf_path    = os.getenv("PDF_PATH",     "data/httpx_src"),
    judge_model = os.getenv("DEEPEVAL_MODEL", "gpt-4o"),
)
