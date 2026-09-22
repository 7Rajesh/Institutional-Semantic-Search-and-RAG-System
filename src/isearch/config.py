"""Settings. Every field can be overridden with an ISEARCH_<FIELD_NAME> environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

_CASTS = {
    "int": int,
    "float": float,
    "str": str,
    "bool": lambda v: v.strip().lower() in {"1", "true", "yes", "on"},
}


@dataclass
class Settings:
    home: Path = field(default_factory=lambda: Path(os.environ.get("ISEARCH_HOME", ".")).resolve())

    # models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # chunking (words)
    chunk_words: int = 180
    chunk_overlap_words: int = 30
    min_chunk_words: int = 15

    # retrieval
    bm25_top_k: int = 30
    dense_top_k: int = 30
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    rerank_input_k: int = 25
    final_top_k: int = 5

    # grading / routing
    relevance_threshold: float = 0.0     # cross-encoder logit. Calibrate with `isearch eval`.
    relevance_window: float = 8.0
    route_margin: float = 0.04
    max_attempts: int = 3

    # conversation
    history_turns: int = 3

    # local LLM (Ollama)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_verify: bool = True

    # service
    admin_key: str = ""                  # if set, admin/upload endpoints require header X-API-Key
    max_upload_mb: int = 20

    @property
    def document_dir(self) -> Path:
        return self.home / "data" / "documents"

    @property
    def artifact_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def db_path(self) -> Path:
        return self.artifact_dir / "assistant.db"

    def ensure_dirs(self) -> None:
        self.document_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        kwargs = {}
        for f in fields(cls):
            if f.name == "home":
                continue
            raw = os.environ.get("ISEARCH_" + f.name.upper())
            if raw is not None:
                kwargs[f.name] = _CASTS[str(f.type)](raw)
        kwargs.update(overrides)
        if "home" in kwargs:
            kwargs["home"] = Path(kwargs["home"]).resolve()
        return cls(**kwargs)
