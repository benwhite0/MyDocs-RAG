import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # repo root


def _load_yaml(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}


_cfg = _load_yaml(ROOT / "config.yaml")


def _get(path, default=None):
    keys = path.split(".")
    cur = _cfg
    for k in keys:
        if cur is None:
            break
        cur = cur.get(k)
    return os.getenv(path.replace(".", "_").upper(), cur if cur is not None else default)


@dataclass
class Settings:
    embedding_model: str = _get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_tokens: int = int(_get("chunk.tokens", 900))
    chunk_overlap: int = int(_get("chunk.overlap", 120))
    k: int = int(_get("retrieval.k", 5))
    bm25_weight: float = float(_get("retrieval.bm25_weight", 0.0))
    rerank_enabled: bool = str(_get("reranker.enabled", "false")).lower() == "true"
    rerank_model: str = _get("reranker.model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    raw_dir: Path = Path(_get("paths.raw", "data/raw"))
    processed_dir: Path = Path(_get("paths.processed", "data/processed"))
    index_dir: Path = Path(_get("paths.index", "data/processed/index"))
    api_host: str = _get("api.host", "0.0.0.0")
    api_port: int = int(_get("api.port", 8000))
    # LLM
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY", None)
    openai_model: str = _get("llm.model", "gpt-4o-mini")
    max_context_chars: int = int(_get("llm.max_context_chars", 12000))
    temperature: float = float(_get("llm.temperature", 0.0))


settings = Settings()
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
