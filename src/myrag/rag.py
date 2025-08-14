import json
import logging
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .settings import settings

logger = logging.getLogger(__name__)


# OpenAI client is optional; only used if API key is present
try:
    from openai import OpenAI  # openai>=1.0 SDK

    logger.info("Successfully imported OpenAI SDK")
except Exception as e:  # pragma: no cover
    logger.error("Failed to import OpenAI SDK: %s", e)
    OpenAI = None  # type: ignore

# Cross-encoder for reranking (optional)
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        self.index = faiss.read_index(str(settings.index_dir / "faiss.index"))
        self.metas: list[dict] = json.loads((settings.index_dir / "metas.json").read_text())
        self.texts: list[str] = [
            json.loads(line)["text"] if line else ""
            for line in (settings.processed_dir / "chunks.jsonl").read_text().splitlines()
        ]
        # BM25 corpus (optional hybrid)
        self._bm25 = (
            BM25Okapi([t.split() for t in self.texts]) if settings.bm25_weight > 0 else None
        )
        # lazily created reranker
        self._reranker = None

    def _get_reranker(self):
        if not settings.rerank_enabled:
            return None
        if CrossEncoder is None:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.rerank_model)
        return self._reranker

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        # retrieve a bit more to allow reranking to help
        retrieve_n = max(k, 15) if settings.rerank_enabled else k
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(
            np.float32
        )
        sims, idxs = self.index.search(q_emb, retrieve_n)
        sim_scores = sims[0].tolist()
        idxs = idxs[0].tolist()

        hybrid = []
        if self._bm25:
            bm_scores = self._bm25.get_scores(query.split())
            for i, score in enumerate(bm_scores):
                hybrid.append((i, settings.bm25_weight * float(score)))
        dense = [(i, float(sim_scores[j])) for j, i in enumerate(idxs)]
        # sum scores by idx
        score_map: dict[int, float] = {}
        for i, s in dense + hybrid:
            score_map[i] = score_map.get(i, 0.0) + s
        ranked = sorted(score_map.items(), key=lambda t: t[1], reverse=True)[:retrieve_n]
        candidates: list[dict[str, Any]] = []
        for i, s in ranked:
            candidates.append(
                {
                    "text": self.texts[i],
                    "score": s,
                    "meta": self.metas[i] if i < len(self.metas) else {},
                }
            )

        # optional rerank
        reranker = self._get_reranker()
        if reranker is not None and candidates:
            pairs = [(query, c["text"]) for c in candidates]
            scores = reranker.predict(pairs)
            for c, rs in zip(candidates, scores):
                c["score"] = float(rs)
            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:k]


def make_prompt(query: str, passages: list[dict[str, Any]]) -> str:
    # Trim context to a safe character budget.
    # Models and tokenizers vary; keep prompts manageable.
    context_parts: list[str] = []
    remaining = max(1000, settings.max_context_chars)
    for p in passages:
        t = p["text"][: min(len(p["text"]), remaining)]
        if not t:
            continue
        context_parts.append(t)
        remaining -= len(t) + 5
        if remaining <= 0:
            break
    context = "\n\n---\n\n".join(context_parts)
    return (
        "You are a helpful assistant. Answer concisely using only the provided context.\n"
        "- If the answer is not in the context, say you don't know.\n"
        "- Cite quotes from context when helpful.\n\n"
        f"Question: {query}\n\n"
        "Context:\n"
        f"{context}\n"
    )


def naive_answer(query: str, passages: list[dict[str, Any]]) -> str:
    # placeholder generator—concatenate top chunks. Useful before wiring an LLM.
    return passages[0]["text"][:500] if passages else "I don't know."


def generate_answer(query: str, passages: list[dict[str, Any]]) -> str:
    """Use OpenAI Chat Completions with retrieval-augmented prompt.

    Falls back to naive_answer if no API key or client is unavailable.
    """
    if OpenAI is None or not settings.openai_api_key:
        logger.info(
            "Answer generation: fallback to simple top-chunk (no OpenAI client or API key)."
        )
        return naive_answer(query, passages)

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = make_prompt(query, passages)

    # Use a system + user message structure; temperature low for factual QA
    messages = [
        {"role": "system", "content": "You are a concise and accurate assistant for RAG QA."},
        {"role": "user", "content": prompt},
    ]
    try:
        logger.info("Making API call to OpenAI with model=%s...", settings.openai_model)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=settings.temperature,
        )
        logger.info("Successfully received response from OpenAI")
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("OpenAI API call failed with error: %s", str(e))
        logger.info("Falling back to simple top-chunk response")
        return naive_answer(query, passages)
