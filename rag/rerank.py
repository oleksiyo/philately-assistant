"""Cross-encoder re-ranking of a retrieved candidate pool.

Local, free (no API calls) — sentence-transformers CrossEncoder scores each
(query, chunk) pair directly, which is more accurate than bi-encoder
similarity but too slow to run over the whole corpus, hence "retrieve a
candidate pool, then rerank" rather than "rerank everything".
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _rerank_text(chunk: dict) -> str:
    return f"{chunk['title']} - {chunk['section']}\n{chunk['text']}"


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return chunks
    model = get_model()
    pairs = [(query, _rerank_text(c)) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
