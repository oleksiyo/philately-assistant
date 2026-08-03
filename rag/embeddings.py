"""Local text embeddings for the vector index (no API key required)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    model = get_model()
    return np.asarray(
        model.encode(texts, show_progress_bar=len(texts) > 50, normalize_embeddings=True)
    )
