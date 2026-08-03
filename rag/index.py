"""Build keyword and vector indexes over the ingested chunk corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from minsearch import Index, VectorSearch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.embeddings import embed

DATA_DIR = ROOT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_IDS_PATH = DATA_DIR / "embeddings_ids.json"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_keyword_index(chunks: list[dict]) -> Index:
    index = Index(text_fields=["title", "section", "text"])
    index.fit(chunks)
    return index


def _embedding_text(chunk: dict) -> str:
    return f"{chunk['title']} - {chunk['section']}\n{chunk['text']}"


def compute_embeddings(chunks: list[dict]) -> np.ndarray:
    return embed([_embedding_text(c) for c in chunks])


def build_vector_index(chunks: list[dict], use_cache: bool = True) -> VectorSearch:
    chunk_ids = [c["chunk_id"] for c in chunks]

    vectors = None
    if use_cache and EMBEDDINGS_PATH.exists() and EMBEDDINGS_IDS_PATH.exists():
        cached_ids = json.loads(EMBEDDINGS_IDS_PATH.read_text())
        if cached_ids == chunk_ids:
            vectors = np.load(EMBEDDINGS_PATH)

    if vectors is None:
        vectors = compute_embeddings(chunks)
        if use_cache:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.save(EMBEDDINGS_PATH, vectors)
            EMBEDDINGS_IDS_PATH.write_text(json.dumps(chunk_ids))

    index = VectorSearch()
    index.fit(vectors, chunks)
    return index


def embed_query(query: str) -> np.ndarray:
    return embed([query])[0]


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    keyword_index = build_keyword_index(chunks)
    print("Keyword index ready")

    vector_index = build_vector_index(chunks)
    print("Vector index ready")

    query = "What is a tete-beche stamp?"
    print(f"\nKeyword search for: {query!r}")
    for doc in keyword_index.search(query, num_results=3):
        print(" -", doc["title"], "/", doc["section"])

    print(f"\nVector search for: {query!r}")
    for doc in vector_index.search(embed_query(query), num_results=3):
        print(" -", doc["title"], "/", doc["section"])
