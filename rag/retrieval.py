"""Retrieval strategies over the chunk corpus: keyword-only, vector-only, hybrid.

Hybrid combines keyword and vector rankings via Reciprocal Rank Fusion (RRF),
since minsearch doesn't expose comparable raw scores across the two indexes.

Default is "vector": eval/retrieval_eval.py on the 200-question golden set at
top_k=5 showed vector beating both keyword and RRF-hybrid on hit-rate and MRR
(see data/retrieval_eval_results.json) — RRF dilutes vector's stronger ranking
with keyword's weaker one here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.index import build_keyword_index, build_vector_index, embed_query, load_chunks

TOP_K = 10
RRF_K = 60
HYBRID_CANDIDATE_POOL = 30


class Retriever:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.keyword_index = build_keyword_index(self.chunks)
        self.vector_index = build_vector_index(self.chunks)

    def search_keyword(self, query: str, num_results: int = TOP_K) -> list[dict]:
        return self.keyword_index.search(query, num_results=num_results)

    def search_vector(self, query: str, num_results: int = TOP_K) -> list[dict]:
        return self.vector_index.search(embed_query(query), num_results=num_results)

    def search_hybrid(
        self,
        query: str,
        num_results: int = TOP_K,
        candidate_pool: int = HYBRID_CANDIDATE_POOL,
    ) -> list[dict]:
        keyword_results = self.search_keyword(query, num_results=candidate_pool)
        vector_results = self.search_vector(query, num_results=candidate_pool)

        rrf_scores: dict[str, float] = {}
        docs_by_id: dict[str, dict] = {}
        for results in (keyword_results, vector_results):
            for rank, doc in enumerate(results, start=1):
                chunk_id = doc["chunk_id"]
                docs_by_id[chunk_id] = doc
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:num_results]
        return [docs_by_id[chunk_id] for chunk_id in ranked_ids]

    def search(self, query: str, method: str = "vector", num_results: int = TOP_K) -> list[dict]:
        if method == "keyword":
            return self.search_keyword(query, num_results)
        if method == "vector":
            return self.search_vector(query, num_results)
        if method == "hybrid":
            return self.search_hybrid(query, num_results)
        raise ValueError(f"Unknown retrieval method: {method!r}")
