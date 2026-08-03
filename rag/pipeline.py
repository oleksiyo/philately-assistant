"""End-to-end RAG flow: rewrite query -> retrieve -> rerank -> build prompt -> call LLM.

Defaults are set from eval/rerank_rewrite_eval.py on the 200-question golden
set (see data/rerank_rewrite_eval_results.json): reranking is a clear win
(MRR 0.727 -> 0.922) so it's on by default; query rewriting alone slightly
*hurt* MRR (0.727 -> 0.691) here, because the golden questions were LLM-
generated from their answer chunk and already closely match its wording, so
it's off by default (still available as a toggle for messier real-world
phrasing the benchmark doesn't capture).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rag.llm import DEFAULT_MODEL, answer as llm_answer
from rag.prompt import DEFAULT_VARIANT, build_messages
from rag.query_rewrite import rewrite_query
from rag.rerank import rerank
from rag.retrieval import Retriever

TOP_K = 5
RERANK_CANDIDATE_POOL = 20


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[dict]
    retrieval_method: str
    prompt_variant: str
    model: str
    latency_seconds: float
    search_query: str
    reranked: bool


class RagPipeline:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    def answer(
        self,
        question: str,
        retrieval_method: str = "vector",
        prompt_variant: str = DEFAULT_VARIANT,
        model: str = DEFAULT_MODEL,
        top_k: int = TOP_K,
        use_query_rewrite: bool = False,
        use_rerank: bool = True,
    ) -> RagAnswer:
        start = time.monotonic()

        search_query = rewrite_query(question, model=model) if use_query_rewrite else question

        candidate_k = RERANK_CANDIDATE_POOL if use_rerank else top_k
        chunks = self.retriever.search(
            search_query, method=retrieval_method, num_results=candidate_k
        )
        if use_rerank:
            # Rerank against the original question: cross-encoders trained on
            # natural Q&A pairs (e.g. MS MARCO) match that phrasing better
            # than a retrieval-optimized rewritten query.
            chunks = rerank(question, chunks, top_k=top_k)

        messages = build_messages(question, chunks, variant=prompt_variant)
        answer_text = llm_answer(messages, model=model)
        latency = time.monotonic() - start

        return RagAnswer(
            question=question,
            answer=answer_text,
            sources=[
                {"title": c["title"], "section": c["section"], "url": c["url"]} for c in chunks
            ],
            retrieval_method=retrieval_method,
            prompt_variant=prompt_variant,
            model=model,
            latency_seconds=latency,
            search_query=search_query,
            reranked=use_rerank,
        )
