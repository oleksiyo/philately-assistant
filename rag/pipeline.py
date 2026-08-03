"""End-to-end RAG flow: retrieve -> build prompt -> call LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass

from rag.llm import DEFAULT_MODEL, answer as llm_answer
from rag.prompt import DEFAULT_VARIANT, build_messages
from rag.retrieval import Retriever

TOP_K = 5


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[dict]
    retrieval_method: str
    prompt_variant: str
    model: str
    latency_seconds: float


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
    ) -> RagAnswer:
        start = time.monotonic()
        chunks = self.retriever.search(question, method=retrieval_method, num_results=top_k)
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
        )
