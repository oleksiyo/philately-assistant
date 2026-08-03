"""Query rewriting: reformulate the user's question into a better search query
before retrieval (expand philatelic jargon/abbreviations, drop filler words).
"""

from __future__ import annotations

from rag.llm import DEFAULT_MODEL, answer as llm_answer

SYSTEM_PROMPT = """\
You rewrite user questions into a clear, standalone search query for a document \
retrieval system about philately (stamp collecting). Expand abbreviations and \
philatelic jargon, remove filler words and phrasing like "can you tell me", keep \
it concise. Return only the rewritten query, nothing else. If the question is \
already a good search query, return it unchanged.
"""


def rewrite_query(question: str, model: str = DEFAULT_MODEL) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    rewritten = llm_answer(messages, model=model, temperature=0)
    return rewritten.strip() or question
