"""Prompt construction for the philately RAG assistant.

Two system-prompt variants are kept side by side ("strict" vs "open") so
LLM evaluation can compare them and pick a winner, per the project's
LLM-evaluation criterion.
"""

from __future__ import annotations

SYSTEM_PROMPTS = {
    "strict": (
        "You are a philately (stamp collecting) reference assistant. "
        "Answer the user's question using ONLY the information in the CONTEXT below. "
        "If the context does not contain the answer, say you don't have enough "
        "information instead of guessing. Cite the source article title(s) you used. "
        "Be concise and precise with philatelic terminology."
    ),
    "open": (
        "You are a philately (stamp collecting) reference assistant. "
        "Use the CONTEXT below as your primary source, and you may supplement it with "
        "your own general knowledge if the context is incomplete. Cite the source "
        "article title(s) you used from the context. Be concise and precise with "
        "philatelic terminology."
    ),
}

DEFAULT_VARIANT = "strict"


def format_context(chunks: list[dict]) -> str:
    blocks = [
        f"[{i}] {chunk['title']} — {chunk['section']}\n{chunk['text']}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_messages(
    question: str, chunks: list[dict], variant: str = DEFAULT_VARIANT
) -> list[dict]:
    if variant not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt variant: {variant!r}")

    context = format_context(chunks)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    return [
        {"role": "system", "content": SYSTEM_PROMPTS[variant]},
        {"role": "user", "content": user_content},
    ]
