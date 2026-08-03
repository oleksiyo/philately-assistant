"""OpenAI chat completion wrapper for the RAG assistant."""

from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = OpenAI()
    return _client


def answer(messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = 0.2) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""
