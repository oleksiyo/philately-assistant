"""Cleaning and chunking of Wikipedia articles into retrieval-ready chunks."""

from __future__ import annotations

SKIP_SECTIONS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
    "citations",
    "sources",
}

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200


def flatten_sections(sections, parent_titles: tuple[str, ...] = ()):
    """Yield (section_path, text) for every section with non-empty own text.

    `section.text` on a wikipediaapi section is only that section's own
    paragraphs, not its subsections', so recursing does not duplicate text.
    """
    for section in sections:
        title = section.title.strip()
        path = parent_titles + (title,)
        if title.lower() not in SKIP_SECTIONS:
            text = section.text.strip()
            if text:
                yield path, text
            yield from flatten_sections(section.sections, path)


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Group paragraphs into chunks up to `max_chars`, dropping tiny leftovers."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 1 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n{para}".strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    big_enough = [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]
    if big_enough:
        return big_enough
    return [text.strip()] if text.strip() else []


def chunk_article(article) -> list[dict]:
    """Turn a wikipediaapi WikipediaPage into a list of chunk dicts."""
    chunks: list[dict] = []
    idx = 0

    def add(section_path: tuple[str, ...], body: str) -> None:
        nonlocal idx
        for piece in split_into_chunks(body):
            chunks.append(
                {
                    "chunk_id": f"{article.title}::{idx}",
                    "title": article.title,
                    "section": " > ".join(section_path) if section_path else "Lead",
                    "url": article.fullurl,
                    "text": piece,
                }
            )
            idx += 1

    summary = article.summary.strip()
    if summary:
        add((), summary)

    for path, text in flatten_sections(article.sections):
        add(path, text)

    return chunks
