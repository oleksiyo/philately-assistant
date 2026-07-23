"""Automated ingestion pipeline: Wikipedia categories -> chunked corpus.

Usage:
    uv run ingestion/ingest.py
    uv run ingestion/ingest.py --max-articles 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from wikipedia_source import (
    DEFAULT_CATEGORIES,
    collect_article_titles,
    fetch_article,
    make_wiki,
)
from chunking import chunk_article

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia philately articles into a chunked corpus."
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-articles", type=int, default=400)
    parser.add_argument("--language", default="en")
    parser.add_argument("--out", default=str(DATA_DIR / "chunks.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wiki = make_wiki(args.language)

    print(f"Collecting article titles from categories: {args.categories}")
    titles = collect_article_titles(
        wiki,
        args.categories,
        max_depth=args.max_depth,
        max_articles=args.max_articles,
    )
    print(f"Found {len(titles)} candidate articles")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_articles = 0
    n_chunks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for title in tqdm(sorted(titles), desc="Fetching + chunking"):
            article = fetch_article(wiki, title)
            if article is None:
                continue
            chunks = chunk_article(article)
            if not chunks:
                continue
            n_articles += 1
            n_chunks += len(chunks)
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Ingested {n_articles} articles into {n_chunks} chunks -> {out_path}")


if __name__ == "__main__":
    main()
