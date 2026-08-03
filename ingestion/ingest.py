"""Automated ingestion pipeline: Wikipedia categories -> chunked corpus.

Orchestrated with dlt: article chunks are extracted as a dlt resource and
loaded into a local DuckDB dataset (schema-managed, "replace" write
disposition on each run), then exported to data/chunks.jsonl — the flat
format the rest of the app (rag/, eval/) reads.

Usage:
    uv run ingestion/ingest.py
    uv run ingestion/ingest.py --max-articles 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dlt
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
        description="Ingest Wikipedia philately articles into a chunked corpus via a dlt pipeline."
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-articles", type=int, default=400)
    parser.add_argument("--language", default="en")
    parser.add_argument("--out", default=str(DATA_DIR / "chunks.jsonl"))
    return parser.parse_args()


@dlt.resource(name="chunks", write_disposition="replace")
def wikipedia_chunks(wiki, titles: list[str]):
    for title in tqdm(titles, desc="Fetching + chunking"):
        article = fetch_article(wiki, title)
        if article is None:
            continue
        yield from chunk_article(article)


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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = dlt.pipeline(
        pipeline_name="philately_ingestion",
        destination=dlt.destinations.duckdb(str(DATA_DIR / "philately.duckdb")),
        dataset_name="raw_chunks",
    )
    load_info = pipeline.run(wikipedia_chunks(wiki, sorted(titles)))
    print(load_info)

    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT chunk_id, title, section, url, text FROM chunks"
        ) as cursor:
            rows = cursor.fetchall()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_articles = set()
    with out_path.open("w", encoding="utf-8") as f:
        for chunk_id, title, section, url, text in rows:
            f.write(
                json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "title": title,
                        "section": section,
                        "url": url,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_articles.add(title)

    print(f"Ingested {len(n_articles)} articles into {len(rows)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
