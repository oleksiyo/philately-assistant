"""Generate a golden question -> chunk_id set for retrieval evaluation.

For a random sample of chunks, ask an LLM to write one natural-language
question that the chunk answers. Requires OPENAI_API_KEY in .env.

Usage:
    uv run python -m eval.generate_golden
    uv run python -m eval.generate_golden --n-samples 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.index import load_chunks

DATA_DIR = ROOT_DIR / "data"
OUT_PATH = DATA_DIR / "golden_qa.jsonl"

MODEL = "gpt-4o-mini"
MIN_CHARS = 300
MIN_SENTENCES = 2

PROMPT_TEMPLATE = """\
You are helping build a test set for a stamp-collecting (philately) Q&A system.

Below is one excerpt from a Wikipedia article titled "{title}" (section: "{section}").
Write exactly one natural-language question that a curious reader could ask, whose \
answer is fully contained in this excerpt. Ask it the way a person would, not by \
quoting the text. Do not mention "the excerpt" or "the text". Return only the \
question, nothing else.

Excerpt:
\"\"\"
{text}
\"\"\"
"""


def is_prose_chunk(chunk: dict) -> bool:
    text = chunk["text"]
    if len(text) < MIN_CHARS:
        return False
    sentence_endings = sum(text.count(c) for c in ".!?")
    return sentence_endings >= MIN_SENTENCES


def generate_question(client: OpenAI, chunk: dict) -> str | None:
    prompt = PROMPT_TEMPLATE.format(
        title=chunk["title"], section=chunk["section"], text=chunk["text"]
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    question = response.choices[0].message.content
    return question.strip() if question else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden Q&A set for retrieval eval.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    chunks = load_chunks()
    candidates = [c for c in chunks if is_prose_chunk(c)]
    print(f"{len(candidates)}/{len(chunks)} chunks pass the prose filter")

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.n_samples, len(candidates)))

    client = OpenAI()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in tqdm(sample, desc="Generating questions"):
            question = generate_question(client, chunk)
            if not question:
                continue
            f.write(
                json.dumps(
                    {
                        "question": question,
                        "chunk_id": chunk["chunk_id"],
                        "title": chunk["title"],
                        "section": chunk["section"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_ok += 1

    print(f"Wrote {n_ok} golden questions -> {out_path}")


if __name__ == "__main__":
    main()
