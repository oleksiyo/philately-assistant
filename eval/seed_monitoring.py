"""Seed the monitoring database with sample traffic for the Grafana dashboard.

Not part of the RAG pipeline itself — this generates realistic conversation
and feedback rows (spread over the past week) so the dashboard has data to
chart. Uses real LLM calls against the golden question set.

Usage:
    uv run python -m eval.seed_monitoring
    uv run python -m eval.seed_monitoring --n-samples 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.db import init_db, log_conversation, log_feedback
from rag.pipeline import RagPipeline

GOLDEN_PATH = ROOT_DIR / "data" / "golden_qa.jsonl"
RETRIEVAL_METHODS = ["vector", "vector", "vector", "hybrid", "keyword"]
PROMPT_VARIANTS = ["strict", "strict", "open"]
SPREAD_DAYS = 7


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed monitoring DB with sample RAG traffic.")
    parser.add_argument("--n-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"{GOLDEN_PATH} not found. Run `uv run python -m eval.generate_golden` first."
        )

    init_db()
    golden = load_golden()
    random.seed(args.seed)
    sample = random.sample(golden, min(args.n_samples, len(golden)))

    pipeline = RagPipeline()

    for item in tqdm(sample, desc="Seeding monitoring traffic"):
        method = random.choice(RETRIEVAL_METHODS)
        variant = random.choice(PROMPT_VARIANTS)
        result = pipeline.answer(item["question"], retrieval_method=method, prompt_variant=variant)

        created_at = datetime.now(timezone.utc) - timedelta(
            hours=random.uniform(0, 24 * SPREAD_DAYS)
        )
        conversation_id = log_conversation(
            question=result.question,
            answer=result.answer,
            retrieval_method=result.retrieval_method,
            prompt_variant=result.prompt_variant,
            model=result.model,
            latency_seconds=result.latency_seconds,
            sources=result.sources,
            created_at=created_at,
        )

        # Simulate feedback a real user base would leave: mostly positive,
        # some negative, many skipped.
        roll = random.random()
        if roll < 0.55:
            log_feedback(conversation_id, 1)
        elif roll < 0.70:
            log_feedback(conversation_id, -1)

    print(f"Seeded {len(sample)} conversations")


if __name__ == "__main__":
    main()
