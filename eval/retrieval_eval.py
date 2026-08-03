"""Compare keyword / vector / hybrid retrieval on the golden Q&A set.

Usage:
    uv run python -m eval.retrieval_eval
    uv run python -m eval.retrieval_eval --top-k 5   # match rag/pipeline.py's TOP_K
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.retrieval import Retriever

GOLDEN_PATH = ROOT_DIR / "data" / "golden_qa.jsonl"
METHODS = ["keyword", "vector", "hybrid"]
TOP_K = 10


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(retriever: Retriever, golden: list[dict], method: str, top_k: int = TOP_K) -> dict:
    hits = 0
    reciprocal_ranks = []
    for item in tqdm(golden, desc=f"Evaluating {method}"):
        results = retriever.search(item["question"], method=method, num_results=top_k)
        rank = next(
            (i for i, doc in enumerate(results, start=1) if doc["chunk_id"] == item["chunk_id"]),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(golden)
    return {
        "method": method,
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval methods on the golden set.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--out", default=str(ROOT_DIR / "data" / "retrieval_eval_results.json"))
    return parser.parse_args()


def main() -> None:
    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"{GOLDEN_PATH} not found. Run `uv run python -m eval.generate_golden` first."
        )

    args = parse_args()
    golden = load_golden()
    print(f"Loaded {len(golden)} golden questions (top_k={args.top_k})")

    retriever = Retriever()
    results = [evaluate(retriever, golden, method, top_k=args.top_k) for method in METHODS]

    print(f"\n{'method':<10} {'hit_rate':>10} {'mrr':>10}")
    for r in results:
        print(f"{r['method']:<10} {r['hit_rate']:>10.3f} {r['mrr']:>10.3f}")

    best = max(results, key=lambda r: r["mrr"])
    print(f"\nBest approach by MRR: {best['method']}")

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"top_k": args.top_k, "n_questions": len(golden), "results": results}, indent=2)
    )
    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
