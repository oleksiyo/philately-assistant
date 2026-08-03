"""Ablation eval: does query rewriting / cross-encoder re-ranking improve
retrieval quality on top of the winning "vector" method?

Usage:
    uv run python -m eval.rerank_rewrite_eval
    uv run python -m eval.rerank_rewrite_eval --n-samples 20   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.query_rewrite import rewrite_query
from rag.rerank import rerank
from rag.retrieval import Retriever

GOLDEN_PATH = ROOT_DIR / "data" / "golden_qa.jsonl"
OUT_PATH = ROOT_DIR / "data" / "rerank_rewrite_eval_results.json"
METHOD = "vector"
TOP_K = 5
CANDIDATE_POOL = 20

VARIANTS = [
    ("baseline", False, False),
    ("+rewrite", True, False),
    ("+rerank", False, True),
    ("+rewrite+rerank", True, True),
]


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(
    retriever: Retriever, golden: list[dict], use_rewrite: bool, use_rerank: bool
) -> dict:
    hits = 0
    reciprocal_ranks = []
    for item in tqdm(golden, desc=f"rewrite={use_rewrite} rerank={use_rerank}"):
        question = item["question"]
        search_query = rewrite_query(question) if use_rewrite else question

        candidate_k = CANDIDATE_POOL if use_rerank else TOP_K
        chunks = retriever.search(search_query, method=METHOD, num_results=candidate_k)
        if use_rerank:
            chunks = rerank(question, chunks, top_k=TOP_K)

        rank = next(
            (i for i, c in enumerate(chunks, start=1) if c["chunk_id"] == item["chunk_id"]),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(golden)
    return {
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation eval for query rewriting + reranking.")
    parser.add_argument("--n-samples", type=int, default=None, help="subsample the golden set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def main() -> None:
    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"{GOLDEN_PATH} not found. Run `uv run python -m eval.generate_golden` first."
        )

    args = parse_args()
    golden = load_golden()
    if args.n_samples:
        random.seed(args.seed)
        golden = random.sample(golden, min(args.n_samples, len(golden)))
    print(f"Evaluating on {len(golden)} golden questions (method={METHOD}, top_k={TOP_K})")

    retriever = Retriever()

    results = {}
    for name, use_rewrite, use_rerank_flag in VARIANTS:
        results[name] = evaluate(retriever, golden, use_rewrite, use_rerank_flag)

    print(f"\n{'variant':<18} {'hit_rate':>10} {'mrr':>10}")
    for name, metrics in results.items():
        print(f"{name:<18} {metrics['hit_rate']:>10.3f} {metrics['mrr']:>10.3f}")

    best = max(results, key=lambda k: results[k]["mrr"])
    print(f"\nBest variant by MRR: {best}")

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(
            {"method": METHOD, "top_k": TOP_K, "n_questions": len(golden), "results": results},
            indent=2,
        )
    )
    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
