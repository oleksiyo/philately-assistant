"""LLM evaluation: compare prompt variants ("strict" vs "open") via LLM-as-judge.

For each sampled golden question, retrieval is run once (vector, matching the
production pipeline), then both prompt variants answer from the *same*
retrieved context so the comparison isolates the prompt, not retrieval noise.
A judge model scores each answer 1-5 on faithfulness + relevance.

Usage:
    uv run python -m eval.llm_eval
    uv run python -m eval.llm_eval --n-samples 20   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.llm import DEFAULT_MODEL, answer as llm_answer
from rag.prompt import build_messages, format_context
from rag.retrieval import Retriever

GOLDEN_PATH = ROOT_DIR / "data" / "golden_qa.jsonl"
OUT_PATH = ROOT_DIR / "data" / "llm_eval_results.json"
VARIANTS = ["strict", "open"]
JUDGE_MODEL = "gpt-4o-mini"
RETRIEVAL_METHOD = "vector"
TOP_K = 5

JUDGE_PROMPT = """\
You are grading an answer produced by a RAG assistant for stamp collectors (philately).

QUESTION:
{question}

CONTEXT the assistant had access to:
{context}

ASSISTANT'S ANSWER:
{answer}

Score the answer from 1 to 5 on:
- Faithfulness: does it stick to the context, without inventing facts not present there?
- Relevance: does it actually answer the question?

Respond with ONLY a single integer from 1 to 5 (overall quality). A 1 means unfaithful or \
irrelevant; a 5 means fully faithful and directly answers the question. If the assistant \
correctly says it doesn't have enough information and the context genuinely lacks the \
answer, that also counts as a 5.
"""


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge(client: OpenAI, question: str, context: str, answer: str) -> int | None:
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    match = re.search(r"[1-5]", content)
    return int(match.group()) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prompt variants via LLM-as-judge.")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_PATH))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"{GOLDEN_PATH} not found. Run `uv run python -m eval.generate_golden` first."
        )

    args = parse_args()
    golden = load_golden()
    random.seed(args.seed)
    sample = random.sample(golden, min(args.n_samples, len(golden)))
    print(f"Evaluating {len(sample)} questions x {len(VARIANTS)} prompt variants")

    retriever = Retriever()
    client = OpenAI()

    scores: dict[str, list[int]] = {variant: [] for variant in VARIANTS}
    records = []

    for item in tqdm(sample, desc="Evaluating prompt variants"):
        question = item["question"]
        chunks = retriever.search(question, method=RETRIEVAL_METHOD, num_results=TOP_K)
        context = format_context(chunks)

        row = {"question": question}
        for variant in VARIANTS:
            messages = build_messages(question, chunks, variant=variant)
            answer_text = llm_answer(messages, model=DEFAULT_MODEL)
            score = judge(client, question, context, answer_text)
            if score is not None:
                scores[variant].append(score)
            row[variant] = {"answer": answer_text, "score": score}
        records.append(row)

    summary = {
        variant: {
            "avg_score": sum(s) / len(s) if s else 0.0,
            "n": len(s),
        }
        for variant, s in scores.items()
    }

    print(f"\n{'variant':<10} {'avg_score':>10} {'n':>6}")
    for variant, stats in summary.items():
        print(f"{variant:<10} {stats['avg_score']:>10.2f} {stats['n']:>6}")

    best = max(summary, key=lambda v: summary[v]["avg_score"])
    print(f"\nBest prompt variant: {best}")

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False)
    )
    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
