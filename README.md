# Philately Assistant

A RAG (Retrieval-Augmented Generation) chatbot / reference assistant for stamp collectors (philatelists). Capstone project for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Problem description

Stamp collecting has a lot of specialized vocabulary (perforation, watermarks, tête-bêche, plating, cancellation...), plus decades of country-specific stamp-issue history and catalog systems (Scott, Yvert, Michel...). That information is scattered across dozens of sites, forums and glossaries and is hard to search.

Philately Assistant lets you ask a natural-language question — *"what is a tête-bêche pair"*, *"what stamps did France issue in the 1900s"* — and get an answer grounded in a corpus of Wikipedia philately articles, with links to the source articles used. If the corpus doesn't actually contain the answer, the assistant says so instead of guessing.

## Dataset

Ingested automatically from the **Wikipedia API** (`ingestion/ingest.py`), starting from three categories and recursing two levels into subcategories:

- `Category:Philately`
- `Category:Postage stamps by country`
- `Category:Compendium of postage stamp issuers`

Current corpus: **400 articles → 3,941 chunks** (chunked by section, then by paragraph groups up to ~1,200 characters), stored as `data/chunks.jsonl` with `{chunk_id, title, section, url, text}`. Content is CC BY-SA (Wikipedia) — each answer links back to the source article(s) it was drawn from.

**Known coverage gap**: the crawl is capped at 400 articles for corpus-size reasons, so some specific terminology (e.g. "tête-bêche" as its own article) isn't present — the assistant correctly declines to answer those rather than hallucinating (see [LLM evaluation](#llm-evaluation)).

## Architecture

```
Wikipedia API
      │
      ▼
ingestion/ingest.py  (category crawl → clean → chunk)
      │
      ▼
data/chunks.jsonl
      │
      ├──────────────┐
      ▼              ▼
keyword index    vector index
(minsearch.Index) (minsearch.VectorSearch +
                   sentence-transformers,
                   local, no API key needed)
      │              │
      └──────┬───────┘
             ▼
   rag/query_rewrite.py — LLM query rewrite (optional, off by default)
             │
             ▼
   rag/retrieval.py — keyword / vector / hybrid (RRF)
             │
             ▼
   rag/rerank.py — cross-encoder re-rank of top-20 → top-5 (on by default)
             │
             ▼
   rag/prompt.py — "strict" vs "open" system prompt
             │
             ▼
   rag/llm.py — OpenAI chat completion (gpt-4o-mini)
             │
             ▼
   app/streamlit_app.py ──► answer + sources
             │
             ▼
   monitoring/db.py → Postgres (conversations + 👍/👎)
             │
             ▼
   Grafana dashboard (7 charts)
```

## Project Structure

```
philately_assistant/
├── ingestion/
│   ├── wikipedia_source.py    # Category crawl (Wikipedia API)
│   ├── chunking.py            # Clean + chunk articles
│   └── ingest.py              # dlt pipeline: crawl -> chunk -> DuckDB -> chunks.jsonl
├── rag/
│   ├── embeddings.py          # sentence-transformers embeddings (local, no API key)
│   ├── index.py               # keyword + vector index builders
│   ├── retrieval.py           # keyword / vector / hybrid (RRF) search
│   ├── query_rewrite.py       # LLM query rewriting (optional, off by default)
│   ├── rerank.py              # cross-encoder re-ranking (on by default)
│   ├── prompt.py              # "strict" vs "open" system prompts
│   ├── llm.py                 # OpenAI chat completion wrapper
│   └── pipeline.py            # end-to-end RagPipeline
├── app/
│   └── streamlit_app.py       # Chat UI
├── eval/
│   ├── generate_golden.py     # LLM-generated golden Q&A set
│   ├── retrieval_eval.py      # keyword vs vector vs hybrid comparison
│   ├── rerank_rewrite_eval.py # re-rank / query-rewrite ablation
│   ├── llm_eval.py            # prompt variant comparison (LLM-as-judge)
│   └── seed_monitoring.py     # sample traffic generator for the dashboard
├── monitoring/
│   ├── db.py                  # Postgres logging (conversations + feedback)
│   └── grafana/               # datasource + dashboard provisioning
├── data/                      # generated corpus, embeddings cache, eval results (gitignored except results)
├── deploy/                    # AWS EC2 deploy/teardown scripts (untested, see Cloud Deployment)
├── docs/                      # course requirements + project plan
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml, uv.lock
├── .env.example
└── README.md
```

The project goes from a raw Wikipedia crawl to a containerized, monitored RAG application — nothing here is a notebook-only prototype; every stage is a runnable script or service.

## Retrieval evaluation

Methodology (`eval/generate_golden.py`, `eval/retrieval_eval.py`):

1. Sampled 200 chunks (prose-only, filtering out list/index pages) and asked an LLM to generate one natural-language question per chunk → `data/golden_qa.jsonl`.
2. Compared **3 retrieval approaches** — keyword-only (`minsearch.Index`), vector-only (`minsearch.VectorSearch` + `all-MiniLM-L6-v2` embeddings), and hybrid (Reciprocal Rank Fusion of the two) — at `top_k=5`, matching what the production pipeline actually retrieves.
3. Metrics: hit-rate and MRR.

| method | hit-rate | MRR |
|---|---|---|
| keyword | 0.740 | 0.540 |
| **vector** | **0.905** | **0.727** |
| hybrid (RRF) | 0.865 | 0.700 |

**Vector search won on both metrics** and is used as the default in `rag/pipeline.py`. This was a genuine (if slightly counterintuitive) finding: RRF-hybrid diluted vector's stronger ranking with keyword's weaker one rather than improving on it here. Hybrid is still fully implemented and selectable in the app UI/`Retriever.search(method=...)` — the *best-search-method bonus* only requires evaluating hybrid, not that it wins.

Full results: `data/retrieval_eval_results.json`. Reproduce with:
```bash
uv run python -m eval.generate_golden
uv run python -m eval.retrieval_eval --top-k 5
```

### Re-ranking & query rewriting (best-practice bonuses)

On top of the winning `vector` method, `eval/rerank_rewrite_eval.py` ablates two more techniques on the same 200-question golden set:

- **Query rewriting** (`rag/query_rewrite.py`): an LLM reformulates the question into a cleaner standalone search query before retrieval.
- **Re-ranking** (`rag/rerank.py`): retrieve a larger candidate pool (20) with `vector`, then a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores and cuts down to `top_k=5`.

| variant | hit-rate | MRR |
|---|---|---|
| baseline (vector) | 0.905 | 0.727 |
| +rewrite | 0.880 | 0.691 |
| **+rerank** | **0.960** | **0.922** |
| +rewrite+rerank | 0.955 | 0.917 |

**Re-ranking is a clear, substantial win** and is on by default in `rag/pipeline.py`. **Query rewriting alone slightly *hurt*** MRR here — the golden questions were LLM-generated directly from their answer chunk, so they already closely match its wording, and rewriting drifts the query away from that precise phrasing. It's implemented and toggleable in the app/pipeline (satisfying the practice), but kept **off by default** based on this measured result; it may still help with genuinely messy conversational phrasing that this benchmark doesn't simulate.

Full results: `data/rerank_rewrite_eval_results.json`. Reproduce with:
```bash
uv run python -m eval.rerank_rewrite_eval
```

## LLM evaluation

Methodology (`eval/llm_eval.py`): for 40 sampled golden questions, retrieval is run once (vector, `top_k=5`), then **two prompt variants** answer from the same context:
- `strict` — answer only from context, say so if the context doesn't have the answer
- `open` — use context as primary source but may supplement with general knowledge

An LLM-as-judge (`gpt-4o-mini`) scores each answer 1–5 on faithfulness + relevance.

| variant | avg score | n |
|---|---|---|
| strict | 5.00 | 40 |
| open | 5.00 | 40 |

Both variants tie on this metric — expected, since golden questions are generated *from* their answer chunk, so both variants always have the answer available and neither needs to guess. The real difference shows up out-of-corpus: asking about "tête-bêche" (a term not covered by the current 400-article crawl), `strict` correctly answers *"I don't have enough information"* instead of inventing an answer from general knowledge. `strict` is kept as the deployed default for that reason.

Full results: `data/llm_eval_results.json`. Reproduce with:
```bash
uv run python -m eval.llm_eval
```

## Interface

Streamlit chat app (`app/streamlit_app.py`): message history, source links per answer, 👍/👎 feedback buttons, and a sidebar to switch retrieval method / prompt variant / query rewriting / re-ranking live (useful for demoing the evaluations above). The rewritten search query is shown under the answer when it differs from what you typed.

## Ingestion pipeline

Fully automated, one command, no manual steps, orchestrated with **[dlt](https://dlthub.com/)**:
```bash
uv run python ingestion/ingest.py
```
Crawls the configured Wikipedia categories (`ingestion/wikipedia_source.py`), cleans wiki markup/boilerplate sections and chunks by section/paragraph (`ingestion/chunking.py`), then a dlt resource (`ingestion/ingest.py::wikipedia_chunks`) loads the chunks into a local DuckDB dataset (`data/philately.duckdb`, schema-managed, `write_disposition="replace"`), which is exported to `data/chunks.jsonl` — the flat format the rest of the app reads.

## Monitoring

Every conversation (question, answer, retrieval method, prompt variant, model, latency, sources, 👍/👎) is logged to Postgres (`monitoring/db.py`). Grafana auto-provisions a dashboard (`monitoring/grafana/`) with **7 charts**:

1. Requests over time
2. Average latency
3. Positive feedback %
4. "No info" answer rate % (proxy for "retrieval found nothing relevant")
5. Retrieval method breakdown
6. Prompt variant breakdown
7. Recent questions table

No manual dashboard setup needed — datasource + dashboard are provisioned automatically on `docker compose up`. To see it populated with sample traffic:
```bash
uv run python -m eval.seed_monitoring
```

## Containerization

`docker compose up` starts everything: `app` (Streamlit, built from the repo `Dockerfile`), `postgres`, `grafana`. No services need to be started separately.

## How to Run Locally and via Docker

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, an OpenAI API key ([create one here](https://platform.openai.com/api-keys) — pay-as-you-go, needs a payment method on the account; this whole project costs a few cents to run).

**Local Setup:**
1. Clone the repository: `git clone <this-repo> && cd philately_assistant`
2. Copy the env template and fill in your key: `cp .env.example .env` (set `OPENAI_API_KEY`; the Postgres/Grafana defaults in there already match `docker-compose.yml`, no need to change them)
3. Install dependencies: `uv sync`
4. Build the corpus (takes ~4 min, ~400 Wikipedia API calls, no key needed for this step): `uv run python ingestion/ingest.py`
5. (optional) Reproduce the evaluation:
   ```bash
   uv run python -m eval.generate_golden
   uv run python -m eval.retrieval_eval --top-k 5
   uv run python -m eval.llm_eval
   uv run python -m eval.rerank_rewrite_eval
   ```
6. Start just Postgres (the app needs it even outside Docker): `docker compose up -d postgres`
7. Run the app directly: `uv run streamlit run app/streamlit_app.py`

**Docker Execution:**
1. Build the corpus first — either locally (`uv run python ingestion/ingest.py`, needs `uv` + Python) or fully through Docker with no local Python at all:
   ```bash
   docker compose --profile ingest run --rm ingest
   ```
   Either way this populates `data/chunks.jsonl`, which the `app` container mounts rather than regenerating itself.
2. Build and start everything: `docker compose up -d --build`
3. Check the app is healthy: `curl http://localhost:8501/_stcore/health`

Either way, once running:
- App: http://localhost:8501
- Grafana: http://localhost:3001 (admin/admin by default — see `.env.example`)
- Postgres: localhost:5432

All dependency versions are pinned in `uv.lock`. `data/chunks.jsonl` and `data/embeddings.npy` are generated by ingestion and gitignored — they are not shipped in the repo, only reproduced by running the pipeline above.

## Cloud Deployment

Scripts (`deploy/`) provision a single AWS EC2 instance and run the exact same `docker-compose.yml` stack on it — no separate cloud-specific config, since the app is already fully containerized.

> **Disclosure:** these scripts were written and reviewed against the AWS CLI docs (every subcommand/flag checked against `--generate-cli-skeleton`) but **not run against a live AWS account** — treat as a documented, ready-to-use deployment path rather than a verified one.

**Prerequisites:** AWS CLI configured (`aws configure`), an existing EC2 key pair (`.pem` file in the repo root), `.env` filled in.

```bash
./deploy/deploy.sh my-key-pair-name        # optional 2nd arg: instance type, default t3.small
```

What it does:
1. Looks up the latest Ubuntu 24.04 AMI and launches an EC2 instance running `deploy/user-data.sh` (installs Docker + the Compose plugin on first boot)
2. Creates a security group opening ports 22 (SSH), 8501 (app), 3001 (Grafana) — tighten `--cidr` in `deploy/deploy.sh` for anything beyond a demo
3. Waits for SSH + cloud-init, then `rsync`s the repo over
4. Runs the corpus build once via the `ingest` Compose profile (no local Python/uv needed on the instance): `docker compose --profile ingest run --rm ingest`
5. Starts everything: `docker compose up -d --build`

Prints the public app/Grafana URLs when done. **Tear down when you're finished to stop billing:**
```bash
./deploy/teardown.sh
```

A `t3.small` runs roughly $0.02/hour — remember to tear down; nothing here auto-stops the instance.

## Next Steps

Known gaps and ideas for improvement, roughly in order of impact:

**Data & retrieval**
- Expand the corpus beyond 400 articles (raise `--max-articles`/`--max-depth` in `ingestion/ingest.py`) or add a second source (e.g. the Colnect API) — the current crawl misses some specific terminology (e.g. "tête-bêche" has no dedicated article), so the assistant correctly declines those questions rather than answering them.
- The query-rewriting eval set (`data/golden_qa.jsonl`) is LLM-generated *from* the answer chunks, so it's already well-phrased — it can't really test whether rewriting helps with genuinely messy, conversational user phrasing. A small hand-written "messy query" eval set would give a fairer read on rewriting's real value (it's currently off by default based on the existing benchmark).
- Minsearch's in-memory vector index is fine at ~4K chunks but won't scale past that without a rewrite — the course also covers PGVector (a production Postgres vector extension) as the next step up.

**Cloud & ops**
- `deploy/` has never been run against a live AWS account — worth doing once end-to-end, and adding an auto-shutdown (e.g. a cron `shutdown` on the instance after N idle hours) so a forgotten deployment doesn't rack up cost.
- `rag/embeddings.py` and `rag/rerank.py` load their models lazily on first request, adding ~15–40s of cold-start latency to the very first query after a container starts. Pre-warming them at app startup (before the first user request) would remove that.
- No CI (lint/tests on push) and no automated test suite — verification this session was all manual smoke tests against a live stack. A basic GitHub Actions workflow (ruff + a couple of pytest smoke tests against `rag/`) would catch regressions cheaply.
- Course module 03 teaches Kestra for ingestion orchestration specifically; we used dlt instead (allowed by the grading rubric, but a Kestra flow wrapping the same `ingestion/ingest.py` logic would align more closely with what the course itself demonstrates).

**Product**
- Conversations are stateless — each question is answered independently, with no follow-up/multi-turn context threaded into retrieval or the prompt.
- Screenshots of the running app and Grafana dashboard aren't in this README yet (needs a manual browser session to capture).

