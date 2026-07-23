# Philately Assistant

A RAG chatbot / reference assistant for stamp collectors. Capstone project for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

> Work in progress. This README will be expanded with an app description, run instructions, evaluation results, and screenshots — see the requirements in `docs/project.md`.

## Documentation

- [`docs/project-plan.md`](docs/project-plan.md) — plan: data, architecture, evaluation, monitoring, grading criteria.
- [`CLAUDE.md`](CLAUDE.md) — context and workflow for the agent/developer.

## Repository structure

```
ingestion/   - scripts for collecting and chunking data (Wikipedia)
rag/         - retrieval + prompt building + LLM calls
app/         - Streamlit UI
eval/        - retrieval/LLM evaluation
monitoring/  - feedback logging, Grafana dashboard
data/        - collected/processed data
docker-compose.yml
```

## Quickstart

_(to be filled in once the code exists)_

1. `cp .env.example .env` and fill in the keys
2. `uv sync`
3. `uv run ingestion/ingest.py`
4. `uv run streamlit run app/streamlit_app.py`
