"""Streamlit chat UI for the philately RAG assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from monitoring.db import init_db, log_conversation, log_feedback
from rag.pipeline import RagPipeline

st.set_page_config(page_title="Philately Assistant", page_icon="📮")


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline()


@st.cache_resource
def ensure_db() -> None:
    init_db()


def render_sources(sources: list[dict]) -> None:
    for s in sources:
        st.markdown(f"- [{s['title']} — {s['section']}]({s['url']})")


def render_feedback(conversation_id: int) -> None:
    key_prefix = f"feedback_{conversation_id}"
    if st.session_state.get(f"{key_prefix}_voted"):
        st.caption("Thanks for the feedback!")
        return

    col1, col2 = st.columns(2)
    if col1.button("👍", key=f"{key_prefix}_up"):
        log_feedback(conversation_id, 1)
        st.session_state[f"{key_prefix}_voted"] = True
        st.rerun()
    if col2.button("👎", key=f"{key_prefix}_down"):
        log_feedback(conversation_id, -1)
        st.session_state[f"{key_prefix}_voted"] = True
        st.rerun()


def main() -> None:
    st.title("📮 Philately Assistant")
    st.caption("A RAG reference assistant for stamp collectors, built on Wikipedia")

    ensure_db()
    pipeline = get_pipeline()

    with st.sidebar:
        st.header("Settings")
        retrieval_method = st.selectbox("Retrieval method", ["vector", "hybrid", "keyword"])
        prompt_variant = st.selectbox("Prompt variant", ["strict", "open"])

    if "history" not in st.session_state:
        st.session_state.history = []

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            with st.expander("Sources"):
                render_sources(item["sources"])
            render_feedback(item["conversation_id"])

    question = st.chat_input("Ask about stamps, philatelic terms, catalogs...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching for an answer..."):
                try:
                    result = pipeline.answer(
                        question,
                        retrieval_method=retrieval_method,
                        prompt_variant=prompt_variant,
                    )
                except Exception as exc:
                    st.error(f"Error calling the LLM: {exc}")
                    return

            st.write(result.answer)
            with st.expander("Sources"):
                render_sources(result.sources)

            conversation_id = log_conversation(
                question=result.question,
                answer=result.answer,
                retrieval_method=result.retrieval_method,
                prompt_variant=result.prompt_variant,
                model=result.model,
                latency_seconds=result.latency_seconds,
                sources=result.sources,
            )

            st.session_state.history.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "sources": result.sources,
                    "conversation_id": conversation_id,
                }
            )
            render_feedback(conversation_id)


if __name__ == "__main__":
    main()
