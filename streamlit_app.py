"""MurshidAI - bilingual RAG assistant for Saudi scholarship students.

Single-process Streamlit app: retrieval and generation happen in-process, so
there is no separate API server or vector database to keep running.
"""

from __future__ import annotations

import streamlit as st

from murshid import config
from murshid.rag import RAGPipeline

EXAMPLES = [
    "ما هي أفضل المدن للدراسة في بريطانيا؟",
    "How do I open a UK bank account as a student?",
    "كيف أجد سكن للطلاب؟",
    "What is student life like in the UK?",
]

st.set_page_config(page_title="MurshidAI - مرشد", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
      .stChatMessage { border-radius: 10px; }
      .source-card {
        background: rgba(128,128,128,0.10);
        border-left: 3px solid #4a8cff;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        margin: 0.4rem 0;
        font-size: 0.87rem;
      }
      .source-meta { opacity: 0.65; font-size: 0.78rem; margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline() -> RAGPipeline:
    """Build the pipeline once per server process."""
    return RAGPipeline()


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, source in enumerate(sources, 1):
            excerpt = source.content[:400] + ("..." if len(source.content) > 400 else "")
            st.markdown(
                f'<div class="source-card">{excerpt}'
                f'<div class="source-meta">#{i} · similarity {source.score:.0%} · '
                f'{source.metadata.get("authors", "unknown")}</div></div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    st.title("🎓 MurshidAI · مرشد")
    st.caption(
        "Ask about studying in the UK, in Arabic or English. "
        "Answers come from real Saudi student Telegram discussions."
    )

    if not config.ANTHROPIC_API_KEY or not config.VOYAGE_API_KEY:
        st.error(
            "Missing API keys. Set ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env "
            "(or in Streamlit secrets when deployed)."
        )
        st.stop()

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        st.stop()

    with st.sidebar:
        st.subheader("About")
        st.write(
            "A retrieval-augmented chatbot over an archive of Saudi scholarship "
            "student discussions. Questions are embedded, matched against the "
            "archive, and answered by Claude using only what was retrieved."
        )
        st.metric("Indexed passages", f"{len(pipeline.store):,}")
        st.caption(f"Claude: `{config.CLAUDE_MODEL}`\n\nEmbeddings: `{config.VOYAGE_MODEL}`")
        st.divider()
        st.caption(
            "⚠️ Community discussions, not official guidance. "
            "Verify anything important with your scholarship office."
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.session_state.setdefault("messages", [])

    if not st.session_state.messages:
        st.write("**Try asking:**")
        columns = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if columns[i % 2].button(example, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask in Arabic or English...") or st.session_state.pop("pending", None)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the archive..."):
                try:
                    answer = pipeline.answer(question)
                except Exception as exc:  # surfaced to the user rather than a blank page
                    st.error(f"Something went wrong: {exc}")
                    st.stop()

            st.markdown(answer.text)
            render_sources(answer.sources)
            st.caption(f"{answer.elapsed:.1f}s · {answer.language}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer.text, "sources": answer.sources}
        )


if __name__ == "__main__":
    main()
