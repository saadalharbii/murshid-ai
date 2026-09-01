"""MurshidAI - bilingual RAG assistant for Saudi scholarship students."""

from __future__ import annotations

import streamlit as st

from murshid import config
from murshid.rag import RAGPipeline, detect_language

EXAMPLES = [
    "ما هي أفضل المدن للدراسة في بريطانيا؟",
    "How do I open a UK bank account as a student?",
    "كيف أجد سكن للطلاب؟",
    "What is student life like in the UK?",
]

st.set_page_config(page_title="MurshidAI - مرشد", page_icon="🎓", layout="centered")

# `unicode-bidi: plaintext` lets the browser pick each paragraph's direction
# from its first strong character, so Arabic containing English (e.g. "QS
# Ranking") stays laid out correctly instead of the Latin run flipping it.
st.markdown(
    """
    <style>
      .rtl, .rtl * { direction: rtl; text-align: right; }
      .ltr, .ltr * { direction: ltr; text-align: left; }
      .rtl p, .rtl li, .ltr p, .ltr li { unicode-bidi: plaintext; }
      .source-card {
        background: rgba(128,128,128,0.10);
        border-left: 3px solid #4a8cff;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        margin: 0.4rem 0;
        font-size: 0.87rem;
        unicode-bidi: plaintext;
      }
      .source-meta { opacity: 0.65; font-size: 0.78rem; margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline() -> RAGPipeline:
    """Built once per server process and reused across requests."""
    return RAGPipeline()


def directional(text: str, language: str) -> str:
    """Wrap text so it renders in the reading direction of `language`."""
    return f'<div class="{"rtl" if language == "arabic" else "ltr"}">\n\n{text}\n\n</div>'


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
        st.error("Missing API keys. Set ANTHROPIC_API_KEY and VOYAGE_API_KEY.")
        st.stop()

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
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
            st.markdown(directional(message["content"], message["language"]), unsafe_allow_html=True)
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask in Arabic or English...") or st.session_state.pop("pending", None)

    if not question:
        return

    # Alignment follows the question's language for the whole exchange, so a
    # reply peppered with English still reads right-to-left for Arabic askers.
    language = detect_language(question)

    st.session_state.messages.append(
        {"role": "user", "content": question, "language": language}
    )
    with st.chat_message("user"):
        st.markdown(directional(question, language), unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Searching the archive..."):
            _, sources, error = pipeline.retrieve(question)

        if error:
            st.error(error)
            return

        if not sources:
            text = (
                "لم أجد في أرشيف المجموعات ما يجيب على سؤالك. جرّب صياغة أخرى."
                if language == "arabic"
                else "I couldn't find anything in the archive that answers that. "
                "Try rephrasing your question."
            )
            st.markdown(directional(text, language), unsafe_allow_html=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "sources": [], "language": language}
            )
            return

        # Stream the answer so text appears as it is generated rather than
        # after the full ~10s round trip.
        placeholder = st.empty()
        parts: list[str] = []
        try:
            for chunk in pipeline.stream_answer(question, language, sources):
                parts.append(chunk)
                placeholder.markdown(
                    directional("".join(parts) + " ▌", language), unsafe_allow_html=True
                )
        except Exception as exc:
            placeholder.error(str(exc))
            return

        text = "".join(parts).strip()
        placeholder.markdown(directional(text, language), unsafe_allow_html=True)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources, "language": language}
    )


if __name__ == "__main__":
    main()
