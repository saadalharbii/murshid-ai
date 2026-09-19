"""MurshidAI - bilingual RAG assistant for Saudi scholarship students."""

from __future__ import annotations

import sys

import streamlit as st

from murshid import __version__, config
from murshid.messages import (
    no_results,
    searching,
    source_authors,
    source_date,
    trouble,
    writing,
)
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
def load_pipeline(version: str) -> RAGPipeline:
    """Built once per server process and reused across requests.

    `version` is part of the cache key: a redeploy that changes the package
    version discards the cached object instead of resurrecting an instance
    built from older code.
    """
    return RAGPipeline()


def directional(text: str, language: str) -> str:
    """Wrap text so it renders in the reading direction of `language`."""
    return f'<div class="{"rtl" if language == "arabic" else "ltr"}">\n\n{text}\n\n</div>'


_EXCERPT_CHARS = 400


def _excerpt(text: str) -> str:
    """Trim to a readable length on a word boundary.

    A hard character cut lands mid-word, and in a passage that answers the
    question the cut often removes the answer. Backing up to the last space
    costs a few characters and reads as a deliberate excerpt.
    """
    if len(text) <= _EXCERPT_CHARS:
        return text
    clipped = text[:_EXCERPT_CHARS]
    spaced = clipped.rsplit(" ", 1)[0]
    return (spaced if len(spaced) > _EXCERPT_CHARS * 0.7 else clipped) + "..."


def render_sources(sources, language: str) -> None:
    """Show the passages the answer was written from.

    The citations in the answer are only worth something if the reader can
    check them, so this panel is built around what helps them judge a passage:
    when it was said, and by how many people. The rerank score is deliberately
    not shown - it is an internal ranking number on a scale no reader knows,
    and dressing it up as a percentage implies a precision it does not have.
    """
    if not sources:
        return

    label = "المصادر" if language == "arabic" else "Sources"
    with st.expander(f"📚 {label} ({len(sources)})"):
        for i, source in enumerate(sources, 1):
            metadata = source.metadata
            date = source_date(metadata.get("date_range", ""), language)
            who = source_authors(metadata.get("authors", ""), language)
            meta = " · ".join(part for part in (f"#{i}", date, who) if part)
            st.markdown(
                f'<div class="source-card">{_excerpt(source.content)}'
                f'<div class="source-meta">{meta}</div></div>',
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
        pipeline = load_pipeline(__version__)
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
            "⚠️ Community discussions, not official guidance, and the archive "
            "is not current. Visa, banking and NHS rules change - verify "
            "anything important with your scholarship office."
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
                render_sources(message.get("sources", []), message["language"])

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
        with st.spinner(searching(language)):
            _, sources, error = pipeline.retrieve(question)

        if error:
            # The underlying message is for the logs; the reader gets a
            # generic line in their own language.
            print(f"retrieval failed: {error}", file=sys.stderr)
            st.error(trouble(language))
            return

        if not sources:
            text = no_results(language)
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
            stream = pipeline.stream_answer(question, language, sources)

            # Generation takes a couple of seconds to produce its first token.
            # Keep a spinner up for exactly that gap - an empty message box
            # there reads as a stall - then let the text itself show progress.
            with st.spinner(writing(language)):
                first = next(stream, "")

            if first:
                parts.append(first)
                placeholder.markdown(
                    directional(first + " ▌", language), unsafe_allow_html=True
                )

            for chunk in stream:
                parts.append(chunk)
                placeholder.markdown(
                    directional("".join(parts) + " ▌", language), unsafe_allow_html=True
                )
        except Exception as exc:
            print(f"generation failed: {exc}", file=sys.stderr)
            placeholder.error(trouble(language))
            return

        text = "".join(parts).strip()
        placeholder.markdown(directional(text, language), unsafe_allow_html=True)
        render_sources(sources, language)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources, "language": language}
    )


if __name__ == "__main__":
    main()
