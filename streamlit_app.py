"""MurshidAI - bilingual RAG assistant for Saudi scholarship students."""

from __future__ import annotations

import html
import sys
import threading
import traceback
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent
_PACKAGE = "murshid"


def _stamp(paths) -> tuple:
    """Identify a set of files by name, size and modification time.

    A missing file is left out rather than raised, so the loader that needs
    it reports the problem in its own words.
    """
    return tuple(
        (path.name, stat.st_size, stat.st_mtime_ns)
        for path in sorted(paths)
        if path.exists()
        for stat in [path.stat()]
    )


@st.cache_resource
def _import_lock() -> threading.Lock:
    """One lock per server process. Each visitor's session runs this script on
    its own thread, and two of them clearing modules at once could interleave
    with each other's imports."""
    return threading.Lock()


def _forget_stale_code() -> tuple:
    """Drop the package from memory if its files changed after it was loaded.

    Streamlit Cloud deploys by pulling new files into the process that is
    already running. This script is re-read on every run, but the package it
    imports stays cached in sys.modules, so new app code met old package code
    and every visitor got an ImportError until the app was rebooted by hand.
    Comparing the files on disk to the ones that were loaded, and re-importing
    on a mismatch, makes every deploy take effect on the next page load.
    """
    stamp = _stamp((_ROOT / _PACKAGE).glob("*.py"))
    loaded = sys.modules.get(_PACKAGE)
    if loaded is not None and getattr(loaded, "_source_stamp", None) != stamp:
        for name in [n for n in sys.modules if n == _PACKAGE or n.startswith(_PACKAGE + ".")]:
            del sys.modules[name]
    return stamp


with _import_lock():
    _CODE_STAMP = _forget_stale_code()

    import murshid
    from murshid import config
    from murshid.messages import (
        TROUBLE_AR,
        TROUBLE_EN,
        no_results,
        searched_for,
        searching,
        source_authors,
        source_date,
        trouble,
        writing,
    )
    from murshid.rag import RAGPipeline, detect_language, past_exchanges, standalone_question

    murshid._source_stamp = _CODE_STAMP

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
      .searched { opacity: 0.65; font-size: 0.8rem; unicode-bidi: plaintext; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading knowledge base...", max_entries=1)
def load_pipeline(code: tuple, index: tuple) -> RAGPipeline:
    """Built once per server process and reused across requests.

    The arguments are only the cache key. A deploy that changes the package
    or the index rebuilds the pipeline instead of serving one built from
    older code or data, and max_entries drops the old one from memory.
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


def render_searched(query: str | None, language: str) -> None:
    """Show what a follow-up was rewritten to. Nothing for a first question."""
    if query:
        # Escaped: the rewrite echoes the reader's own words back into HTML.
        text = html.escape(searched_for(query, language))
        st.markdown(
            f'<div class="searched {"rtl" if language == "arabic" else "ltr"}">{text}</div>',
            unsafe_allow_html=True,
        )


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
        pipeline = load_pipeline(_CODE_STAMP, _stamp([config.INDEX_PATH]))
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

        span = pipeline.store.year_range()
        if span:
            low, high = span
            st.caption(
                f"Covering {low}" if low == high else f"Covering {low}–{high}"
            )
        st.caption(f"Claude: `{config.CLAUDE_MODEL}`\n\nEmbeddings: `{config.VOYAGE_MODEL}`")
        st.divider()
        st.caption(
            "⚠️ Community discussions, not official guidance. Rules on visas, "
            "banking and the NHS change over time, and passages carry their "
            "date - verify anything important with your scholarship office."
        )
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    st.session_state.setdefault("messages", [])

    if not st.session_state.messages:
        st.write("**Try asking:**")
        columns = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if columns[i % 2].button(example, key=f"ex{i}", width="stretch"):
                st.session_state.pending = example
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(directional(message["content"], message["language"]), unsafe_allow_html=True)
            if message["role"] == "assistant":
                render_searched(message.get("searched"), message["language"])
                render_sources(message.get("sources", []), message["language"])

    question = st.chat_input("Ask in Arabic or English...") or st.session_state.pop("pending", None)

    if not question:
        return

    # Alignment follows the question's language for the whole exchange, so a
    # reply peppered with English still reads right-to-left for Arabic askers.
    language = detect_language(question)
    history = past_exchanges(st.session_state.messages)

    st.session_state.messages.append(
        {"role": "user", "content": question, "language": language}
    )
    with st.chat_message("user"):
        st.markdown(directional(question, language), unsafe_allow_html=True)

    with st.chat_message("assistant"):
        try:
            answer(pipeline, question, language, history)
        except Exception:
            # Every expected failure already has a fallback or a message of
            # its own. This is for the unexpected: the reader still gets a
            # line in their language, and the traceback goes to the logs.
            traceback.print_exc()
            st.error(trouble(language))


def answer(pipeline: RAGPipeline, question: str, language: str, history) -> None:
    """Search, write and show the reply to one question."""
    with st.spinner(searching(language)):
        # A follow-up like "what about Manchester?" is rewritten into a
        # full question first, and that is what is searched and answered.
        query = standalone_question(question, history)
        _, sources, error = pipeline.retrieve(query)
    searched = query if query.casefold() != question.casefold() else None

    if error:
        # The underlying message is for the logs; the reader gets a
        # generic line in their own language.
        print(f"retrieval failed: {error}", file=sys.stderr)
        st.error(trouble(language))
        return

    if not sources:
        text = no_results(language)
        st.markdown(directional(text, language), unsafe_allow_html=True)
        render_searched(searched, language)
        st.session_state.messages.append(
            {"role": "assistant", "content": text, "sources": [],
             "searched": searched, "language": language}
        )
        return

    # Stream the answer so text appears as it is generated rather than
    # after the full ~10s round trip.
    placeholder = st.empty()
    parts: list[str] = []
    try:
        stream = pipeline.stream_answer(query, language, sources)

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
        print(f"generation failed: {exc!r}", file=sys.stderr)
        placeholder.error(trouble(language))
        return

    text = "".join(parts).strip()
    if not text:
        # A stream that ends without a word is a failure too, and an empty
        # reply bubble looks more broken than saying so.
        print("generation failed: empty reply", file=sys.stderr)
        placeholder.error(trouble(language))
        return
    placeholder.markdown(directional(text, language), unsafe_allow_html=True)
    render_searched(searched, language)
    render_sources(sources, language)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources,
         "searched": searched, "language": language}
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # The last resort, for a failure outside any single question - the
        # page itself failing to draw. A visitor sees an apology in both
        # languages instead of a traceback, and the chat is cleared so that a
        # conversation that cannot be redrawn does not fail on every rerun.
        traceback.print_exc()
        st.session_state.messages = []
        st.error(f"{TROUBLE_EN}\n\n{TROUBLE_AR}")
