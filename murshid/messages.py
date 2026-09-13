"""User-facing copy, kept out of the Streamlit entrypoint.

Living here rather than in `streamlit_app.py` means tests can import these
without pulling in the Streamlit runtime, which costs minutes of import time.
"""

from __future__ import annotations

# The app answers in the question's language, so its failures should speak it
# too - an Arabic asker hitting an outage got an English error, which reads as
# a broken page rather than a temporary problem.
TROUBLE_AR = "تعذّر الوصول إلى الخدمة الآن. حاول مرة أخرى بعد قليل."
TROUBLE_EN = "The service is unavailable right now. Please try again shortly."

NO_RESULTS_AR = "لم أجد في أرشيف المجموعات ما يجيب على سؤالك. جرّب صياغة أخرى."
NO_RESULTS_EN = (
    "I couldn't find anything in the archive that answers that. "
    "Try rephrasing your question."
)


def trouble(language: str) -> str:
    """A generic, user-actionable failure message in the reader's language."""
    return TROUBLE_AR if language == "arabic" else TROUBLE_EN


def no_results(language: str) -> str:
    """Shown when retrieval succeeds but finds nothing relevant."""
    return NO_RESULTS_AR if language == "arabic" else NO_RESULTS_EN
