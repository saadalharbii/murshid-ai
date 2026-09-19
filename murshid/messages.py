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


SEARCHING_AR = "أبحث في الأرشيف..."
SEARCHING_EN = "Searching the archive..."

WRITING_AR = "أصيغ الإجابة..."
WRITING_EN = "Writing the answer..."


def searching(language: str) -> str:
    """Shown while retrieval runs."""
    return SEARCHING_AR if language == "arabic" else SEARCHING_EN


def writing(language: str) -> str:
    """Shown after retrieval, while waiting on the first token.

    Retrieval and generation are separate waits of a few seconds each. One
    unchanging spinner across both reads as a stall; naming the current stage
    shows the request is progressing.
    """
    return WRITING_AR if language == "arabic" else WRITING_EN


def no_results(language: str) -> str:
    """Shown when retrieval succeeds but finds nothing relevant."""
    return NO_RESULTS_AR if language == "arabic" else NO_RESULTS_EN


_MONTHS_AR = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
    7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر",
}
_MONTHS_EN = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def source_date(date_range: str, language: str) -> str:
    """Render a chunk's date range as a month and year.

    The archive is old enough that its age is the most decision-relevant fact
    about any passage - banking and visa rules have moved on since. A full
    timestamp ("03.08.2019 10:42:52 UTC+00:00") buries that in precision no
    reader needs; the month and year make it obvious at a glance.
    """
    stamp = (date_range or "").split(" to ")[0].strip()
    parts = stamp.split(".")
    if len(parts) < 3:
        return ""
    try:
        month = int(parts[1])
        year = int(parts[2].split()[0])
    except (ValueError, IndexError):
        return ""
    months = _MONTHS_AR if language == "arabic" else _MONTHS_EN
    name = months.get(month)
    return f"{name} {year}" if name else str(year)


def source_authors(authors: str, language: str) -> str:
    """Describe who is speaking, without pretending a deleted account is a name.

    Two thirds of chunks are attributed to "Deleted Account" - Telegram's
    placeholder for users who since left. Printing that as an author is noise,
    so it becomes a count of participants instead.
    """
    names = [a.strip() for a in (authors or "").split(",") if a.strip()]
    real = [a for a in names if a != "Deleted Account"]

    if not names:
        return ""
    if not real:
        if language == "arabic":
            return "طالب" if len(names) == 1 else f"{len(names)} طلاب"
        return "a student" if len(names) == 1 else f"{len(names)} students"

    shown = ", ".join(real[:2])
    extra = len(names) - min(len(real), 2)
    if extra <= 0:
        return shown
    if language == "arabic":
        return f"{shown} +{extra}"
    return f"{shown} +{extra}"
