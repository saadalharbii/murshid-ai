"""Strip personal contact details from message text before it is indexed.

The archive is a public group, but that does not make every line in it safe to
republish. Members posted phone numbers, Telegram handles and email addresses
in the course of ordinary conversation, and an indexed chunk is shown verbatim
in the app's sources panel - which no system prompt can police, because the
panel renders retrieved text directly rather than passing it through the model.

So the scrub happens at parse time, before anything is embedded or committed.
Redaction is deliberately blunt: a placeholder keeps the sentence readable and
retrievable ("call the embassy on [phone]" still matches a query about the
embassy) while removing the detail itself. Over-redacting a published embassy
switchboard costs little; under-redacting someone's mobile is not recoverable
once the index is committed and pushed.
"""

from __future__ import annotations

import re

_PHONE = "[phone]"
_HANDLE = "[handle]"
_EMAIL = "[email]"
_LINK = "[link]"

# Ordered: the first pattern to match a span wins, so the specific forms run
# before the general digit run that would otherwise swallow them.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Emails first - an address contains an @handle that would otherwise be
    # redacted separately, leaving a broken "[handle].com" behind.
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b"), _EMAIL),
    # Telegram invite and channel links, with or without a scheme.
    (re.compile(r"(?:https?://)?t\.me/[A-Za-z0-9_+/]+"), _LINK),
    # International numbers: +966..., 00966..., with spaces, dashes or
    # parentheses as separators.
    (re.compile(r"(?:\+|00)\d[\d\s\-()]{7,17}\d"), _PHONE),
    # National mobile formats seen in this corpus: Saudi 05xxxxxxxx and
    # UK 07xxxxxxxxx.
    (re.compile(r"\b0[57]\d{8,9}\b"), _PHONE),
    # Any remaining bare run of 9-15 digits. Long enough not to catch years,
    # prices, IELTS scores or dates; short enough to catch a local number
    # typed without a country code.
    (re.compile(r"\b\d{9,15}\b"), _PHONE),
    # Telegram handles. Four or more characters after the @ to avoid eating
    # ordinary uses of "@" in English text.
    (re.compile(r"@[A-Za-z][A-Za-z0-9_]{3,31}"), _HANDLE),
)


def scrub(text: str) -> str:
    """Replace contact details in `text` with neutral placeholders.

    Applied to every message at parse time, so the index, the sources panel
    and anything else downstream only ever see redacted text.
    """
    if not text:
        return text

    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def contains_contact_details(text: str) -> bool:
    """True if `text` still holds something the scrubber should have removed.

    Used by the tests and by ingest as a last check before an index is
    written, on the principle that a redaction bug should fail the build
    rather than ship quietly.
    """
    return any(pattern.search(text) for pattern, _ in _PATTERNS)
