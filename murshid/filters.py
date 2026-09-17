"""Drop messages and chunks that carry no retrievable meaning.

Measured on the corpus, roughly one message in eight is conversational glue -
"thanks", "yes", a greeting, a lone emoji. None of it answers a question, but
all of it is embedded, stored and searched, and it dilutes the chunks it sits
inside.

The larger problem is at the chunk level: 19% of chunks are a question with no
answer in them. Those are worse than useless. A user's query is itself a
question, so a question-shaped chunk scores high on similarity, gets retrieved,
and gives the model nothing to answer from - which is exactly the behaviour
that made the reranker look like it was over-promoting noise.

Both filters are deliberately conservative. Losing a short chunk that did
contain an answer costs more than keeping a little filler, so the thresholds
are set where measurement showed the content was genuinely empty.
"""

from __future__ import annotations

import re
from typing import Any

# Acknowledgements and one-word replies. Matched on the whole message after
# stripping punctuation, so "تمام." and "تمام" both go.
_FILLER = frozenset(
    """
    لا ايه اي نعم ايوه ايوا اي وه صحيح صح تمام عادي يب اها اوك اوكي طيب
    شكرا شكراً مشكور مشكوره يسلمو تسلم تسلمي العفو تفضل تفضلي ok okay yes no
    thanks ty ايش وش هلا اهلا مرحبا
    """.split()
)

# Greetings and blessings. These open or close an exchange rather than
# contributing to it.
_RITUAL = re.compile(
    r"^(?:"
    r"السلام\s*عليكم|وعليكم\s*السلام|صباح\s*الخير|مساء\s*الخير|"
    r"حياك|حياكم|يعطيك\s*العافي|الله\s*يعافيك|جزاك\s*الله|الله\s*يجزا|"
    r"ما\s*شاء\s*الله|بالتوفيق|الله\s*يوفق|تحياتي"
    r")",
)

# Anything with no letters at all: punctuation, emoji, digits on their own.
_NO_LETTERS = re.compile(r"^[^\w؀-ۿ]*$|^[\W\d_]+$")

# A message must clear this to count as content on its own. Set from the
# measured distribution: the 10th percentile is 11 characters and is almost
# entirely acknowledgements.
_MIN_MESSAGE_CHARS = 12

# A greeting or blessing longer than this is carrying something else too.
_RITUAL_MAX_CHARS = 30

# A chunk holding a question and nothing longer than this has no answer in it.
_ANSWER_CHARS = 40

_QUESTION = re.compile(r"[؟?]")
_SPEAKER = re.compile(r"^[^:\n]{0,40}:\s*")
_PUNCT_STRIP = re.compile(r"[\s\.\,\!\?\؟\،\ـ\'\"‏‎]+")


def is_filler(text: str) -> bool:
    """True when a message is acknowledgement rather than content."""
    stripped = _PUNCT_STRIP.sub(" ", text).strip().lower()

    if not stripped:
        return True
    if _NO_LETTERS.match(text.strip()):
        return True
    if stripped in _FILLER:
        return True
    # A ritual opener or blessing is filler whenever it is the whole message.
    # The length bound is generous rather than tight - "السلام عليكم" is
    # exactly 12 characters, and gating on _MIN_MESSAGE_CHARS missed it - but
    # it still stops a greeting that leads into a real question from being
    # dropped along with the question.
    if _RITUAL.match(stripped) and len(stripped) <= _RITUAL_MAX_CHARS:
        return True

    # Short messages made only of filler words - "تمام شكرا", "ايه صح".
    words = stripped.split()
    if len(words) <= 3 and all(word in _FILLER for word in words):
        return True

    return len(stripped) < 3


def drop_filler(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove acknowledgement messages before chunking.

    Done before segmentation on purpose: filler sits between real exchanges and
    otherwise pads the character budget, pushing genuine content into separate
    chunks.
    """
    return [m for m in messages if not is_filler(m.get("content", ""))]


def body_lines(content: str) -> list[str]:
    """The message texts inside a rendered chunk, without speaker prefixes."""
    return [
        _SPEAKER.sub("", line).strip()
        for line in content.split("\n")
        if line.strip()
    ]


def is_question_only(content: str) -> bool:
    """True when a chunk asks something and never answers it.

    Retrieval matches a question against these strongly - a query is a question
    too - so they surface often and contribute nothing. A chunk counts as
    answered if any line after the first question is long enough to assert
    something.
    """
    lines = body_lines(content)
    if not lines:
        return True

    first_question = next(
        (i for i, line in enumerate(lines) if _QUESTION.search(line)), None
    )
    if first_question is None:
        return False

    return not any(len(line) > _ANSWER_CHARS for line in lines[first_question + 1 :])


def keep_chunk(chunk: dict[str, Any]) -> bool:
    """Whether a built chunk is worth embedding."""
    content = chunk.get("content", "")
    lines = body_lines(content)

    # A chunk of nothing but short lines carries no answer regardless of how
    # many messages it holds.
    if not any(len(line) > _ANSWER_CHARS for line in lines):
        return False

    return not is_question_only(content)


def drop_empty_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove chunks with no answer in them."""
    return [c for c in chunks if keep_chunk(c)]
