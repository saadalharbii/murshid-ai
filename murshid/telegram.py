"""Parser for Telegram Desktop HTML exports.

Message text is scrubbed of contact details as it is parsed - see `scrub.py`
for why that happens here rather than downstream.

Chunking is conversation-aware rather than a fixed character budget. A plain
500-character split produced chunks averaging 7.4 messages from 3.8 different
authors, so every chunk was a blend of unrelated topics and looked vaguely
similar to every query - the measured cause of compressed similarity scores.
Splitting on conversation boundaries instead roughly halves that mixing.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .scrub import scrub

_WHITESPACE = re.compile(r"\s+")
_REPLY_TARGET = re.compile(r"go_to_message(\d+)")

# A pause this long usually means the previous exchange finished.
#
# Tuned twice. On the original single-year sample 3 minutes was right: it gave
# ~2.2 authors per chunk where 10 minutes ran back up to 3.0. Widening the
# corpus to nine years changed the answer - the group is far busier, so a
# 3-minute window keeps merging unrelated threads and 28% of chunks ran over
# 500 characters. A retrieval miss made it concrete: "how do I register with
# the cultural mission" returned a passage about registering with the police,
# because both sat in the same sprawling chunk. At 1 minute the same corpus
# yields 47 short focused chunks on that topic, mean length drops from 407 to
# 283, and long chunks fall to 14%.
_GAP = timedelta(minutes=1)

# How far back a reply may point and still count as continuing the current
# conversation. 367 messages arriving after a pause reply into this window;
# treating them as new conversations would sever live threads.
_REPLY_LOOKBACK = 5

# Chunks shorter than this are folded into the previous one. Left alone, 37%
# of chunks are under 100 characters - fragments like "yes" that carry no
# retrievable meaning on their own. This floor cuts that to 18% while keeping
# authors per chunk at 2.2.
_MIN_CHARS = 60

# Upper bound before a conversation is split, always on a message boundary.
_MAX_CHARS = 1200


class TelegramParser:
    """Extracts messages from Telegram HTML exports and groups them into chunks."""

    def parse_file(self, path: Path) -> list[dict[str, Any]]:
        """Parse one exported HTML file into message dicts."""
        soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")

        title = soup.find("div", class_="text bold")
        chat_name = title.get_text(strip=True) if title else "Unknown"

        messages = []
        # Telegram omits the sender on consecutive messages from the same
        # person ("joined" messages, 25% of this corpus). Carrying the last
        # sender forward keeps those attributed instead of "Unknown".
        last_author = "Unknown"

        for div in soup.find_all("div", class_="message"):
            if "service" in div.get("class", []):
                continue
            message = self._parse_message(div, chat_name, last_author)
            if message:
                last_author = message["metadata"]["author"]
                messages.append(message)

        return messages

    def _parse_message(self, div, chat_name: str, last_author: str) -> dict[str, Any] | None:
        text_div = div.find("div", class_="text")
        if not text_div:
            return None

        content = _WHITESPACE.sub(" ", text_div.get_text(separator="\n", strip=True)).strip()
        if not content:
            return None

        # Redact contact details here, at the single point where message text
        # enters the system. Everything downstream - chunks, embeddings, the
        # committed index, the sources panel - then works from scrubbed text
        # by construction rather than by remembering to filter later.
        content = scrub(content)

        author_div = div.find("div", class_="from_name")
        date_div = div.find("div", class_="date")
        date = date_div.get("title", "") if date_div else ""

        return {
            "content": content,
            "metadata": {
                "source": "telegram",
                "chat_name": chat_name,
                # Scrub the display name too: Telegram lets people use a
                # handle as their name, and _render writes "Author: message",
                # so an unscrubbed name is a contact detail inside the chunk
                # text itself.
                "author": scrub(author_div.get_text(strip=True))
                if author_div
                else last_author,
                "date": date,
                "timestamp": self._parse_date(date),
                "message_id": div.get("id", ""),
                "reply_to": self._parse_reply_to(div),
            },
        }

    @staticmethod
    def _parse_date(title: str) -> datetime | None:
        """Read Telegram's '01.08.2019 17:25:45 UTC+00:00' timestamp."""
        try:
            return datetime.strptime(title.split(" UTC")[0], "%d.%m.%Y %H:%M:%S")
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_reply_to(div) -> str | None:
        """Return the message id this message replies to, if any."""
        reply_div = div.find("div", class_="reply_to")
        if not reply_div:
            return None
        anchor = reply_div.find("a", href=True)
        match = _REPLY_TARGET.search(anchor["href"]) if anchor else None
        return f"message{match.group(1)}" if match else None

    def parse_directory(self, directory: Path) -> list[dict[str, Any]]:
        """Parse every messages*.html file in a directory, in order."""
        messages: list[dict[str, Any]] = []
        for path in sorted(Path(directory).glob("messages*.html")):
            messages.extend(self.parse_file(path))
        return messages

    def chunk(
        self, messages: list[dict[str, Any]], size: int = _MAX_CHARS
    ) -> list[dict[str, Any]]:
        """Group messages into chunks that follow conversation boundaries.

        There is no character overlap between chunks: splitting on message
        boundaries means no sentence is ever severed, which is the only thing
        the old 50-character overlap achieved. Keeping it would splice the
        tail of one conversation onto the head of an unrelated one.
        """
        if not messages:
            return []

        groups = self._segment(messages)
        return [self._render(group) for group in self._pack(groups, size)]

    @staticmethod
    def _segment(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Split the stream wherever a conversation appears to end.

        A long pause ends a conversation, unless the next message is a reply
        into the exchange still in progress.
        """
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for message in messages:
            if not current:
                current = [message]
                continue

            previous_time = current[-1]["metadata"].get("timestamp")
            current_time = message["metadata"].get("timestamp")
            paused = (
                previous_time is not None
                and current_time is not None
                and current_time - previous_time > _GAP
            )

            reply_to = message["metadata"].get("reply_to")
            recent = {m["metadata"].get("message_id") for m in current[-_REPLY_LOOKBACK:]}
            continues_thread = bool(reply_to) and reply_to in recent

            if paused and not continues_thread:
                groups.append(current)
                current = [message]
            else:
                current.append(message)

        if current:
            groups.append(current)
        return groups

    @classmethod
    def _pack(
        cls, groups: list[list[dict[str, Any]]], max_chars: int
    ) -> list[list[dict[str, Any]]]:
        """Emit one chunk per conversation, absorbing fragments and splitting
        anything oversized on a message boundary."""
        chunks: list[list[dict[str, Any]]] = []

        for group in groups:
            length = cls._length(group)

            if (
                chunks
                and length < _MIN_CHARS
                and cls._length(chunks[-1] + group) <= max_chars
            ):
                chunks[-1] = chunks[-1] + group
                continue

            if length <= max_chars:
                chunks.append(list(group))
                continue

            part: list[dict[str, Any]] = []
            for message in group:
                if part and cls._length(part + [message]) > max_chars:
                    chunks.append(part)
                    part = [message]
                else:
                    part.append(message)
            if part:
                chunks.append(part)

        return chunks

    @staticmethod
    def _length(messages: list[dict[str, Any]]) -> int:
        """Character length of a group once rendered as Author: message lines."""
        return sum(len(m["metadata"].get("author", "")) + len(m["content"]) + 3 for m in messages)

    @classmethod
    def _render(cls, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Format a conversation as speaker-attributed lines.

        Keeping "Author: message" structure preserves who answered whom, which
        a single blended paragraph loses - it matters to both the reranker and
        to Claude when a thread contains disagreement.
        """
        content = "\n".join(
            f"{m['metadata'].get('author', 'Unknown')}: {m['content']}" for m in group
        )
        return {"content": content, "metadata": cls._merge([m["metadata"] for m in group])}

    @staticmethod
    def _merge(metadata: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine the metadata of every message contributing to a chunk."""
        if not metadata:
            return {}

        merged = dict(metadata[0])
        authors = list(dict.fromkeys(m.get("author", "Unknown") for m in metadata))
        merged["authors"] = ", ".join(authors[:5])

        dates = [m["date"] for m in metadata if m.get("date")]
        if dates:
            merged["date_range"] = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"

        merged["message_count"] = len(metadata)
        # Per-message fields describe one message, not the group they were
        # merged into; keeping them would misattribute the chunk.
        for key in ("timestamp", "reply_to", "message_id", "author"):
            merged.pop(key, None)
        return merged
