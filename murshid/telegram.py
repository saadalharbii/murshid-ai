"""Parser for Telegram Desktop HTML exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

_WHITESPACE = re.compile(r"\s+")


class TelegramParser:
    """Extracts messages from Telegram HTML exports and groups them into chunks."""

    def parse_file(self, path: Path) -> list[dict[str, Any]]:
        """Parse one exported HTML file into message dicts."""
        soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")

        title = soup.find("div", class_="text bold")
        chat_name = title.get_text(strip=True) if title else "Unknown"

        messages = []
        for div in soup.find_all("div", class_="message"):
            if "service" in div.get("class", []):
                continue
            message = self._parse_message(div, chat_name)
            if message:
                messages.append(message)

        return messages

    def _parse_message(self, div, chat_name: str) -> dict[str, Any] | None:
        text_div = div.find("div", class_="text")
        if not text_div:
            return None

        content = _WHITESPACE.sub(" ", text_div.get_text(separator="\n", strip=True)).strip()
        if not content:
            return None

        author_div = div.find("div", class_="from_name")
        date_div = div.find("div", class_="date")

        return {
            "content": content,
            "metadata": {
                "source": "telegram",
                "chat_name": chat_name,
                "author": author_div.get_text(strip=True) if author_div else "Unknown",
                "date": date_div.get("title", "") if date_div else "",
                "message_id": div.get("id", ""),
            },
        }

    def parse_directory(self, directory: Path) -> list[dict[str, Any]]:
        """Parse every messages*.html file in a directory, in order."""
        messages: list[dict[str, Any]] = []
        for path in sorted(Path(directory).glob("messages*.html")):
            messages.extend(self.parse_file(path))
        return messages

    def chunk(
        self, messages: list[dict[str, Any]], size: int = 500, overlap: int = 50
    ) -> list[dict[str, Any]]:
        """Group consecutive messages into overlapping chunks for embedding."""
        chunks: list[dict[str, Any]] = []
        buffer = ""
        buffered_meta: list[dict] = []

        for message in messages:
            content = message["content"]

            if buffer and len(buffer) + len(content) > size:
                chunks.append({"content": buffer.strip(), "metadata": self._merge(buffered_meta)})
                buffer = (buffer[-overlap:] if len(buffer) > overlap else buffer) + "\n\n" + content
                buffered_meta = [message["metadata"]]
            else:
                buffer = f"{buffer}\n\n{content}" if buffer else content
                buffered_meta.append(message["metadata"])

        if buffer:
            chunks.append({"content": buffer.strip(), "metadata": self._merge(buffered_meta)})

        return chunks

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
        return merged
