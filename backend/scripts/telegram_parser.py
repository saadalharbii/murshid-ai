"""
Parser for Telegram HTML exports.
Extracts messages, metadata, and handles Arabic text encoding.
"""

from bs4 import BeautifulSoup
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
import re


class TelegramParser:
    """Parser for Telegram HTML export files."""

    def __init__(self):
        """Initialize the parser."""
        self.messages: List[Dict[str, Any]] = []

    def parse_html_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a single Telegram HTML export file.

        Args:
            file_path: Path to the HTML file

        Returns:
            List of parsed messages with content and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'lxml')

            # Extract chat name
            chat_name = soup.find('div', class_='text bold')
            chat_name_text = chat_name.get_text(strip=True) if chat_name else "Unknown"

            # Find all message divs
            message_divs = soup.find_all('div', class_='message')

            parsed_messages = []

            for msg_div in message_divs:
                # Skip service messages (date markers, etc.)
                if 'service' in msg_div.get('class', []):
                    continue

                message_data = self._parse_message(msg_div, chat_name_text)
                if message_data and message_data['content'].strip():
                    parsed_messages.append(message_data)

            logger.info(f"Parsed {len(parsed_messages)} messages from {file_path}")
            return parsed_messages

        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []

    def _parse_message(self, msg_div, chat_name: str) -> Dict[str, Any]:
        """
        Parse a single message div.

        Args:
            msg_div: BeautifulSoup div element containing the message
            chat_name: Name of the chat/group

        Returns:
            Dictionary with message content and metadata
        """
        try:
            # Extract message ID
            msg_id = msg_div.get('id', 'unknown')

            # Extract author name
            from_name_div = msg_div.find('div', class_='from_name')
            author = from_name_div.get_text(strip=True) if from_name_div else "Unknown"

            # Extract date/time
            date_div = msg_div.find('div', class_='date')
            date_str = date_div.get('title', '') if date_div else ''

            # Extract message text
            text_div = msg_div.find('div', class_='text')
            if not text_div:
                return None

            # Get text content, preserving line breaks
            message_text = text_div.get_text(separator='\n', strip=True)

            # Clean up the text
            message_text = self._clean_text(message_text)

            if not message_text:
                return None

            return {
                'content': message_text,
                'metadata': {
                    'source': 'telegram',
                    'chat_name': chat_name,
                    'author': author,
                    'date': date_str,
                    'message_id': msg_id
                }
            }

        except Exception as e:
            logger.error(f"Error parsing individual message: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text from HTML

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs (optional - keep if they're useful)
        # text = re.sub(r'http[s]?://\S+', '', text)

        # Remove special characters that might interfere (keep Arabic)
        # text = re.sub(r'[^\w\s\u0600-\u06FF.,!?;:()\-]', '', text)

        return text.strip()

    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Parse all HTML files in a directory.

        Args:
            directory_path: Path to directory containing HTML files

        Returns:
            List of all parsed messages
        """
        all_messages = []
        directory = Path(directory_path)

        # Find all .html files
        html_files = list(directory.glob('messages*.html'))

        logger.info(f"Found {len(html_files)} HTML files to parse")

        for html_file in sorted(html_files):
            messages = self.parse_html_file(str(html_file))
            all_messages.extend(messages)

        logger.info(f"Total messages parsed: {len(all_messages)}")
        return all_messages

    def chunk_messages(
        self,
        messages: List[Dict[str, Any]],
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Split messages into chunks for embedding.
        Groups consecutive messages into chunks of approximately chunk_size tokens.

        Args:
            messages: List of parsed messages
            chunk_size: Approximate size of each chunk (in characters, rough token estimate)
            overlap: Number of characters to overlap between chunks

        Returns:
            List of chunks with combined content and merged metadata
        """
        chunks = []
        current_chunk = ""
        current_metadata = []

        for msg in messages:
            content = msg['content']
            metadata = msg['metadata']

            # If adding this message would exceed chunk_size and we have content
            if len(current_chunk) + len(content) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': self._merge_metadata(current_metadata)
                })

                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + content
                current_metadata = [metadata]
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + content
                else:
                    current_chunk = content
                current_metadata.append(metadata)

        # Add final chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': self._merge_metadata(current_metadata)
            })

        logger.info(f"Created {len(chunks)} chunks from {len(messages)} messages")
        return chunks

    def _merge_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge metadata from multiple messages in a chunk.

        Args:
            metadata_list: List of metadata dictionaries

        Returns:
            Merged metadata dictionary
        """
        if not metadata_list:
            return {}

        merged = metadata_list[0].copy()

        # Combine authors
        authors = list(set(m.get('author', 'Unknown') for m in metadata_list))
        merged['authors'] = ', '.join(authors[:5])  # Limit to 5 authors

        # Use first and last dates
        dates = [m.get('date', '') for m in metadata_list if m.get('date')]
        if dates:
            merged['date_range'] = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else dates[0]

        # Count messages in chunk
        merged['message_count'] = len(metadata_list)

        return merged


if __name__ == "__main__":
    # Test the parser
    parser = TelegramParser()

    # Parse a single file
    messages = parser.parse_html_file("../../ChatExport_2025-10-26/messages.html")
    print(f"Parsed {len(messages)} messages")

    if messages:
        print("\nFirst message:")
        print(messages[0])

    # Parse entire directory
    all_messages = parser.parse_directory("../../ChatExport_2025-10-26/")
    print(f"\nTotal messages from directory: {len(all_messages)}")

    # Create chunks
    chunks = parser.chunk_messages(all_messages, chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks")

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0])
