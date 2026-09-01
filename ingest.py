"""Build the vector index from Telegram HTML exports.

Run this locally whenever the source data changes; commit the resulting
data/index.npz so the deployed app has no database to depend on.

    python ingest.py                      # default export directory
    python ingest.py --dir path/to/export
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from pathlib import Path

from murshid import config
from murshid.embeddings import EmbeddingError, embed_documents
from murshid.store import VectorStore
from murshid.telegram import TelegramParser

DEFAULT_EXPORT = Path(__file__).parent / "ChatExport_2025-10-26"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_EXPORT,
                        help="directory of Telegram HTML exports")
    parser.add_argument("--out", type=Path, default=config.INDEX_PATH,
                        help="where to write the index")
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"error: no such directory: {args.dir}", file=sys.stderr)
        return 1

    print(f"Parsing {args.dir} ...")
    messages = TelegramParser().parse_directory(args.dir)
    if not messages:
        print("error: no messages found", file=sys.stderr)
        return 1

    chunks = TelegramParser().chunk(messages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"  {len(messages):,} messages -> {len(chunks):,} chunks")

    # Embedding is slow on Voyage's free tier, so checkpoint as we go: a
    # re-run picks up where an interrupted one left off.
    cache = args.out.with_suffix(".partial.npy")
    done_vectors: list[list[float]] = []
    if cache.exists():
        done_vectors = [list(v) for v in np.load(cache)]
        print(f"Resuming from {cache.name} ({len(done_vectors)} already embedded)")

    def progress(done: int, total: int) -> None:
        print(f"  embedded {len(done_vectors) + done}/{total + len(done_vectors)}",
              end="\r", flush=True)

    remaining = [c["content"] for c in chunks][len(done_vectors):]

    print(f"Embedding with {config.VOYAGE_MODEL} ...")
    try:
        if remaining:
            for start in range(0, len(remaining), 80):
                block = remaining[start : start + 80]
                done_vectors.extend(embed_documents(block, progress=progress))
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, np.asarray(done_vectors, dtype=np.float32))
    except (EmbeddingError, KeyboardInterrupt) as exc:
        print(f"\nstopped: {exc}", file=sys.stderr)
        print(f"progress saved - re-run to resume from {len(done_vectors)}", file=sys.stderr)
        return 1

    vectors = done_vectors

    path = VectorStore.save(vectors, [c["content"] for c in chunks],
                            [c["metadata"] for c in chunks], args.out)
    size_mb = path.stat().st_size / 1_000_000
    print(f"\nWrote {path} ({size_mb:.1f} MB, {len(vectors):,} vectors)")

    cache.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
