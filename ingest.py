"""Build the vector index from Telegram HTML exports.

Run this locally whenever the source data changes; commit the resulting
data/index.npz so the deployed app has no database to depend on.

    python ingest.py                      # default export directory
    python ingest.py --dir path/to/export
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
from pathlib import Path

from murshid import config
from murshid.embeddings import EmbeddingError, embed_documents
from murshid.scrub import contains_contact_details
from murshid.store import VectorStore
from murshid.telegram import TelegramParser

DEFAULT_EXPORT = Path(__file__).parent / "ChatExport_2025-10-26"


def corpus_fingerprint(contents: list[str]) -> str:
    """Identify a corpus by its chunk count and content hash.

    Both matter: a change in chunking that happens to preserve the count still
    invalidates every checkpointed vector, because the text behind each
    position moved.
    """
    digest = hashlib.sha256()
    digest.update(str(len(contents)).encode())
    for content in contents:
        # Hash the length alongside the bytes so chunk boundaries are part of
        # the identity. Without it ["ab", "cd"] and ["abc", "d"] hash the same:
        # identical concatenated text, different boundaries, and every vector
        # after the first would be misaligned.
        encoded = content.encode("utf-8")
        digest.update(str(len(encoded)).encode())
        digest.update(b"\x00")
        digest.update(encoded)
    return f"{len(contents)}:{digest.hexdigest()[:16]}"


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

    # The parser scrubs contact details, but the index is committed and public,
    # so a redaction bug must stop the build rather than ship quietly.
    leaked = [c for c in chunks if contains_contact_details(c["content"])]
    if leaked:
        print(f"error: {len(leaked)} chunks still contain contact details after "
              f"scrubbing - refusing to build an index. First offender:\n"
              f"  {leaked[0]['content'][:200]}", file=sys.stderr)
        return 1

    # Embedding is checkpointed so an interrupted run resumes instead of
    # restarting. The checkpoint is only valid for the exact corpus that
    # produced it: vectors are matched to chunks by position, so resuming
    # against different chunks pairs every vector with the wrong text and
    # produces an index whose citations point at unrelated passages. The
    # fingerprint below ties the checkpoint to its corpus, and a mismatch
    # discards it rather than silently corrupting the index.
    cache = args.out.with_suffix(".partial.npy")
    stamp = cache.with_suffix(".corpus")
    fingerprint = corpus_fingerprint([c["content"] for c in chunks])

    done_vectors: list[list[float]] = []
    if cache.exists():
        if stamp.exists() and stamp.read_text().strip() == fingerprint:
            done_vectors = [list(v) for v in np.load(cache)]
            print(f"Resuming from {cache.name} ({len(done_vectors)} already embedded)")
        else:
            print(f"Discarding {cache.name}: it was built from a different corpus")
            cache.unlink(missing_ok=True)
            stamp.unlink(missing_ok=True)

    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fingerprint)

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
                stamp.write_text(fingerprint)
    except (EmbeddingError, KeyboardInterrupt) as exc:
        print(f"\nstopped: {exc}", file=sys.stderr)
        print(f"progress saved - re-run to resume from {len(done_vectors)}", file=sys.stderr)
        return 1

    vectors = done_vectors

    # A last check before writing: save() pairs vectors to contents by index,
    # so a length mismatch here would ship a broken index.
    if len(vectors) != len(chunks):
        print(f"error: {len(vectors)} vectors for {len(chunks)} chunks - refusing "
              f"to write a mismatched index. Delete {cache.name} and re-run.",
              file=sys.stderr)
        return 1

    path = VectorStore.save(vectors, [c["content"] for c in chunks],
                            [c["metadata"] for c in chunks], args.out)
    size_mb = path.stat().st_size / 1_000_000
    print(f"\nWrote {path} ({size_mb:.1f} MB, {len(vectors):,} vectors)")

    cache.unlink(missing_ok=True)
    stamp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
