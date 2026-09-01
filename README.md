# 🎓 MurshidAI · مرشد

A bilingual (Arabic/English) retrieval-augmented chatbot that answers questions
about studying in the UK, using an archive of real Saudi scholarship student
discussions from Telegram.

Ask a question in either language; MurshidAI finds the most relevant passages
from ~4,600 archived messages and has Claude answer using only what it found.

## How it works

```
question ──▶ Voyage embedding ──▶ cosine search over a prebuilt
                                   numpy index (620 passages)
                                            │
                                      top 5 passages
                                            │
                                            ▼
                           Claude ──▶ answer, in the question's language
```

The corpus never changes at runtime, so embeddings are computed once by
`ingest.py` and committed as a single `data/index.npz`. The running app needs
no vector database and no local ML model — it loads a ~2 MB array and does a
matrix multiply. That keeps the deployed footprint small enough for free
hosting and removes the moving parts that a live demo would otherwise depend on.

## Project layout

```
murshid/
  config.py       settings from environment
  telegram.py     Telegram HTML export parser + chunker
  embeddings.py   Voyage AI client (query/document encoding)
  store.py        numpy vector store: save, load, cosine search
  rag.py          language detection, retrieval, Claude prompting
ingest.py         builds data/index.npz from the HTML export
streamlit_app.py  the chat interface
```

## Running locally

Requires Python 3.10+, an [Anthropic API key](https://console.anthropic.com/),
and a [Voyage AI key](https://www.voyageai.com/) (both have free tiers).

```bash
pip install -r requirements.txt

cp .env.example .env        # then add your two API keys

python ingest.py            # build the index (only needed once)
streamlit run streamlit_app.py
```

## Design notes

**Why Voyage instead of a local embedding model.** The original version used
`sentence-transformers`, which pulls in PyTorch — about 1 GB of dependencies
plus a model download. Voyage encodes queries and documents differently
(`input_type`), which suits question-to-passage retrieval better than a
symmetric paraphrase model, handles Arabic well, and keeps the deployed app
small enough to host for free.

**Why a committed index instead of a vector database.** An earlier version
stored vectors in hosted Postgres/pgvector. For a corpus that is fixed at build
time, that adds an external service that can expire or go offline without
adding capability. A committed array cannot.

**Language handling.** The response language follows the question, decided by
comparing Arabic against Latin character counts. Retrieval is unaffected —
the embedding model is multilingual, so an English question can surface an
Arabic passage and vice versa.

## Data and privacy

The archive comes from a public Saudi student Telegram group. The repository
includes five sample export files, selected because they contain no phone
numbers or usernames, so that ingestion is reproducible. Answers are grounded
in community discussion, not official guidance.

## License

MIT — see [LICENSE](LICENSE).
