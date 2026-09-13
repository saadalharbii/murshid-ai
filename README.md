# 🎓 MurshidAI · مرشد

A bilingual (Arabic/English) chatbot that answers questions about studying in
the UK, grounded in an archive of real discussions between Saudi scholarship
students on Telegram.

Ask a question in either language. MurshidAI searches ~4,600 archived messages,
picks the passages that actually answer it, and has Claude write a reply that
cites them — using only what it found, never its own general knowledge.

## How it works

```
question
   │
   ├─▶ embedding ──▶ cosine search over a prebuilt index ──▶ 40 candidates
   │                                                              │
   │                        reranking (a second, more accurate model)
   │                                                              │
   └──────────────────────────────────────────────▶ the best 5 passages
                                                                  │
                                        Claude ──▶ cited answer, in the
                                                   question's language
```

Retrieval happens in two stages. The first casts a wide net quickly; the second
looks at each candidate alongside the question and reorders them properly. The
extra stage matters here because similarity scores in this corpus sit very
close together — a single-stage search returns passages that all look about
equally relevant, and the best ones are often ranked well below the top five.

The corpus never changes while the app runs, so all the embeddings are computed
once by `ingest.py` and committed as a single `data/index.npz` (934 passages,
~4 MB). The running app loads that array and does a matrix multiply. There is
no vector database, no local ML model, and nothing to keep online.

## Project layout

```
murshid/
  config.py       settings, read from the environment
  telegram.py     HTML export parser and conversation-aware chunker
  embeddings.py   turns text into vectors
  rerank.py       reorders candidates by relevance
  store.py        numpy vector store: save, load, cosine search
  rag.py          language detection, retrieval, prompting
  claude.py       Anthropic Messages API client
ingest.py         builds data/index.npz from the HTML export
streamlit_app.py  the chat interface
eval/             measures retrieval and answer quality
tests/            unit tests
```

## Running locally

You need Python 3.10 or newer, an [Anthropic API key](https://console.anthropic.com/),
and an [embeddings API key](https://www.voyageai.com/). Both have free tiers.

```bash
pip install -r requirements.txt

cp .env.example .env        # then fill in your two API keys

streamlit run streamlit_app.py
```

The index is committed, so the app runs straight away. Rebuild it only if you
change the source data or the chunking:

```bash
python ingest.py
```

## Evaluating changes

Retrieval and answer quality are measured rather than eyeballed, so a change
can be shown to help.

```bash
python eval/run_eval.py             # retrieval: recall@5, MRR, score spread
python eval/run_answer_eval.py      # answers: citations, refusal accuracy
python -m pytest tests/ -q          # unit tests
```

Both harnesses cache their results, and the free checks in `run_answer_eval.py`
make no model calls at all. Adding `--judge` to it asks Claude to grade each
answer for faithfulness and relevance, which does cost a few cents. Each
script's docstring explains how to read its numbers and where they mislead.

## Design notes

**Chunking follows conversations, not character counts.** Splitting the archive
every 500 characters produced chunks averaging 7.4 messages from 3.8 different
people, so most chunks were a blend of unrelated topics and looked vaguely
similar to every question. Chunks now break where conversations do — on a pause
in the discussion, unless the next message is a reply to one still on screen —
which halves that mixing. Each line keeps its speaker, so it stays clear who
answered whom.

**A hosted embedding service instead of a local model.** The original version
used `sentence-transformers`, which pulls in PyTorch: about 1 GB of
dependencies plus a model download. A hosted service handles Arabic well,
encodes questions and passages differently (which suits question-to-passage
search), and keeps the deployed app small enough to host for free.

**A committed index instead of a vector database.** An earlier version kept its
vectors in hosted Postgres. For a corpus fixed at build time, that adds a
service that can expire or go offline without adding any capability. A
committed array cannot.

**Language handling.** The reply language follows the question, decided by
comparing Arabic against Latin character counts, and the text is aligned to
match. Retrieval is unaffected: the embeddings are multilingual, so an English
question can surface an Arabic passage and the other way round.

## Data and privacy

The archive comes from a public Saudi student Telegram group. This repository
includes five sample export files, chosen because they contain no phone numbers
or usernames, so ingestion is reproducible.

Answers reflect what students told each other, not official guidance. The
assistant is told to say so on official matters, to flag advice that may have
gone stale, and to present both sides where students disagreed.

## License

MIT — see [LICENSE](LICENSE).
