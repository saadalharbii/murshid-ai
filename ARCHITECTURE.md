# Architecture notes

Engineering notes for working in this repository: how the pipeline fits
together, and the reasoning behind decisions that are not obvious from the
code.

## Project

MurshidAI is a bilingual (Arabic/English) RAG chatbot answering questions about
studying in the UK, grounded in an archive of Saudi scholarship student
Telegram discussions. The design priority is that it runs unattended, with
nothing to restart or renew, ahead of adding features.

## Architecture

Single Streamlit process. No API server, no database.

```
question -> Voyage embedding -> cosine search over data/index.npz (top-10)
         -> Voyage rerank -> top-5 passages
         -> Claude -> answer in the question's language
```

Cosine similarity barely separates chunks in this corpus, so a cross-encoder
reranker orders the final passages. Note that the rerank score is a weak
confidence signal, not a safety net - measured, answerable and unanswerable
questions overlap (0.51-0.84 against 0.37-0.66). Declining an out-of-archive
question is the system prompt's job; `RERANK_THRESHOLD` only trims the worst
matches first. The candidate pool is deliberately narrow:
widening it measurably hurts, because the reranker over-promotes chunks that
restate the question rather than answer it, and a wider net gives it more of
those to find. See the note in `config.py`. If reranking fails the vector order
is used instead - a worse answer beats none.

Follow-ups are rewritten before any of this. Retrieval sees one question at a
time, so "what about Manchester?" after a question about London rent would
search for Manchester in general. When there is earlier conversation, Claude
first turns the latest message into a standalone question (about 0.7s), and
that is what is searched and answered; the app shows it under the answer. The
rewrite prompt tells it to copy back anything that does not refer to the
chat word for word - an earlier version helpfully added "in the UK" to
standalone questions, which quietly changed their search results. A first
question skips the call entirely, and a failed rewrite searches the question
as typed.

The same holds one step earlier. If the question cannot be embedded at all,
keyword search (BM25 over words and 4-letter fragments, `murshid/lexical.py`)
supplies the passages and Claude is told search is degraded, so a gap reads
as "search is limited right now" rather than "the archive does not cover
this". It is markedly weaker - 24 of 28 eval questions against Voyage's 28,
and 6 of 10 in English - but it needs no service, so a Voyage outage costs
answer quality instead of the whole app. Local embedding models were measured
as a replacement for Voyage and rejected: the best one that fits Streamlit's
1 GB (multilingual-e5-small) found relevant passages for 23 of 28 questions,
6 of 10 in English, and returned spam for questions Voyage answers well.

Past the fallbacks, the page itself is built not to show a traceback. Every
transport failure - a reset connection, a reply cut off mid-stream, a garbled
body - is mapped onto the error its fallback handles, and anything unforeseen
still reaches the visitor as one "try again shortly" line in their language,
with the traceback in the logs. Deploys are covered too: Streamlit Cloud pulls
new files into the running process, which once left old package code loaded
under new app code, so the app re-imports `murshid/` whenever its files
change. `tests/test_app.py` runs the page against a faked network, with every
service down or garbled, and reproduces that deploy.

- `murshid/config.py` - settings from environment
- `murshid/telegram.py` - HTML export parser and conversation-aware chunker
- `murshid/scrub.py` - redacts contact details at parse time
- `murshid/filters.py` - drops filler messages and answerless chunks
- `murshid/embeddings.py` - Voyage AI client
- `murshid/claude.py` - Anthropic Messages API client
- `murshid/store.py` - numpy vector store
- `murshid/rag.py` - language detection, retrieval, prompting
- `murshid/rerank.py` - Voyage reranker, with vector-order fallback
- `murshid/lexical.py` - keyword search, the fallback when embedding fails
- `eval/` - retrieval and answer-quality harnesses (see their docstrings)
- `ingest.py` - builds `data/index.npz`
- `streamlit_app.py` - chat interface

## Commands

```bash
streamlit run streamlit_app.py    # run the app
python ingest.py                  # rebuild the index (only after data changes)
python -m pytest tests/ -q        # run tests
```

## Design constraints

- **The API clients use `urllib`, not the vendor SDKs.** During development the
  Anthropic SDK's HTTP stack hung indefinitely on some macOS Python installs,
  and urllib keeps the deployed app to three dependencies. Requests use
  certifi's CA bundle, because python.org builds on macOS cannot always read
  the system trust store.
- **No torch or sentence-transformers in `requirements.txt`.** Streamlit
  Community Cloud caps memory at 1GB, and every runtime dependency is
  reinstalled on each cold start.
- **`data/index.npz` is committed on purpose.** It is the reason the demo has
  no database to keep alive. Rebuild and re-commit it when the corpus changes.
- **Ingestion checkpoints after every batch**, so an interrupted run resumes
  rather than restarting. Rate limits are waited out when the API answers 429
  instead of throttling every batch in advance.

## Ingestion

`ingest.py` runs parse -> scrub -> filter -> chunk -> embed. Two of those steps
exist because of measurement rather than taste:

- **scrub** removes phone numbers, handles, emails and invite links. It has to
  happen at parse time: the sources panel renders retrieved chunk text
  directly, so anything in the index is on the page, and the system prompt
  never sees it. `ingest.py` refuses to write an index if any chunk still
  holds contact details.
- **filter** drops acknowledgements before chunking and answerless chunks
  after. 19% of chunks were a question with no reply in them; they score well
  against a user's question precisely because they are questions, and then
  contribute nothing. Removing them is what moved reranking ahead of plain
  vector search on the eval.

## Data and privacy

`data/telegram_sample/` holds 67 pages of a Telegram HTML export, sampled
evenly across 2017-2025; only the message pages are kept, since the parser
reads nothing else. Contact details are redacted at parse time (see Ingestion
above), and `ingest.py` refuses to build an index that still contains any, so
neither the index nor the app carries them.
