# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

MurshidAI is a bilingual (Arabic/English) RAG chatbot answering questions about
studying in the UK, grounded in an archive of Saudi scholarship student
Telegram discussions. It is a portfolio project: the priority is that it runs
reliably with no maintenance, not that it has many features.

## Architecture

Single Streamlit process. No API server, no database.

```
question -> Voyage embedding -> cosine search over data/index.npz
         -> top-5 passages -> Claude -> answer in the question's language
```

- `murshid/config.py` - settings from environment
- `murshid/telegram.py` - HTML export parser and chunker
- `murshid/embeddings.py` - Voyage AI client
- `murshid/claude.py` - Anthropic Messages API client
- `murshid/store.py` - numpy vector store
- `murshid/rag.py` - language detection, retrieval, prompting
- `ingest.py` - builds `data/index.npz`
- `streamlit_app.py` - chat interface

## Commands

```bash
streamlit run streamlit_app.py    # run the app
python ingest.py                  # rebuild the index (only after data changes)
python -m pytest tests/ -q        # run tests
```

## Two virtualenvs

`.venv-app/` (anthropic-free, ~50MB) is what the app runs on and what deploys.
`.venv/` additionally has torch and sentence-transformers and is left over from
the previous implementation; do not use it. Imports hang inside it because it
has both `httpx` and `httpx2` installed.

## Conventions

- **Both API clients use `urllib`, not `requests` or the Anthropic SDK.** The
  SDK's httpx2 stack hangs indefinitely on this machine. Keep using urllib with
  the certifi CA bundle - plain `ssl.create_default_context()` fails with
  CERTIFICATE_VERIFY_FAILED on this Python install.
- **Never add torch or sentence-transformers to `requirements.txt`.** Streamlit
  Community Cloud caps memory at 1GB; the deployed app must stay small.
- **`data/index.npz` is committed on purpose.** It is the reason the demo has
  no external dependencies. Rebuild and re-commit it when the corpus changes.
- Voyage's free tier allows 3 requests/minute, so `ingest.py` rate-limits and
  checkpoints; a re-run resumes rather than restarting.

## Data and privacy

`ChatExport_2025-10-26/` holds five sample export files, chosen because they
contain no phone numbers or usernames. A `contacts/` directory of vcards with
real names and phone numbers was removed in commit 2ddca47 and is gitignored -
do not reintroduce it. Note that it remains present in commits before 2ddca47.
