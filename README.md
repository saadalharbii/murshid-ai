# 🎓 MurshidAI · مرشد

A bilingual (Arabic/English) chatbot that answers questions about studying in
the UK, grounded in an archive of real discussions between Saudi scholarship
students on Telegram.

Ask a question in either language. MurshidAI searches ~4,600 archived messages,
picks the passages that actually answer it, and has Claude write a reply that
cites them — using only what it found, never its own general knowledge.

## How it works

```
question ─▶ embedding ─▶ search a prebuilt index ─▶ 40 candidates
                                                         │
                                              reranking picks the best 5
                                                         │
                                    Claude ─▶ cited answer, in the
                                              question's language
```

Retrieval runs in two stages because similarity scores in this corpus sit very
close together — a single-stage search returns passages that all look about
equally relevant, and the best ones often rank well below the top five. The
first stage casts a wide net; the second compares each candidate against the
question and reorders them properly.

Embeddings are computed once by `ingest.py` and committed as `data/index.npz`
(934 passages, ~4 MB), so the app just loads an array and does a matrix
multiply. No vector database, no local ML model, nothing to keep online.

## Running it

Needs Python 3.10+ and two API keys.

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
streamlit run streamlit_app.py
```

The index is committed, so it runs straight away. Rebuild it with
`python ingest.py` only if you change the source data or the chunking.

## Checking changes

Retrieval and answer quality are measured rather than eyeballed:

```bash
python eval/run_eval.py             # recall@5, MRR, score spread
python eval/run_answer_eval.py      # citations, refusal accuracy
python -m pytest tests/ -q
```

`run_answer_eval.py` makes no model calls by default; `--judge` adds a graded
pass for faithfulness and relevance. Each script's docstring explains how to
read its numbers and where they mislead.

## Notes

**Chunking follows conversations, not character counts.** Splitting every 500
characters produced chunks averaging 7.4 messages from 3.8 different people —
topic soup that looked vaguely similar to every question. Chunks now break
where conversations do, and each line keeps its speaker.

**No local ML model, no vector database.** The first version used
`sentence-transformers` (~1 GB of PyTorch) and hosted Postgres. For a corpus
fixed at build time, both added weight and services that can expire without
adding capability.

**Language handling.** The reply language follows the question, decided by
comparing Arabic against Latin character counts, and the text is aligned to
match. Retrieval is unaffected — the embeddings are multilingual, so an English
question can surface an Arabic passage and the other way round.

## Data

The archive comes from a public Saudi student Telegram group. This repo
includes five sample export files, chosen because they contain no phone numbers
or usernames, so ingestion is reproducible.

Answers reflect what students told each other, not official guidance. The
assistant is told to say so on official matters, to flag stale advice, and to
present both sides where students disagreed.
