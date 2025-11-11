# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MurshidAI is a bilingual (Arabic/English) RAG-based chatbot for Saudi scholarship students. The system uses vector search (Supabase pgvector) combined with Claude 3 Haiku to answer questions from Telegram group data, FAQs, and notes.

## Architecture

**Three-tier architecture:**
1. **Frontend**: Streamlit app (port 8501) - chat interface + admin panel
2. **Backend**: FastAPI (port 8000) - REST API with RAG pipeline
3. **External Services**: Supabase (pgvector), Anthropic Claude, Sentence Transformers

**Data flow:**
```
User Query → Embedding → Vector Search (Supabase) → Top-K Retrieval →
Context + Query → Claude Prompt → Answer (language-matched)
```

## Key Commands

### Environment Setup
```bash
# Create .env from template
cp .env.example .env

# Install backend dependencies
cd backend && pip install -r requirements.txt

# Install frontend dependencies
cd frontend && pip install -r requirements.txt
```

### Running the Application

**Development (local):**
```bash
# Backend (Terminal 1)
cd backend
uvicorn app.main:app --reload

# Frontend (Terminal 2)
cd frontend
streamlit run streamlit_app.py
```

**Docker:**
```bash
docker-compose up --build
```

### Data Ingestion

**Ingest Telegram HTML exports:**
```bash
cd backend
python scripts/ingest_telegram.py --dir ../ChatExport_2025-10-26
```

**Add single text content:**
```bash
python scripts/ingest_telegram.py --text "Your content here"
```

### API Testing

**Query endpoint:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي أفضل الجامعات؟"}'
```

**View API docs:**
```
http://localhost:8000/docs
```

## Critical Architecture Details

### RAG Pipeline (`backend/app/rag.py`)

The `RAGPipeline` class orchestrates the entire Q&A flow:

1. **Language Detection**: Unicode range regex (`\u0600-\u06FF` for Arabic)
2. **Context Retrieval**: Encodes query → vector search (default: top-5, threshold 0.5)
3. **Prompt Construction**: Language-specific system prompts (Arabic/English)
4. **Answer Generation**: Claude API call with retrieved context

**Important**: Language detection determines the response language. Unsupported languages get bilingual rejection message.

### Database Layer (`backend/app/database.py`)

**Supabase Integration:**
- Table: `documents` (id, content, metadata JSONB, embedding vector(768))
- Vector search via `match_documents()` RPC function (cosine similarity)
- Embedding dimension: **768** (for `paraphrase-multilingual-mpnet-base-v2`)

**Critical**: The database schema MUST be created manually in Supabase SQL editor before first use (see `database.py` docstring or README).

### Telegram Parser (`backend/scripts/telegram_parser.py`)

Extracts messages from Telegram Desktop HTML exports:
- Parses: author, date, content, chat name
- Handles UTF-8 encoding for Arabic text
- Creates chunks with overlap (default: 500 chars, 50 overlap)
- Merges metadata for multi-message chunks

### Embeddings (`backend/app/embeddings.py`)

**Singleton pattern**: `get_embedding_model()` creates global instance
- Model: `paraphrase-multilingual-mpnet-base-v2` (supports 50+ languages)
- First run downloads ~500MB model (cached locally)
- Outputs 768-dimensional vectors

### Configuration (`backend/app/config.py`)

All settings loaded via Pydantic from `.env`:
- **Required**: `SUPABASE_URL`, `SUPABASE_KEY`, `ANTHROPIC_API_KEY`
- **Tunable**: `CHUNK_SIZE` (500), `CHUNK_OVERLAP` (50), `TOP_K_RESULTS` (5), `CLAUDE_TEMPERATURE` (0.3)

## Important Implementation Notes

### Language Handling
- **Detection**: Character-based (not langdetect library) for speed
- **System Prompts**: Separate Arabic/English prompts in `rag.py:_build_system_prompt()`
- **Rejection**: Unsupported languages get bilingual message

### Vector Search
- Uses cosine similarity (formula: `1 - embedding <=> query_embedding`)
- Default threshold: 0.5 (configurable via `retrieve_context()`)
- IVFFlat index for performance (100 lists)

### Data Chunking Strategy
- Chunks are created from consecutive messages, not individual messages
- Overlap preserves context across chunk boundaries
- Metadata merges: multiple authors, date ranges, message counts

### Admin Endpoints (`backend/app/routers/admin.py`)
- `/api/admin/upload/text`: Manual text upload (FAQs, notes)
- `/api/admin/upload/html`: Telegram HTML file upload
- `/api/admin/stats`: Database statistics
- `/api/admin/documents`: DELETE endpoint (destructive!)

## Environment Variables Required

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
ANTHROPIC_API_KEY=sk-ant-...
```

## Supabase Setup Requirement

Before running, execute this SQL in Supabase SQL Editor (from `database.py` or README):
- Enable `vector` extension
- Create `documents` table with vector(768) column
- Create IVFFlat index
- Create `match_documents()` function

## Frontend Architecture (`frontend/streamlit_app.py`)

**Two-page app:**
1. **Chat Page**: Message history, query input, source display (expandable)
2. **Admin Page**: Stats, text upload, HTML file upload

**Session State**: Stores `messages` list and `show_sources` toggle

## Deployment Notes

- Railway/Render: Set environment variables in dashboard
- Docker: Uses `docker-compose.yml` with backend/frontend services
- CORS: Currently allows all origins (tighten in production)

## Common Patterns

**Singleton instances:**
- `db` in `database.py`
- `embedding_model` via `get_embedding_model()`
- `rag_pipeline` via `get_rag_pipeline()`

**Error handling:**
- All modules use `loguru` for logging
- API errors return HTTPException with status codes
- Database failures logged but don't crash app
