# 🎓 MurshidAI - مرشد

**Bilingual AI Assistant for Saudi Scholarship Students**

MurshidAI is a Retrieval-Augmented Generation (RAG) chatbot that helps Saudi scholarship students in the UK by answering questions based on community knowledge from Telegram groups, FAQs, and notes. It supports both Arabic and English.

## ✨ Features

- 🌐 **Bilingual Support**: Automatically detects and responds in Arabic or English
- 🧠 **RAG Architecture**: Combines vector search with Claude 3 Haiku for accurate answers
- 📚 **Knowledge Base**: Built from Telegram group messages, FAQs, and manual notes
- ⚡ **Fast API**: RESTful API built with FastAPI
- 🎨 **User-Friendly UI**: Streamlit-based chat interface
- 🔧 **Admin Panel**: Easy data management and upload
- 🐳 **Docker Support**: Containerized for easy deployment

## 🏗️ Architecture

```
┌─────────────┐
│  Streamlit  │  Frontend (Port 8501)
│   Frontend  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   FastAPI   │  Backend (Port 8000)
│   Backend   │
└──────┬──────┘
       │
       ├──────► Supabase (PostgreSQL + pgvector)
       ├──────► Sentence Transformers (Embeddings)
       └──────► Anthropic Claude 3 Haiku (LLM)
```

## 📋 Prerequisites

- Python 3.10 or higher
- Supabase account (free tier works)
- Anthropic API key (Claude)
- Docker (optional, for containerized deployment)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/murshid-ai.git
cd murshid-ai
```

### 2. Set Up Supabase

1. Create a free account at [Supabase](https://supabase.com)
2. Create a new project
3. Go to **SQL Editor** and run the following SQL:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create function for similarity search
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(768),
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
$$;
```

4. Get your **Project URL** and **anon/public key** from Settings → API

### 3. Get Anthropic API Key

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Create an API key
4. Copy the key (you'll need it in the next step)

### 4. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Adjust these if needed
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
CLAUDE_TEMPERATURE=0.3
```

### 5. Install Dependencies

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Frontend
```bash
cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 6. Ingest Your Data

The project includes Telegram export data in `ChatExport_2025-10-26/`. To ingest it:

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python scripts/ingest_telegram.py --dir ../ChatExport_2025-10-26
```

This will:
- Parse all HTML message files
- Create text chunks
- Generate embeddings using Sentence Transformers
- Upload everything to Supabase

**Note**: First run will download the embedding model (~500MB). This is normal and only happens once.

### 7. Run the Application

#### Option A: Run Locally

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
source venv/bin/activate
streamlit run streamlit_app.py
```

Then visit:
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

#### Option B: Run with Docker

```bash
# Make sure .env file is configured
docker-compose up --build
```

Then visit:
- **Frontend**: http://localhost:8501
- **Backend**: http://localhost:8000

## 📖 Usage

### Chat Interface

1. Open the Streamlit app at http://localhost:8501
2. Type your question in Arabic or English
3. The system will:
   - Detect your language
   - Search for relevant context in the database
   - Generate an answer using Claude
   - Display sources (optional)

**Example Questions:**
- Arabic: `ما هي أفضل الجامعات في لندن؟`
- English: `What are the best universities in London?`
- Arabic: `كيف أجدد تأشيرتي؟`
- English: `How do I renew my visa?`

### Admin Panel

Access the admin panel from the sidebar to:

1. **View Statistics**: See total documents in database
2. **Upload Text**: Manually add FAQs or notes
3. **Upload HTML Files**: Add more Telegram exports

## 🔧 Adding New Data

### Method 1: Via Streamlit Admin Panel

1. Go to Admin Panel in the sidebar
2. Choose upload method:
   - **Text**: Paste content directly
   - **HTML Files**: Upload Telegram export files

### Method 2: Via Command Line

```bash
cd backend
python scripts/ingest_telegram.py --dir /path/to/telegram/export

# Or ingest a single text
python scripts/ingest_telegram.py --text "Your content here"
```

### Method 3: Via API

```bash
curl -X POST "http://localhost:8000/api/admin/upload/text" \
  -H "Content-Type: application/json" \
  -d '{"content": "Your FAQ content", "metadata": {"source": "manual", "type": "faq"}}'
```

## 📁 Project Structure

```
murshid-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration
│   │   ├── models.py            # Pydantic models
│   │   ├── database.py          # Supabase client
│   │   ├── embeddings.py        # Sentence Transformers
│   │   ├── rag.py               # RAG pipeline
│   │   └── routers/
│   │       ├── query.py         # Q&A endpoint
│   │       └── admin.py         # Admin endpoints
│   ├── scripts/
│   │   ├── telegram_parser.py   # HTML parser
│   │   └── ingest_telegram.py   # Data ingestion
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py         # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── ChatExport_2025-10-26/       # Telegram data
├── .env.example
├── docker-compose.yml
└── README.md
```

## 🌐 API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

### Key Endpoints

- `POST /api/query` - Ask a question
- `POST /api/admin/upload/text` - Upload text content
- `POST /api/admin/upload/html` - Upload HTML files
- `GET /api/admin/stats` - Get database statistics
- `GET /health` - Health check

## 🚢 Deployment

### Railway (Recommended - Free Tier Available)

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and initialize:
```bash
railway login
railway init
```

3. Add environment variables in Railway dashboard

4. Deploy:
```bash
railway up
```

### Render

1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables
6. Deploy

### Docker Deployment (Any Platform)

The project includes Docker configuration for easy deployment on any platform that supports Docker (AWS, GCP, Azure, DigitalOcean, etc.).

## 🛠️ Configuration

All configuration is done via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Your Supabase project URL | Required |
| `SUPABASE_KEY` | Your Supabase anon key | Required |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `CHUNK_SIZE` | Text chunk size (characters) | 500 |
| `CHUNK_OVERLAP` | Overlap between chunks | 50 |
| `TOP_K_RESULTS` | Number of results to retrieve | 5 |
| `CLAUDE_TEMPERATURE` | Claude temperature (0-1) | 0.3 |
| `CLAUDE_MODEL` | Claude model name | claude-3-haiku-20240307 |
| `EMBEDDING_MODEL` | Sentence Transformers model | paraphrase-multilingual-mpnet-base-v2 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- UI powered by [Streamlit](https://streamlit.io/)
- LLM by [Anthropic Claude](https://www.anthropic.com/)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- Vector database by [Supabase](https://supabase.com/)

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

Made with ❤️ for Saudi scholarship students in the UK