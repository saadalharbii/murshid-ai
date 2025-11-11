# Quick Start Guide - MurshidAI

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Supabase account created
- [ ] Anthropic API key obtained
- [ ] Git installed

## Step-by-Step Setup (5 minutes)

### 1. Supabase Setup

**Create Project:**
1. Go to https://supabase.com and sign up
2. Click "New Project"
3. Choose organization and set project name

**Run SQL:**
1. Go to SQL Editor in left sidebar
2. Copy and paste the SQL from README (section 2.3)
3. Click "Run"

**Get Credentials:**
1. Go to Settings → API
2. Copy "Project URL"
3. Copy "anon/public" key

### 2. Anthropic API Key

1. Visit https://console.anthropic.com/
2. Sign in or create account
3. Go to API Keys
4. Create new key
5. Copy the key (starts with `sk-ant-`)

### 3. Environment Setup

**Windows:**
```bash
# Clone and navigate
cd murshid-ai

# Copy environment file
copy .env.example .env

# Edit .env with your credentials
notepad .env

# Backend setup
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup (new terminal)
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Clone and navigate
cd murshid-ai

# Copy environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or vim, code, etc.

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup (new terminal)
cd frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Ingest Data

```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python scripts/ingest_telegram.py --dir ../ChatExport_2025-10-26
```

Wait for it to complete (first time downloads ~500MB model).

### 5. Run Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
source venv/bin/activate  # Windows: venv\Scripts\activate
streamlit run streamlit_app.py
```

### 6. Test

Open http://localhost:8501 and ask:
- `ما هي أفضل الجامعات في لندن؟`
- `How do I renew my visa?`

## Common Issues

### "Module not found" error
```bash
# Make sure you're in the virtual environment
# You should see (venv) in your terminal

# Reinstall dependencies
pip install -r requirements.txt
```

### "Database connection failed"
- Check your SUPABASE_URL and SUPABASE_KEY in `.env`
- Make sure you ran the SQL script in Supabase

### "Anthropic API error"
- Check your ANTHROPIC_API_KEY in `.env`
- Verify you have credits in your Anthropic account

### Embedding model download slow
- First time only, downloads ~500MB
- Subsequent runs use cached model
- Be patient, it's normal

### Port already in use
```bash
# Backend (8000)
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Frontend (8501)
# Similar process for port 8501
```

## Quick Commands Reference

### Data Management

**Add new text manually:**
```bash
cd backend
python scripts/ingest_telegram.py --text "Your FAQ content here"
```

**Add new Telegram export:**
```bash
python scripts/ingest_telegram.py --dir /path/to/export
```

**View database stats:**
```bash
curl http://localhost:8000/api/admin/stats
```

### Development

**Reset database (WARNING: Deletes all data):**
```bash
curl -X DELETE http://localhost:8000/api/admin/documents
```

**Test API directly:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي أفضل الجامعات؟"}'
```

**View API docs:**
Open http://localhost:8000/docs

## Docker Quick Start

If you prefer Docker:

```bash
# 1. Make sure .env is configured
cp .env.example .env
# Edit .env with your credentials

# 2. Build and run
docker-compose up --build

# 3. Access
# Frontend: http://localhost:8501
# Backend: http://localhost:8000
```

## Next Steps

1. Add your own FAQs via Admin Panel
2. Customize prompts in `backend/app/rag.py`
3. Adjust chunk size in `.env` if needed
4. Deploy to Railway or Render (see README)

## Getting Help

- Check full README for detailed documentation
- API docs: http://localhost:8000/docs
- GitHub Issues: Report bugs or ask questions
