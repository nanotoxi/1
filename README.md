# Deploy – everything in one place

This folder contains the full backend so you can **upload or push it in one go**.

## What’s inside

- **Api.py** – Single FastAPI app (auth, predict, RAG, admin, reports)
- **database.py** – PostgreSQL (predictions, logs, stats)
- **requirements.txt** – Python deps (FastAPI, uvicorn, psycopg2, etc.)
- **Procfile** + **railway.json** – How to run on Railway
- **env.example** – Copy to `.env` and set `DATABASE_URL`, `JWT_SECRET`
- **run_backend.ps1 / .bat** – Run backend locally
- **push_to_github.ps1 / .bat** – Push to GitHub
- **DEPLOY.md, DEPLOY_CHECKLIST.md, SINGLE_BACKEND.md, REPO_FILES_ADDED.md** – Docs
- **test_db_connection.py, test_endpoints.py** – Optional tests
- **Nanotoxi-Rag-main/** – RAG (FAISS + LLM); needed for similar records and RAG explanation

## Use this folder

1. **Add `.env`** here (copy from `env.example`), set at least:
   - `DATABASE_URL=postgresql://...`
   - `JWT_SECRET=...` (optional)

2. **Upload / push**
   - Zip this `deploy` folder and upload where you need, or
   - From inside `deploy`:  
     `git init` → add remote → `git add -A` → `git commit` → `git push`

3. **Deploy to Railway**
   - Connect this repo (or upload) to Railway
   - Set `DATABASE_URL` in the web service Variables
   - Deploy

4. **Run locally**
   - From this folder: `python Api.py` or `.\run_backend.ps1`

All in one place – no need to gather files from elsewhere.
