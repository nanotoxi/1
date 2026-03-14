# Files to have in the repo (unified backend + deploy)

Use this as a checklist. All paths are from the **project root**.

---

## New files (add these)

| File | Purpose |
|------|---------|
| `DEPLOY.md` | How to deploy backend to cloud (Railway) and use the URL directly |
| `SINGLE_BACKEND.md` | Single-backend docs: run, endpoints, DB, users |
| `run_backend.ps1` | PowerShell script to run backend from project root |
| `run_backend.bat` | Batch script to run backend from project root |
| `test_db_connection.py` | One-off script to test DB connection and insert one log row |
| `test_endpoints.py` | Script to smoke-test API + RAG endpoints (optional) |
| `REPO_FILES_ADDED.md` | This file: list of files to add/keep in repo |

---

## Modified files (ensure these are in repo with latest content)

| File | What changed |
|------|----------------|
| `Api.py` | Single FastAPI app: auth, predict (+ RAG explanation), report, admin, RAG; CORS; .env load; DB save + logging; `reload=False` |
| `database.py` | PostgreSQL: `_load_dotenv_if_needed()`, `_get_database_url()`, Railway SSL (`sslmode=require`), all table create/save/log/stats |
| `requirements.txt` | Added: `fastapi`, `uvicorn[standard]`, `psycopg2-binary`, `PyJWT` |
| `Procfile` | `web: uvicorn Api:app --host 0.0.0.0 --port ${PORT:-8000}` |
| `railway.json` | `startCommand`: `uvicorn Api:app --host 0.0.0.0 --port $PORT`; healthcheck `/health` |
| `env.example` | Added `DATABASE_URL`, `JWT_SECRET`; note about copying to `.env` |

---

## Existing files to keep (no change or minor)

| File | Note |
|------|------|
| `main.py` | Legacy Flask app; not used by single backend (Procfile runs Api.py) |
| `Nanotoxi-Rag-main/` | RAG subproject; Api.py imports from `Nanotoxi-Rag-main/backend/rag` |
| `runtime.txt` | e.g. `python-3.11.0`; keep for Railway if needed |
| `.gitignore` | Should include `.env` so secrets aren’t committed |

---

## Do not add to repo (local / secrets)

| Item | Reason |
|------|--------|
| `.env` | Contains secrets (DATABASE_URL, JWT_SECRET). Use env.example as template; set Variables in Railway. |

---

## Quick tree (root only)

```
.
├── Api.py                 # Single backend entry (FastAPI)
├── database.py            # PostgreSQL + init/save/log/stats
├── requirements.txt       # Includes FastAPI, uvicorn, psycopg2, PyJWT
├── Procfile               # web: uvicorn Api:app ...
├── railway.json           # deploy startCommand + healthcheck
├── env.example            # Template for .env (DATABASE_URL, JWT_SECRET, ...)
├── DEPLOY.md              # Deploy to cloud and use URL directly
├── SINGLE_BACKEND.md      # Run locally, endpoints, DB, users
├── run_backend.ps1        # Run backend (Windows PowerShell)
├── run_backend.bat        # Run backend (Windows CMD)
├── test_db_connection.py  # Test DB connection + one insert
├── test_endpoints.py      # Smoke-test endpoints (optional)
├── REPO_FILES_ADDED.md    # This file
├── main.py                # (optional) legacy Flask
└── Nanotoxi-Rag-main/     # RAG (Api.py uses backend/rag)
```

Commit and push the **new** and **modified** files above so the repo is ready to deploy and use as the single backend.

---

## Push this folder to your GitHub

**Repo:** [https://github.com/Yash-povie/updated-Nanotoxi-test.git](https://github.com/Yash-povie/updated-Nanotoxi-test.git)

From the project root, run:

**PowerShell:**
```powershell
.\push_to_github.ps1
```

**CMD:**
```cmd
push_to_github.bat
```

**Or manually:**
```bash
git remote add origin https://github.com/Yash-povie/updated-Nanotoxi-test.git
# if origin exists: git remote set-url origin https://github.com/Yash-povie/updated-Nanotoxi-test.git
git add -A
git commit -m "Unified backend: Api.py + PostgreSQL + RAG + deploy"
git push -u origin main
```

If the remote already has commits (e.g. from the existing [updated-Nanotoxi-test](https://github.com/Yash-povie/updated-Nanotoxi-test) content), either:
- `git pull origin main --rebase` then `git push origin main`, or
- Push to a new branch first: `git push -u origin your-branch`, then open a PR on GitHub.
