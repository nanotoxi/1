# Single backend (run from here)

This repo is the **one backend** for Nanotox AI. Run it locally; no separate “online” backend needed.

## Run

From the project root:

```bash
python Api.py
```

Or on Windows:

```powershell
.\run_backend.ps1
```

```cmd
run_backend.bat
```

Server: **http://0.0.0.0:8000** (docs: http://127.0.0.1:8000/docs)

## Database (predictions + logs in cloud)

- **Local:** put `DATABASE_URL=postgresql://...` in a **.env** file in the project root.
- **Cloud (Railway):** `.env` is not deployed. In Railway → your **web service** (the one running this API) → **Variables** → add:
  - `DATABASE_URL` = your Postgres connection URL (e.g. from Railway Postgres “Connect”).
  Then **redeploy** the service. After that, the cloud backend will write predictions and logs to the same database.

## Requirements

- **.env** (local) or **Variables** (Railway) with at least:
  - `DATABASE_URL=postgresql://...` (e.g. Railway Postgres)
  - Optional: `JWT_SECRET=...`
- **Ollama** (optional): for RAG explanations; run `ollama serve` and use model from `Nanotoxi-Rag-main/backend/rag/config.py` (e.g. `llama3.2:1b-instruct`).
- **RAG index** (optional): build FAISS index in `Nanotoxi-Rag-main/backend` for similar records and RAG answers.

## Endpoints (all in one app)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Service info |
| GET | `/health` | No | Health check |
| POST | `/auth/token` | No | Login → JWT (body: `username`, `password`) |
| POST | `/predict` | Bearer | **Prediction** + optional **RAG explanation** (`use_rag: true`) + optional report (`generate_report: true`) |
| GET | `/report/{report_id}` | Bearer | Download PDF report |
| GET | `/dataset/info` | Bearer (admin) | Dashboard stats from DB |
| GET | `/logs` | Bearer (admin) | System logs from DB |
| GET | `/rag/info` | No | RAG config (vector store, LLM model) |
| POST | `/rag/search` | Bearer | RAG retrieval only (no LLM) |
| POST | `/rag/answer` | Bearer | RAG question-answering (with LLM) |

## Prediction flow

1. **POST /predict** with nanoparticle fields.
2. Response includes: `prediction`, `confidence`, `similar_records` (from RAG/FAISS).
3. If `use_rag: true`, response also includes **explanation** (RAG + LLM, e.g. Ollama).
4. If `generate_report: true`, response includes `report_url`; use **GET /report/{id}** to download PDF.

All prediction requests are stored in PostgreSQL (when `DATABASE_URL` is set).

## Users (built-in)

- **admin** / titan2024 → role `admin` (can call `/dataset/info`, `/logs`)
- **researcher1** / nano@123 → role `customer` (predict, RAG, report)
