# RAG Module (Phase 1)

## Overview

Indexing pipeline for the Nanotoxi AI RAG system. Uses **100% open-source, local** components:

- **Embeddings:** `sentence-transformers` (e.g. `all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`)
- **Vector store:** FAISS (default) or ChromaDB (persisted on disk)
- **LLM (local, free):**
  - Hugging Face (default): `Qwen/Qwen2.5-0.5B-Instruct`
  - Or Ollama (recommended for easier local serving)

## Training CSV Format

Place your training dataset at `backend/data/training_data.csv` (or set `RAG_TRAINING_CSV`).

## Excel Dataset (Mumbai Final) → training CSV

If your dataset is in Excel (e.g. `Pravin sir 03_03_26 data set  mumbai- Final.xlsx`), convert it first:

```bash
cd backend
source .venv/bin/activate
python -m rag.prepare_dataset --xlsx "../Pravin sir 03_03_26 data set  mumbai- Final.xlsx"
```

This writes:

- `backend/data/training_data.csv` (cleaned, indexer-ready)
- `backend/data/classified_data_raw.csv` (raw export of the source sheet)

Expected columns (configurable in `config.py` or via `--columns-json`):

| Column           | Description                    |
|------------------|--------------------------------|
| `id`             | Unique sample/row identifier   |
| `size`           | Particle size (e.g. nm)        |
| `zeta_potential` | Surface charge                 |
| `concentration`  | Concentration                  |
| `composition`    | e.g. Ag, TiO2, Au             |
| `coating`        | e.g. citrate, PEG, silica      |
| `toxicity`       | Known toxicity (e.g. high/low/medium) |

If your CSV uses different names, set `DEFAULT_FEATURE_COLUMNS` in `config.py` or pass `--columns-json '{"size":"diameter_nm",...}'`.

## Requirements (add to backend)

```
sentence-transformers>=2.2.0
chromadb>=0.4.0
faiss-cpu>=1.7.0
pandas>=2.0.0
openpyxl>=3.1.0
transformers>=4.41.0
accelerate>=0.20.0
sentencepiece>=0.2.0
```

## Run Indexing (all in one)

**One command from project root** (creates venv, installs deps, runs indexer with dataset):

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python -m rag.indexer --vector-store faiss
```

Or use the script (from repo root; uses venv if present, defaults to FAISS):

```bash
chmod +x backend/run_rag_index.sh && ./backend/run_rag_index.sh
```

The script sets `HF_HOME` to `backend/.cache/huggingface` so the embedding model is cached inside the project.
If `RAG_SOURCE_XLSX` exists (defaults to `../Pravin sir 03_03_26 data set  mumbai- Final.xlsx`), it will auto-generate `backend/data/training_data.csv` before indexing.

**From the backend directory** (with venv activated):

```bash
# Default: FAISS (ChromaDB has Pydantic v1 issues on Python 3.14)
python -m rag.indexer

# ChromaDB (use on Python 3.12 or earlier if needed)
RAG_VECTOR_STORE=chromadb python -m rag.indexer

# Custom CSV
python -m rag.indexer --csv /path/to/your_data.csv --vector-store faiss
```

**Note:** On Python 3.14, use `--vector-store faiss`; ChromaDB’s Settings class fails with Pydantic v1.

Environment variables:

- `RAG_TRAINING_CSV` – path to training CSV
- `RAG_EMBEDDING_MODEL` – e.g. `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`
- `RAG_VECTOR_STORE` – `chromadb` or `faiss`
- `RAG_CHROMA_DIR` – ChromaDB persist directory
- `RAG_FAISS_INDEX` – path to FAISS index file
- `RAG_LLM_MODEL` – local LLM model name (default `Qwen/Qwen2.5-0.5B-Instruct`)
- `RAG_LLM_MAX_NEW_TOKENS` – answer length limit
- `RAG_LLM_BACKEND` – `hf` (default) or `ollama`
- `RAG_OLLAMA_URL` – Ollama base URL (default `http://localhost:11434`)
- `RAG_OLLAMA_MODEL` – Ollama model name (e.g. `qwen2.5:1.5b-instruct`)

Index is saved under `backend/rag_index/` by default.

## Run Retrieval + LLM Q&A (local)

After you have indexed (FAISS), you can run a local API:

```bash
cd backend
source .venv/bin/activate
python app.py
```

Endpoints:

- `GET /health`
- `POST /rag/search` body: `{ "query": "...", "top_k": 5 }`
- `POST /rag/answer` body: `{ "question": "...", "top_k": 5 }` (returns answer + citations)

## Ollama option (recommended)

1) Install Ollama on your machine, then pull a small model:

```bash
ollama pull qwen2.5:1.5b-instruct
```

2) Run the backend using Ollama as the LLM:

```bash
cd backend
source .venv/bin/activate
export RAG_LLM_BACKEND=ollama
export RAG_OLLAMA_MODEL="qwen2.5:1.5b-instruct"
python app.py
```

## Quick tests

End-to-end smoke test (prepare → index → answer):

```bash
cd backend
source .venv/bin/activate
python -m rag.smoke_test
```

Retrieval evaluation (small sanity set):

```bash
python -m rag.eval
```

## Threshold benchmark (50–100+ questions, accuracy by field)

Runs a battery of numeric questions against the RAG pipeline and reports accuracy per field (Toxic %, Zeta_mV_median, Size_nm_median, Hydro_size_nm_median, Exposure_hrs_median, LC50_val_median, IC50_val_median). Uses `backend/data/thresholds_raw.csv` (generated when you run `prepare_dataset` with both Excel sheets).

```bash
cd backend
source .venv/bin/activate
python -m rag.threshold_benchmark
```

- **Default:** 20 questions per field (140 total with 7 fields). Summary is printed to stdout.
- **Quick run:** `RAG_BENCHMARK_PER_FIELD=5 python -m rag.threshold_benchmark` (35 questions).
- **Full JSON:** `RAG_BENCHMARK_JSON=1 python -m rag.threshold_benchmark` for machine-readable output with examples.
