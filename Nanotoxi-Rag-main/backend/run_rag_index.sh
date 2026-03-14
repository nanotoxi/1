#!/usr/bin/env bash
# Run RAG Phase 1 indexing: load CSV -> embed -> store in ChromaDB/FAISS
# Run from project root or backend: ./backend/run_rag_index.sh
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"

# Use project-local cache so Hugging Face models are stored under backend/
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME/hub"

# Prefer venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Using venv at .venv"
fi

echo "Installing RAG dependencies (if needed)..."
"$PIP" install -q -r requirements.txt 2>/dev/null || true

# If the Mumbai Excel dataset exists, convert it to backend/data/training_data.csv first
RAG_SOURCE_XLSX="${RAG_SOURCE_XLSX:-../Pravin sir 03_03_26 data set  mumbai- Final.xlsx}"
if [ -f "$RAG_SOURCE_XLSX" ]; then
  echo "Preparing training CSV from Excel: $RAG_SOURCE_XLSX"
  "$PYTHON" -m rag.prepare_dataset --xlsx "$RAG_SOURCE_XLSX" --out "$(pwd)/data/training_data.csv"
fi

# ChromaDB has compatibility issues on Python 3.14; use FAISS by default
export RAG_VECTOR_STORE="${RAG_VECTOR_STORE:-faiss}"
echo "Running indexer (dataset: data/training_data.csv, store: $RAG_VECTOR_STORE)..."
"$PYTHON" -m rag.indexer --vector-store "$RAG_VECTOR_STORE"

echo "Done. Index saved under backend/rag_index/"
