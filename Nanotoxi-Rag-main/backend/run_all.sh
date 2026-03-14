#!/usr/bin/env bash
set -euo pipefail

# One command: venv -> deps -> prepare (both Excel sheets) -> index (FAISS) -> start API

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  echo "Creating venv at backend/.venv"
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Preparing + indexing..."
chmod +x ./run_rag_index.sh
./run_rag_index.sh

echo "Starting API on http://localhost:5000"
echo "Endpoints:"
echo "  GET  /health"
echo "  POST /rag/search  {\"query\":\"...\",\"top_k\":5}"
echo "  POST /rag/answer  {\"question\":\"...\",\"top_k\":5}"

# Gunicorn production-ish server (1 worker; increase for higher throughput)
exec gunicorn -w "${WEB_CONCURRENCY:-1}" -b "0.0.0.0:${PORT:-5000}" "app:app"

