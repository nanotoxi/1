# Single backend runner - Nanotox AI (FastAPI + RAG + PostgreSQL)
# Run from project root. Uses .env for DATABASE_URL and JWT_SECRET.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Write-Host "Create a .env file (copy from env.example) with at least DATABASE_URL=..." -ForegroundColor Yellow
}

Write-Host "Starting single backend (Api.py) on http://0.0.0.0:8000 ..." -ForegroundColor Cyan
python Api.py
