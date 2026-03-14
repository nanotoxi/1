@echo off
cd /d "%~dp0"
if not exist .env (
    echo Create a .env file with DATABASE_URL=... and optionally JWT_SECRET=...
    echo Copy from env.example
)
echo Starting single backend (Api.py) on http://0.0.0.0:8000 ...
python Api.py
