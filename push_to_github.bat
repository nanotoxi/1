@echo off
cd /d "%~dp0"
set REPO_URL=https://github.com/Yash-povie/updated-Nanotoxi-test.git

if not exist .git (
    echo Initializing git repo...
    git init
)

git remote 2>nul | findstr /x "origin" >nul || git remote add origin %REPO_URL%
git remote set-url origin %REPO_URL%

git add -A
git status --short
git commit -m "Unified backend: Api.py + PostgreSQL + RAG + deploy docs and scripts"
git branch --show-current 2>nul || git checkout -b main
git push -u origin main
echo Done. Repo: %REPO_URL%
