# Push this project to your GitHub repo: https://github.com/Yash-povie/updated-Nanotoxi-test.git
# Run from project root. You must have git installed and be logged in (e.g. gh auth or git credential).

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/Yash-povie/updated-Nanotoxi-test.git"
Set-Location $PSScriptRoot

# Ensure we're in a git repo
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repo..."
    git init
}

# Add the remote if not already added (use 'origin' or replace with your remote name)
$remotes = git remote
if ($remotes -notmatch "origin") {
    git remote add origin $RepoUrl
    Write-Host "Added remote: origin -> $RepoUrl"
} else {
    $url = git remote get-url origin 2>$null
    if ($url -ne $RepoUrl) {
        git remote set-url origin $RepoUrl
        Write-Host "Set origin to: $RepoUrl"
    }
}

# Stage all files (respects .gitignore so .env won't be added)
git add -A
$status = git status --short
if (-not $status) {
    Write-Host "Nothing to commit (working tree clean)."
    exit 0
}

Write-Host "Staged files:"
git status --short

# Commit
$msg = "Unified backend: Api.py + PostgreSQL + RAG + deploy docs and scripts"
git commit -m $msg

# Push (use main or master depending on your default branch)
$branch = git branch --show-current
if (-not $branch) {
    git checkout -b main
    $branch = "main"
}
Write-Host "Pushing to origin/$branch..."
git push -u origin $branch

Write-Host "Done. Repo: $RepoUrl"
