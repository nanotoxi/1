"""
Run this from the project root to verify DB connection and one insert.
  python test_db_connection.py
"""
import os
import sys
from pathlib import Path

# Load .env from project root
root = Path(__file__).resolve().parent
env_path = root / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
        print(f"[test] Loaded .env from {env_path}")
    except ImportError:
        print("[test] python-dotenv not installed, using existing env")
else:
    print(f"[test] No .env at {env_path}")

url = os.getenv("DATABASE_URL", "").strip()
print(f"[test] DATABASE_URL set: {bool(url)}")
if not url:
    print("Set DATABASE_URL in .env or environment and try again.")
    sys.exit(1)

# Railway SSL
if "sslmode" not in url.lower() and ("rlwy.net" in url or "railway" in url.lower()):
    url = url + ("&" if "?" in url else "?") + "sslmode=require"
    print("[test] Added sslmode=require for Railway")

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("INSERT INTO logs (level, message, username, endpoint) VALUES (%s,%s,%s,%s) RETURNING id;", ("INFO", "test_db_connection.py", "test", "script"))
    row_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    print(f"[test] OK: inserted log row id={row_id}. Check Railway Data -> logs table.")
except Exception as e:
    print(f"[test] FAILED: {e}")
    sys.exit(1)
