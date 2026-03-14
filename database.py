# database.py
# PostgreSQL connection and table initialization
# Uses DATABASE_URL env variable (e.g. Railway connection URL)

import os
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor


def _load_dotenv_if_needed():
    """Load .env so DATABASE_URL is set (e.g. in uvicorn worker)."""
    if os.getenv("DATABASE_URL", "").strip():
        return
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        env_file = base / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file, override=True)
                break
            except ImportError:
                break


def _get_database_url():
    """Read DATABASE_URL when needed (so .env is always applied)."""
    _load_dotenv_if_needed()
    raw = os.getenv("DATABASE_URL", "").strip() or None
    if not raw:
        return None
    # If Railway injected an unresolved reference (e.g. ${{...}}postgresql://...postgresql://...),
    # use only the first valid postgresql:// URL.
    if "postgresql://" in raw:
        start = raw.find("postgresql://")
        rest = raw[start:]
        # Take up to the next "postgresql://" or end (avoids duplicated URLs)
        next_start = rest.find("postgresql://", 1)
        url = rest[:next_start] if next_start > 0 else rest
    else:
        url = raw
    url = url.strip()
    if not url:
        return None
    # Public Railway Postgres (rlwy.net / .railway.app) often needs sslmode; internal (postgres.railway.internal) does not
    if "sslmode" not in url.lower() and ".railway.internal" not in url and ("rlwy.net" in url or ".railway.app" in url):
        url = url + ("&" if "?" in url else "?") + "sslmode=require"
    return url


def get_connection():
    """Get a fresh PostgreSQL connection."""
    url = _get_database_url()
    if not url:
        raise RuntimeError("DATABASE_URL environment variable not set.")
    return psycopg2.connect(url, cursor_factory=RealDictCursor)


def init_db():
    """
    Create all required tables if they don't exist.
    Call this once on app startup.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Predictions table — stores every prediction request
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id                SERIAL PRIMARY KEY,
            request_id        VARCHAR(64) UNIQUE NOT NULL,
            username          VARCHAR(128),
            composition       VARCHAR(256),
            nanoparticle_size FLOAT,
            zeta_potential    FLOAT,
            morphology        VARCHAR(128),
            cell_type         VARCHAR(256),
            dosage_in_vitro   FLOAT,
            organic_inorganic VARCHAR(64),
            surface_chemistry VARCHAR(256),
            prediction        VARCHAR(32),
            confidence        FLOAT,
            explanation       TEXT,
            report_path       VARCHAR(512),
            timestamp         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Similar records table — stores retrieved NPs per prediction
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS similar_records (
            id                SERIAL PRIMARY KEY,
            prediction_id     VARCHAR(64) REFERENCES predictions(request_id),
            nanoparticle_name VARCHAR(256),
            composition       VARCHAR(256),
            outcome           VARCHAR(64),
            similarity_score  FLOAT
        );
        """
    )

    # Logs table — stores system events and errors
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id        SERIAL PRIMARY KEY,
            level     VARCHAR(16),
            message   TEXT,
            username  VARCHAR(128),
            endpoint  VARCHAR(128),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")


def save_prediction(
    request_id,
    username,
    query_params,
    prediction,
    confidence,
    similar_records,
    explanation=None,
    report_path=None,
):
    """Save a prediction + its similar records to the database."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions (
            request_id,
            username,
            composition,
            nanoparticle_size,
            zeta_potential,
            morphology,
            cell_type,
            dosage_in_vitro,
            organic_inorganic,
            surface_chemistry,
            prediction,
            confidence,
            explanation,
            report_path
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (request_id) DO NOTHING;
        """,
        (
            request_id,
            username,
            query_params.get("composition"),
            query_params.get("Nanoparticle size"),
            query_params.get("Zeto Potential"),
            query_params.get("Morphology"),
            query_params.get("Cell Type"),
            query_params.get("Dosage in vitro"),
            query_params.get("organic/inorganic"),
            query_params.get("Surface Chemistry"),
            prediction,
            confidence,
            explanation,
            report_path,
        ),
    )

    # Save similar records
    for r in similar_records:
        cur.execute(
            """
            INSERT INTO similar_records (
                prediction_id,
                nanoparticle_name,
                composition,
                outcome,
                similarity_score
            ) VALUES (%s,%s,%s,%s,%s);
            """,
            (
                request_id,
                r.get("nanoparticle name", r.get("nanoparticle_name", "")),
                r.get("composition", ""),
                r.get("toxic/non-toxic", r.get("outcome", "")),
                r.get("similarity_score", 0.0),
            ),
        )

    conn.commit()
    cur.close()
    conn.close()


def log_event(level: str, message: str, username: str = None, endpoint: str = None):
    """Write a log entry to the logs table."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO logs (level, message, username, endpoint)
            VALUES (%s,%s,%s,%s);
            """,
            (level, message, username, endpoint),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        # Never let logging crash the main app
        pass


def get_dashboard_stats() -> dict:
    """Aggregate stats for the admin dashboard."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total FROM predictions;")
    total = cur.fetchone()["total"]

    cur.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE prediction = 'Toxic';")
    toxic = cur.fetchone()["cnt"]

    cur.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE prediction = 'Non-Toxic';")
    non_toxic = cur.fetchone()["cnt"]

    cur.execute("SELECT AVG(confidence) AS avg_conf FROM predictions;")
    avg_conf = cur.fetchone()["avg_conf"]

    cur.execute(
        """
        SELECT DATE(timestamp) AS day, COUNT(*) AS count
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY day DESC
        LIMIT 30;
        """
    )
    daily = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT composition, COUNT(*) AS count
        FROM predictions
        GROUP BY composition
        ORDER BY count DESC
        LIMIT 10;
        """
    )
    top_compositions = [dict(r) for r in cur.fetchall()]

    cur.close()
    conn.close()

    return {
        "total_predictions": total,
        "toxic_count": toxic,
        "non_toxic_count": non_toxic,
        "avg_confidence": round(float(avg_conf), 3) if avg_conf else 0.0,
        "daily_predictions": daily,
        "top_compositions": top_compositions,
    }


def get_logs(limit: int = 100) -> list:
    """Fetch recent log entries."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT level, message, username, endpoint, timestamp
        FROM logs
        ORDER BY timestamp DESC
        LIMIT %s;
        """,
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows

