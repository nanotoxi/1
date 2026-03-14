# api.py
# Secure FastAPI endpoints for nanoparticle cytotoxicity prediction
# Auth: JWT Bearer tokens

import os
import sys
from pathlib import Path

# Load .env from project root first so DATABASE_URL is set before any other code runs
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=True)
    except ImportError:
        pass
import uuid
import datetime
from typing import Optional, List, Any

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
import jwt

# Ensure local modules (e.g., database.py) are importable when running directly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from database import (
    init_db,
    save_prediction,
    get_dashboard_stats,
    get_logs as db_get_logs,
    log_event,
)

# Add Nanotoxi-RAG backend to PYTHONPATH so we can import `rag.*`
PROJECT_ROOT = Path(__file__).resolve().parent
RAG_BACKEND_ROOT = PROJECT_ROOT / "Nanotoxi-Rag-main" / "backend"
if RAG_BACKEND_ROOT.exists():
    sys.path.append(str(RAG_BACKEND_ROOT))

try:
    from rag.config import (
        VECTOR_STORE_BACKEND,
        LLM_BACKEND,
        LLM_MODEL,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
    )
    from rag.qa import answer_question, get_retriever
except Exception:
    # RAG is optional – endpoints will raise if used without a proper setup
    VECTOR_STORE_BACKEND = "unknown"
    LLM_BACKEND = "unknown"
    LLM_MODEL = "unknown"
    OLLAMA_BASE_URL = ""
    OLLAMA_MODEL = ""

# ── Config ────────────────────────────────────────────────────────
SECRET_KEY   = os.getenv("JWT_SECRET", "project-titan-secret-change-in-prod")
ALGORITHM    = "HS256"
TOKEN_EXPIRY = 24  # hours

app = FastAPI(title="Nanotox AI API", version="1.0.0", description="Single backend: predictions, RAG, admin.")
bearer = HTTPBearer()

# CORS so frontend (local or deployed) can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """
    Initialize PostgreSQL tables and warm up RAG retriever.
    """
    db_url_set = bool(os.getenv("DATABASE_URL", "").strip())
    print(f"[startup] DATABASE_URL set: {db_url_set}")
    if not db_url_set:
        print("[startup] WARNING: Set DATABASE_URL (e.g. in Railway Variables for cloud) or predictions/logs will not be saved.")
    try:
        init_db()
        print("[startup] Database initialized successfully.")
    except Exception as e:
        print(f"[startup] Failed to init DB: {e}")
    # Warm up FAISS retriever (optional)
    try:
        get_retriever()
    except Exception as e:
        print(f"[startup] Failed to warm RAG retriever: {e}")

# ── Pydantic Models ───────────────────────────────────────────────
class NanoparticleInput(BaseModel):
    composition:        str
    nanoparticle_size:  Optional[float] = None
    zeta_potential:     Optional[float] = None
    morphology:         Optional[str]   = None
    cell_type:          Optional[str]   = None
    dosage_in_vitro:    Optional[float] = None
    organic_inorganic:  Optional[str]   = "inorganic"
    surface_chemistry:  Optional[str]   = None
    top_k:              Optional[int]   = 5
    use_rag:            Optional[bool]  = False  # toggle RAG explanation
    generate_report:    Optional[bool]  = False  # toggle PDF report

class TokenRequest(BaseModel):
    username: str
    password: str

class PredictionResponse(BaseModel):
    request_id:      str
    prediction:      str
    confidence:      float
    similar_records: list
    explanation:     Optional[str]
    report_url:      Optional[str]
    timestamp:       str


class RagSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class RagHit(BaseModel):
    score: float
    id: str
    document: str
    metadata: dict


class RagSearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[RagHit]


class RagAnswerRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    top_k: Optional[int] = 5
    ensemble_prediction: Optional[str] = None


class RagAnswerResponse(BaseModel):
    question: str
    llm_backend: str
    llm_model: str
    top_k: int
    answer: str
    answer_source: str
    hits: List[dict]


def _similar_records_from_rag_query(query: str, top_k: int = 5) -> list[dict]:
    """
    Use the Nanotoxi-RAG FAISS retriever to build similar nanoparticle records
    compatible with the existing dashboard/database schema.
    """
    if LLM_BACKEND == "unknown":
        return []
    try:
        hits = get_retriever().search(query, top_k=top_k)
    except Exception:
        return []

    records: list[dict] = []
    for h in hits:
        md = h.metadata or {}
        name = (
            md.get("nanoparticle_name")
            or md.get("Nanoparticle Name")
            or md.get("composition")
            or ""
        )
        composition = md.get("composition", "")
        toxicity = md.get("toxicity") or md.get("Toxicity Classification") or ""
        records.append(
            {
                "nanoparticle name": name,
                "composition": composition,
                "toxic/non-toxic": toxicity,
                "similarity_score": float(h.score),
            }
        )
    return records

# ── Dummy user store (replace with DB in Task 4) ──────────────────
USERS = {
    "admin": {
        "password": "titan2024",
        "role": "admin",
    },
    "researcher1": {
        "password": "nano@123",
        "role": "customer",
    },
}


# ── JWT helpers ───────────────────────────────────────────────────
def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRY),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_admin(username: str = Depends(verify_token)) -> str:
    """
    Dependency that ensures the current user is an admin.
    """
    user = USERS.get(username)
    role = user.get("role") if user else None
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return username

# ── Dummy predictor (swap with real model in Task 4) ──────────────
def run_prediction(params: dict) -> dict:
    import random
    random.seed(sum(ord(c) for c in str(params)))
    label = random.choice(["Toxic", "Non-Toxic"])
    confidence = round(random.uniform(0.65, 0.97), 2)
    return {"label": label, "confidence": confidence}

# ── Routes ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "Project Titan API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}

# ── Auth ──────────────────────────────────────────────────────────
@app.post("/auth/token")
def login(body: TokenRequest):
    """
    Exchange username + password for a JWT token.

    Example:
        POST /auth/token
        { "username": "admin", "password": "titan2024" }
    """
    user = USERS.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    role = user.get("role", "customer")
    token = create_token(body.username, role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": TOKEN_EXPIRY,
        "role": role,
    }

# ── Prediction ────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(
    body: NanoparticleInput,
    username: str = Depends(verify_token)
):
    """
    Submit nanoparticle parameters and receive a cytotoxicity prediction.
    Requires: Authorization: Bearer <token>

    Optional flags:
        use_rag=true        → includes AI explanation (requires LLM)
        generate_report=true → generates downloadable PDF report
    """
    import traceback
    print(f"[predict] request received for user={username}", flush=True)
    request_id = str(uuid.uuid4())

    # Map input to internal format
    query_params = {
        "composition":       body.composition,
        "Nanoparticle size": body.nanoparticle_size,
        "Zeto Potential":    body.zeta_potential,
        "Morphology":        body.morphology,
        "Cell Type":         body.cell_type,
        "Dosage in vitro":   body.dosage_in_vitro,
        "organic/inorganic": body.organic_inorganic,
        "Surface Chemistry": body.surface_chemistry,
    }

    # Run prediction (currently dummy logic)
    prediction = run_prediction(query_params)

    # Retrieve similar nanoparticles using RAG retriever (FAISS)
    rag_query = (
        f"Nanoparticle with composition={body.composition}, "
        f"size={body.nanoparticle_size}, zeta_potential={body.zeta_potential}, "
        f"morphology={body.morphology}, cell_type={body.cell_type}, "
        f"dosage_in_vitro={body.dosage_in_vitro}"
    )
    similar_records = _similar_records_from_rag_query(
        rag_query, top_k=body.top_k or 5
    )

    # RAG explanation (optional) – uses unified Nanotoxi-RAG backend
    explanation = None
    if body.use_rag:
        try:
            question = (
                f"Explain the cytotoxicity for nanoparticle with composition={body.composition}, "
                f"size={body.nanoparticle_size}, zeta_potential={body.zeta_potential}, "
                f"morphology={body.morphology}, cell_type={body.cell_type}, "
                f"dosage_in_vitro={body.dosage_in_vitro}."
            )
            rag_result = answer_question(
                question,
                top_k=body.top_k or 5,
                ensemble_prediction=prediction["label"],
            )
            explanation = rag_result.get("answer")
        except Exception as e:
            explanation = f"Explanation unavailable: {str(e)}"

    # Report generation (optional)
    report_url = None
    report_path = None
    if body.generate_report:
        try:
            from report_generator import generate_report
            reports_dir = PROJECT_ROOT / "reports"
            reports_dir.mkdir(exist_ok=True)
            report_path = reports_dir / f"report_{request_id}.pdf"
            generate_report(query_params, similar_records, output_path=str(report_path))
            report_url = f"/report/{request_id}"
        except Exception:
            report_url = None

    # Persist to PostgreSQL
    print(f"[predict] attempting DB save request_id={request_id}", flush=True)
    try:
        save_prediction(
            request_id=request_id,
            username=username,
            query_params=query_params,
            prediction=prediction["label"],
            confidence=prediction["confidence"],
            similar_records=similar_records,
            explanation=explanation,
            report_path=str(report_path) if report_path else None,
        )
        log_event(
            level="INFO",
            message=f"Prediction created: {request_id}",
            username=username,
            endpoint="/predict",
        )
        print(f"[predict] Saved to DB: {request_id}", flush=True)
    except Exception as e:
        print(f"[predict] Failed to save to DB: {e}", flush=True)
        traceback.print_exc()
        try:
            log_event(
                level="ERROR",
                message=f"Failed to save prediction {request_id}: {e}",
                username=username,
                endpoint="/predict",
            )
        except Exception:
            pass

    return PredictionResponse(
        request_id      = request_id,
        prediction      = prediction["label"],
        confidence      = prediction["confidence"],
        similar_records = similar_records,
        explanation     = explanation,
        report_url      = report_url,
        timestamp       = datetime.datetime.utcnow().isoformat()
    )

# ── Report download ───────────────────────────────────────────────
@app.get("/report/{report_id}")
def download_report(
    report_id: str,
    username: str = Depends(verify_token)
):
    """
    Download a previously generated PDF report by ID.
    Requires: Authorization: Bearer <token>
    """
    path = PROJECT_ROOT / "reports" / f"report_{report_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=f"cytotoxicity_report_{report_id}.pdf")

# ── Dataset info (for dashboard — Task 2) ─────────────────────────
@app.get("/dataset/info")
def dataset_info(username: str = Depends(require_admin)):
    """
    Returns dataset statistics for the admin dashboard.
    Backed by PostgreSQL `predictions` table.
    """
    try:
        stats = get_dashboard_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {e}")

# ── Prediction logs (for dashboard — Task 2) ──────────────────────
@app.get("/logs")
def get_logs(username: str = Depends(require_admin)):
    """
    Returns prediction logs for the admin dashboard from PostgreSQL.
    """
    try:
        return {"logs": db_get_logs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {e}")


# ── RAG endpoints (unified Nanotoxi-RAG backend) ──────────────────
@app.get("/rag/info")
def rag_info():
    """
    Inspect RAG backend configuration (vector store + LLM).
    Public, no auth required – useful for quick checks.
    """
    if LLM_BACKEND == "unknown":
        raise HTTPException(status_code=503, detail="RAG backend not configured")
    llm_model = OLLAMA_MODEL if LLM_BACKEND.lower() == "ollama" else LLM_MODEL
    return {
        "vector_store": VECTOR_STORE_BACKEND,
        "llm_backend": LLM_BACKEND,
        "llm_model": llm_model,
        "ollama_url": OLLAMA_BASE_URL if LLM_BACKEND.lower() == "ollama" else "",
        "note": "RAG = FAISS retrieval + local LLM (HF or Ollama).",
    }


@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(
    body: RagSearchRequest,
    username: str = Depends(verify_token),
):
    """
    Low-level RAG retrieval endpoint (no LLM).
    """
    if LLM_BACKEND == "unknown":
        raise HTTPException(status_code=503, detail="RAG backend not configured")
    query = (body.query or "").strip()
    top_k = body.top_k or 5
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'")
    try:
        hits = get_retriever().search(query, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG search failed: {e}")
    mapped_hits: List[RagHit] = [
        RagHit(score=h.score, id=h.id, document=h.document, metadata=h.metadata)
        for h in hits
    ]
    return RagSearchResponse(query=query, top_k=top_k, hits=mapped_hits)


@app.post("/rag/answer", response_model=RagAnswerResponse)
def rag_answer(
    body: RagAnswerRequest,
    username: str = Depends(verify_token),
):
    """
    Full RAG question‑answering endpoint using Nanotoxi-RAG pipeline.
    """
    if LLM_BACKEND == "unknown":
        raise HTTPException(status_code=503, detail="RAG backend not configured")
    question = (body.question or body.query or "").strip()
    top_k = body.top_k or 5
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' (or 'query')")
    try:
        result: dict[str, Any] = answer_question(
            question,
            top_k=top_k,
            ensemble_prediction=body.ensemble_prediction,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG answer failed: {e}")
    return RagAnswerResponse(
        question=result.get("question", question),
        llm_backend=result.get("llm_backend", LLM_BACKEND),
        llm_model=result.get("llm_model", LLM_MODEL),
        top_k=result.get("top_k", top_k),
        answer=result.get("answer", ""),
        answer_source=result.get("answer_source", "llm"),
        hits=result.get("hits", []),
    )


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # reload=False so the same process that loaded .env handles requests (avoids worker without DATABASE_URL)
    uvicorn.run("Api:app", host="0.0.0.0", port=8000, reload=False)