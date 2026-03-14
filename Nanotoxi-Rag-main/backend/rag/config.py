"""
RAG configuration: paths, model names, and vector store settings.
All paths are relative to the backend directory unless absolute.
"""
import os
from pathlib import Path

# Base paths (backend root)
BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_INDEX_DIR = BACKEND_ROOT / "rag_index"

# Hugging Face cache (keep inside project, avoids ~/.cache permission issues)
HF_HOME_DIR = BACKEND_ROOT / ".cache" / "huggingface"
HF_HUB_DIR = HF_HOME_DIR / "hub"
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_DIR))

# Training dataset
TRAINING_CSV_PATH = os.environ.get(
    "RAG_TRAINING_CSV",
    str(DEFAULT_DATA_DIR / "training_data.csv"),
)

# Embedding model (sentence-transformers, runs locally)
EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
# Alternative: "BAAI/bge-small-en-v1.5"

# Local LLM model
# Default to Ollama backend with Llama 3.2 1B Instruct, override via env if needed.
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "ollama")  # "hf" or "ollama"

LLM_MODEL = os.environ.get(
    "RAG_LLM_MODEL",
    "llama3.2:1b-instruct",
)
LLM_MAX_NEW_TOKENS = int(os.environ.get("RAG_LLM_MAX_NEW_TOKENS", "192"))
LLM_TEMPERATURE = float(os.environ.get("RAG_LLM_TEMPERATURE", "0.0"))

# Ollama (local)
OLLAMA_BASE_URL = os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "llama3.2:1b-instruct")

# Vector store
# NOTE: On Python 3.14, ChromaDB currently has compatibility issues with Pydantic v1,
# so FAISS is a safer default unless explicitly overridden.
VECTOR_STORE_BACKEND = os.environ.get("RAG_VECTOR_STORE", "faiss")  # or "chromadb"
CHROMA_PERSIST_DIR = os.environ.get(
    "RAG_CHROMA_DIR",
    str(DEFAULT_INDEX_DIR / "chroma_db"),
)
FAISS_INDEX_PATH = os.environ.get(
    "RAG_FAISS_INDEX",
    str(DEFAULT_INDEX_DIR / "faiss_index.bin"),
)
METADATA_PATH = str(DEFAULT_INDEX_DIR / "metadata.pkl")  # for FAISS: store ids + metadata

# Chroma collection name
CHROMA_COLLECTION_NAME = "nanoparticle_toxicity"

# Default CSV column mapping (override via env or config if your CSV differs)
# Keys: internal name -> CSV column name
DEFAULT_FEATURE_COLUMNS = {
    "id": "id",           # unique row/sample id
    "size": "size",       # e.g. size_nm or diameter_nm
    "zeta_potential": "zeta_potential",
    "concentration": "concentration",
    "composition": "composition",
    "coating": "coating",
    "toxicity": "toxicity",  # or viability, label, class, etc.
    "source_sheet": "source_sheet",
    "nanoparticle_name": "Nanoparticle Name",
    "morphology": "Morphology",
    "cell_viability": "Cell Viability %",
    "lc50": "LC50 Value",
    "ic50_ec50": "IC50/EC 50  value",
    "ld50": "LD50 Value",
    "exposure_time": "Exposure Time (in hrs)",
    "target": "Target organism/cells",
    "reference": "Reference Paper/s",
}
# If your CSV uses different names, set RAG_FEATURE_COLUMNS_JSON or edit this dict.
