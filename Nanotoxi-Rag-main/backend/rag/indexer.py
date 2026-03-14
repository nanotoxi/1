"""
Phase 1: RAG Dataset Indexing (Open-Source Embeddings & Vector DB).

Loads the training dataset (CSV), converts rows to text representations,
generates local embeddings via sentence-transformers, and stores them
in ChromaDB (or FAISS). 100% local — no proprietary APIs.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DEFAULT_FEATURE_COLUMNS,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TRAINING_CSV_PATH,
    VECTOR_STORE_BACKEND,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the training dataset from CSV.
    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(csv_path or TRAINING_CSV_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training CSV not found: {path}. "
            "Set RAG_TRAINING_CSV or place training_data.csv in backend/data/"
        )
    df = pd.read_csv(path)
    logger.info("Loaded training data: %s rows, %s columns", len(df), len(df.columns))
    return df


def row_to_text(
    row: pd.Series,
    feature_columns: dict[str, str] | None = None,
    include_id: bool = True,
) -> str:
    """
    Convert a single row of features into a natural-language text representation
    suitable for embedding. Used for both indexing and later retrieval consistency.
    """
    cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    parts = []
    for key, col_name in cols.items():
        if key == "id" and not include_id:
            continue
        if col_name not in row.index:
            continue
        val = row[col_name]
        if pd.isna(val):
            continue
        if isinstance(val, float):
            val = round(val, 4) if key != "id" else int(val) if val == int(val) else val
        parts.append(f"{key.replace('_', ' ')}: {val}")
    return ". ".join(parts) if parts else str(row.to_dict())


def build_documents_and_metadata(
    df: pd.DataFrame,
    feature_columns: dict[str, str] | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Build document texts, metadata dicts, and document IDs from the dataframe.
    Returns (documents, metadatas, ids).
    """
    cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    id_col = cols.get("id", "id")
    if id_col not in df.columns:
        id_col = df.columns[0]
    documents = []
    metadatas = []
    ids = []
    for idx, row in df.iterrows():
        doc_id = str(row.get(id_col, idx))
        text = row_to_text(row, cols)
        documents.append(text)
        meta = {}
        for k, v in row.items():
            if pd.isna(v):
                meta[k] = ""
            else:
                v = v.item() if hasattr(v, "item") else v  # numpy/pandas scalar -> Python
                if isinstance(v, bool):
                    meta[k] = v
                elif isinstance(v, (int, float)):
                    meta[k] = int(v) if v == int(v) else float(v)
                else:
                    meta[k] = str(v)
        metadatas.append(meta)
        ids.append(doc_id)
    return documents, metadatas, ids


def get_embedding_model():
    """Lazy-load sentence-transformers model (local, no API)."""
    from sentence_transformers import SentenceTransformer
    # Force cache into backend/.cache/huggingface (see rag.config)
    from .config import HF_HOME_DIR
    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(HF_HOME_DIR))


def index_with_chromadb(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    persist_directory: str | Path | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model = None,
) -> Any:
    """
    Create embeddings and store in ChromaDB. Persists to disk.
    """
    import chromadb
    from chromadb.config import Settings

    persist_dir = str(persist_directory or CHROMA_PERSIST_DIR)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    if embedding_model is None:
        embedding_model = get_embedding_model()

    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Nanoparticle toxicity training samples"},
    )
    # Chroma expects list of lists for embeddings
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("ChromaDB index saved to %s (collection: %s)", persist_dir, collection_name)
    return client


def index_with_faiss(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    embedding_model = None,
) -> Any:
    """
    Create embeddings and store in FAISS index + pickle metadata.
    """
    import faiss
    import numpy as np

    if embedding_model is None:
        embedding_model = get_embedding_model()

    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    idx_path = Path(index_path or FAISS_INDEX_PATH)
    meta_path = Path(metadata_path or METADATA_PATH)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"ids": ids, "metadatas": metadatas, "documents": documents}, f)
    logger.info("FAISS index saved to %s, metadata to %s", idx_path, meta_path)
    return index


def run_indexing(
    csv_path: str | Path | None = None,
    feature_columns: dict[str, str] | None = None,
    vector_store: str | None = None,
) -> dict[str, Any]:
    """
    Full indexing pipeline: load CSV -> text -> embed -> store.
    Returns a summary dict (paths, counts, backend).
    """
    vector_store = vector_store or VECTOR_STORE_BACKEND
    df = load_training_data(csv_path)
    documents, metadatas, ids = build_documents_and_metadata(df, feature_columns)
    if not documents:
        raise ValueError("No documents produced from the dataset. Check CSV and column mapping.")

    model = get_embedding_model()
    if vector_store == "chromadb":
        index_with_chromadb(documents, metadatas, ids, embedding_model=model)
        return {
            "backend": "chromadb",
            "persist_dir": CHROMA_PERSIST_DIR,
            "collection": CHROMA_COLLECTION_NAME,
            "num_documents": len(documents),
            "embedding_model": EMBEDDING_MODEL,
        }
    elif vector_store == "faiss":
        index_with_faiss(documents, metadatas, ids, embedding_model=model)
        return {
            "backend": "faiss",
            "index_path": FAISS_INDEX_PATH,
            "metadata_path": METADATA_PATH,
            "num_documents": len(documents),
            "embedding_model": EMBEDDING_MODEL,
        }
    else:
        raise ValueError(f"Unknown vector_store: {vector_store}. Use 'chromadb' or 'faiss'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index training data for RAG (Phase 1)")
    parser.add_argument("--csv", default=None, help="Path to training CSV")
    parser.add_argument("--vector-store", choices=["chromadb", "faiss"], default=None)
    parser.add_argument("--columns-json", default=None, help="JSON map of feature name -> CSV column name")
    args = parser.parse_args()
    feature_cols = None
    if args.columns_json:
        feature_cols = json.loads(args.columns_json)
    result = run_indexing(
        csv_path=args.csv,
        feature_columns=feature_cols,
        vector_store=args.vector_store,
    )
    print(json.dumps(result, indent=2))
