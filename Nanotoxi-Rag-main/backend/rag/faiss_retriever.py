from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FaissHit:
    score: float
    id: str
    document: str
    metadata: dict[str, Any]

class FaissRetriever:
    def __init__(
        self,
        *,
        index_path: str | Path,
        metadata_path: str | Path,
        embedding_model_name: str,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model_name = embedding_model_name
        self._index = None
        self._meta = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from .config import HF_HOME_DIR

            self._model = SentenceTransformer(self.embedding_model_name, cache_folder=str(HF_HOME_DIR))
        if self._index is None or self._meta is None:
            # Load FAISS after torch/model initialization to avoid rare native crashes.
            self._index, self._meta = load_faiss_index(self.index_path, self.metadata_path)

    def search(self, query_text: str, *, top_k: int = 5) -> list[FaissHit]:
        import numpy as np

        self._ensure_loaded()
        q = self._model.encode([query_text], show_progress_bar=False).astype("float32")
        distances, indices = self._index.search(q, top_k)

        ids = self._meta.get("ids", [])
        docs = self._meta.get("documents", [])
        metas = self._meta.get("metadatas", [])

        hits: list[FaissHit] = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist(), strict=False):
            if idx < 0 or idx >= len(ids):
                continue
            hits.append(
                FaissHit(
                    score=float(-dist),
                    id=str(ids[idx]),
                    document=str(docs[idx]),
                    metadata=dict(metas[idx]) if idx < len(metas) else {},
                )
            )
        return hits


def load_faiss_index(index_path: str | Path, metadata_path: str | Path):
    import pickle

    import faiss  # type: ignore

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def query_faiss(
    query_text: str,
    *,
    index_path: str | Path,
    metadata_path: str | Path,
    embedding_model_name: str,
    top_k: int = 5,
) -> list[FaissHit]:
    retriever = FaissRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model_name=embedding_model_name,
    )
    return retriever.search(query_text, top_k=top_k)

