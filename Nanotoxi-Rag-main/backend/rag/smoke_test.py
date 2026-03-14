from __future__ import annotations

import json
from pathlib import Path

from .config import EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH, TRAINING_CSV_PATH
from .faiss_retriever import query_faiss
from .prepare_dataset import DEFAULT_XLSX, prepare_from_excel
from .indexer import run_indexing
from .qa import answer_question


def main() -> None:
    xlsx = Path(DEFAULT_XLSX)
    if xlsx.is_file():
        prep = prepare_from_excel(xlsx_path=xlsx)
    else:
        prep = {"note": "Excel not found; using existing training_data.csv"}

    idx = run_indexing(csv_path=TRAINING_CSV_PATH, vector_store="faiss")

    hits = query_faiss(
        "What is the typical zeta potential and toxicity for Aluminum/Alumina nanoparticles?",
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL,
        top_k=5,
    )

    qa = answer_question(
        "What is the typical zeta potential and toxicity for Aluminum/Alumina nanoparticles?",
        top_k=5,
    )

    summary = {
        "prepare": prep,
        "index": idx,
        "qa_answer": qa["answer"],
        "query_hits": [
            {
                "score": h.score,
                "id": h.id,
                "source_sheet": h.metadata.get("source_sheet", ""),
                "composition": h.metadata.get("composition", h.metadata.get("Nanoparticle Category", "")),
                "toxicity": h.metadata.get("toxicity", h.metadata.get("Toxicity Classification", "")),
            }
            for h in hits
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

