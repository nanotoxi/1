from __future__ import annotations

import json

from .config import EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH
from .faiss_retriever import FaissRetriever


EVAL_QUERIES = [
    {
        "query": "What is the toxicity percentage for Aluminum/Alumina category?",
        "must_have": {"source_sheet": "Thresholds by Category", "composition": "Aluminum/Alumina"},
    },
    {
        "query": "Find records about Copper/Copper Oxide toxicity.",
        "must_have": {"composition": "Copper/Copper Oxide"},
    },
    {
        "query": "Show me nanoparticle data for Antimony Oxide.",
        "must_have": {"composition": "Antimony Oxide"},
    },
]


def _match(hit_meta: dict, must_have: dict) -> bool:
    for k, v in must_have.items():
        if str(hit_meta.get(k, "")).strip() != str(v).strip():
            return False
    return True


def main() -> None:
    retriever = FaissRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL,
    )

    results = []
    passed = 0
    for item in EVAL_QUERIES:
        q = item["query"]
        must = item["must_have"]
        hits = retriever.search(q, top_k=10)
        ok = any(_match(h.metadata, must) for h in hits)
        passed += int(ok)
        results.append(
            {
                "query": q,
                "pass": ok,
                "must_have": must,
                "top_hits": [
                    {
                        "score": h.score,
                        "source_sheet": h.metadata.get("source_sheet", ""),
                        "composition": h.metadata.get("composition", ""),
                        "toxicity": h.metadata.get("toxicity", ""),
                    }
                    for h in hits[:5]
                ],
            }
        )

    print(json.dumps({"passed": passed, "total": len(EVAL_QUERIES), "results": results}, indent=2))


if __name__ == "__main__":
    main()

