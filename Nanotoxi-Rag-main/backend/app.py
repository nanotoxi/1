from __future__ import annotations

import os
from flask import Flask, jsonify, request

from rag.config import LLM_BACKEND, LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL, VECTOR_STORE_BACKEND
from rag.qa import answer_question, get_retriever


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/rag/info")
    def rag_info():
        llm_model = OLLAMA_MODEL if LLM_BACKEND.lower() == "ollama" else LLM_MODEL
        return jsonify(
            {
                "vector_store": VECTOR_STORE_BACKEND,
                "llm_backend": LLM_BACKEND,
                "llm_model": llm_model,
                "ollama_url": OLLAMA_BASE_URL if LLM_BACKEND.lower() == "ollama" else "",
                "note": "RAG = FAISS retrieval + local LLM (HF or Ollama).",
            }
        )

    @app.post("/rag/search")
    def rag_search():
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        top_k = int(body.get("top_k") or 5)
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400

        hits = get_retriever().search(query, top_k=top_k)
        return jsonify(
            {
                "query": query,
                "top_k": top_k,
                "hits": [
                    {
                        "score": h.score,
                        "id": h.id,
                        "document": h.document,
                        "metadata": h.metadata,
                    }
                    for h in hits
                ],
            }
        )

    @app.post("/rag/answer")
    def rag_answer():
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or body.get("query") or "").strip()
        top_k = int(body.get("top_k") or 5)
        ensemble_prediction = (body.get("ensemble_prediction") or "").strip() or None
        if not question:
            return jsonify({"error": "Missing 'question' (or 'query')"}), 400
        return jsonify(answer_question(question, top_k=top_k, ensemble_prediction=ensemble_prediction))

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

