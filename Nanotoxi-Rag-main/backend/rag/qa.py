from __future__ import annotations

import re
from typing import Any

from .config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    LLM_BACKEND,
    LLM_MODEL,
    METADATA_PATH,
    OLLAMA_MODEL,
)
from .faiss_retriever import FaissRetriever, FaissHit
from .llm import get_llm


_RETRIEVER: FaissRetriever | None = None
_THRESHOLDS_SHEET_NAME = "Thresholds by Category"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _extract_category(question: str) -> str | None:
    """
    Best-effort extraction of category name from questions like:
      - What is the Toxic % for Aluminum/Alumina category?
      - ... for category 'Gadolinium Oxide' ...
    """
    q = (question or "").strip()
    if not q:
        return None

    m = re.search(r"category\s*['\"]([^'\"]+)['\"]", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"\bfor\s+(.+?)\s+category\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" '\"\t\r\n")

    return None


def _is_threshold_question(question: str) -> bool:
    q = _norm(question)
    return any(
        key in q
        for key in [
            "toxic %",
            "toxicity percentage",
            "toxic percent",
            "zeta_mv_median",
            "median zeta",
            "zeta potential median",
        ]
    )


def _wants_toxic_pct(question: str) -> bool:
    q = _norm(question)
    return ("toxic %" in q) or ("toxicity percentage" in q) or ("toxic percent" in q)


def _wants_zeta_median(question: str) -> bool:
    q = _norm(question)
    return ("zeta_mv_median" in q) or ("median zeta" in q) or ("zeta potential median" in q)


def _threshold_field_from_question(question: str) -> tuple[str, str] | None:
    """
    Map common numeric questions to a Thresholds-by-Category metadata field.
    Returns (field_key, units_hint).
    """
    q = _norm(question)

    if _wants_toxic_pct(question):
        return ("Toxic %", "%")
    if _wants_zeta_median(question):
        return ("Zeta_mV_median", "mV")

    # Size-related (hydro before generic size so "hydro_size_nm_median" doesn't match "size_nm_median")
    if "hydro_size_nm_median" in q or ("hydrodynamic" in q and "median" in q):
        return ("Hydro_size_nm_median", "nm")
    if "size_nm_median" in q or ("size" in q and "median" in q):
        return ("Size_nm_median", "nm")

    # Exposure
    if "exposure_hrs_median" in q or ("exposure" in q and "median" in q):
        return ("Exposure_hrs_median", "hrs")

    # Threshold medians
    if "lc50 threshold" in q and "median" in q:
        return ("LC50 Threshold (median)", "")
    if "ic50 threshold" in q and "median" in q:
        return ("IC50 Threshold (median)", "")

    # Value medians
    if "lc50" in q and "median" in q:
        return ("LC50_val_median", "")
    if ("ic50" in q or "ec50" in q) and "median" in q:
        return ("IC50_val_median", "")
    if "ld50" in q and "median" in q:
        return ("LD50_val_median", "")

    # Viability/dissolution etc.
    if ("viability" in q or "cell viability" in q) and "median" in q:
        return ("Viability_pct_median", "%")
    if "dissolution" in q and "median" in q:
        return ("Dissolution_median", "")

    return None


def _extract_numeric_from_text(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def _prioritize_threshold_hits(question: str, hits: list[FaissHit]) -> list[FaissHit]:
    """
    Reduce hallucinations for threshold-style questions by:
      - extracting category
      - preferring Thresholds-by-Category row for that category
      - filtering out other categories where possible
    """
    if not hits:
        return hits

    category = _extract_category(question)
    if not category:
        return hits

    cat_norm = _norm(category)

    def hit_comp(h: FaissHit) -> str:
        return str(h.metadata.get("composition") or h.metadata.get("Category") or h.metadata.get("Nanoparticle Category") or "")

    def hit_sheet(h: FaissHit) -> str:
        return str(h.metadata.get("source_sheet") or "")

    # First: keep only matching category hits (if any)
    matching = [h for h in hits if _norm(hit_comp(h)) == cat_norm]
    if not matching:
        return hits

    # Second: for threshold questions, ensure the thresholds row is first (if present)
    if _is_threshold_question(question):
        thresholds = [h for h in matching if _norm(hit_sheet(h)) == _norm(_THRESHOLDS_SHEET_NAME)]
        non_thresholds = [h for h in matching if _norm(hit_sheet(h)) != _norm(_THRESHOLDS_SHEET_NAME)]
        if thresholds:
            return thresholds + non_thresholds
    return matching


def get_retriever() -> FaissRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = FaissRetriever(
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATA_PATH,
            embedding_model_name=EMBEDDING_MODEL,
        )
    return _RETRIEVER


def build_prompt(question: str, hits: list[FaissHit], ensemble_prediction: str | None = None) -> str:
    context_lines = []
    for i, h in enumerate(hits, start=1):
        src = h.metadata.get("source_sheet", "")
        comp = h.metadata.get("composition", h.metadata.get("Nanoparticle Category", ""))
        tox = h.metadata.get("toxicity", h.metadata.get("Toxicity Classification", ""))
        context_lines.append(f"[{i}] source={src} composition={comp} toxicity={tox}\n{h.document}")

    context = "\n\n".join(context_lines) if context_lines else "(no context)"

    format_hint = ""
    if _is_threshold_question(question):
        # Helps with automated scoring and reduces rambling.
        format_hint = "Return ONLY the numeric value if the question asks for a single number.\n"

    alignment_directive = ""
    if ensemble_prediction and str(ensemble_prediction).strip():
        pred = str(ensemble_prediction).strip().upper()
        alignment_directive = (
            f"Ensemble ML Prediction: {pred}\n"
            "CRITICAL: Your explanation MUST align with the provided Ensemble ML Prediction. "
            "If the model predicts TOXIC, you cannot state the particle is safe or non-toxic, regardless of the retrieved context. "
            "If the model predicts SAFE, you cannot conclude the particle is highly toxic.\n"
        )
        if pred in ("TOXIC", "HIGHLY TOXIC"):
            alignment_directive += (
                "Your final conclusion must state that the particle is toxic (or high toxicity). Do not say it is safe or non-toxic.\n\n"
            )
        else:
            alignment_directive += (
                "Your final conclusion must state that the particle is safe or low/no toxicity. Do not conclude that it is highly toxic.\n\n"
            )

    return (
        "You are a scientific assistant. Answer the question using ONLY the provided context.\n"
        "If the context is insufficient, say what is missing.\n\n"
        f"{alignment_directive}"
        f"{format_hint}"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def answer_question(question: str, *, top_k: int = 5, ensemble_prediction: str | None = None) -> dict[str, Any]:
    retriever = get_retriever()
    hits = retriever.search(question, top_k=top_k)
    hits = _prioritize_threshold_hits(question, hits)

    # If this looks like a Thresholds-by-Category question, do a second-stage retrieval
    # targeted at the extracted category to reduce category mixups.
    category = _extract_category(question)
    if category and _is_threshold_question(question):
        has_threshold_row = any(
            _norm(str(h.metadata.get("source_sheet", ""))) == _norm(_THRESHOLDS_SHEET_NAME) for h in hits
        )
        if not has_threshold_row:
            boosted_query = f"{category} {_THRESHOLDS_SHEET_NAME} Toxic % Zeta_mV_median"
            extra = retriever.search(boosted_query, top_k=max(25, top_k * 6))
            # merge (dedupe by id)
            seen = set()
            merged: list[FaissHit] = []
            for h in hits + extra:
                if h.id in seen:
                    continue
                seen.add(h.id)
                merged.append(h)
            hits = _prioritize_threshold_hits(question, merged)

    # Deterministic answers for threshold-style numeric questions.
    # This dramatically improves accuracy vs letting the LLM guess a number.
    category = _extract_category(question)
    threshold_field = _threshold_field_from_question(question)
    if category and threshold_field is not None:
        field_key, _units = threshold_field
        cat_norm = _norm(category)
        threshold_hit = next(
            (
                h
                for h in hits
                if _norm(str(h.metadata.get("source_sheet", ""))) == _norm(_THRESHOLDS_SHEET_NAME)
                and _norm(str(h.metadata.get("composition") or h.metadata.get("Category") or "")) == cat_norm
            ),
            None,
        )
        if threshold_hit is not None:
            md = threshold_hit.metadata
            raw = md.get(field_key, "")
            # Fallbacks for legacy fields that might be present as derived columns
            if raw == "" and field_key == "Zeta_mV_median":
                raw = md.get("zeta_potential_mv", md.get("zeta_potential", ""))

            value = raw if isinstance(raw, (int, float)) else _extract_numeric_from_text(str(raw))

            if value is not None:
                return {
                    "question": question,
                    "llm_backend": LLM_BACKEND,
                    "llm_model": (OLLAMA_MODEL if LLM_BACKEND.lower() == "ollama" else LLM_MODEL),
                    "top_k": top_k,
                    "answer": str(round(float(value), 4)),
                    "answer_source": "deterministic_threshold_lookup",
                    "threshold_field": field_key,
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

    prompt = build_prompt(question, hits, ensemble_prediction=ensemble_prediction)
    llm = get_llm()
    result = llm.generate(prompt)
    return {
        "question": question,
        "llm_backend": LLM_BACKEND,
        "llm_model": (OLLAMA_MODEL if LLM_BACKEND.lower() == "ollama" else LLM_MODEL),
        "top_k": top_k,
        "answer": result.text,
        "answer_source": "llm",
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

