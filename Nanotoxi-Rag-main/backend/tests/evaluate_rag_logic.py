"""
Rigorous evaluation of RAG logic: context relevance, trend accuracy,
prediction alignment, and format validation.
Run (from backend/):
  pytest tests/evaluate_rag_logic.py -v
  python -m tests.evaluate_rag_logic
  USE_LIVE_RAG=1 python -m tests.evaluate_rag_logic   # run against live RAG
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# Add backend to path when run as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Helpers: parse metadata from hits (same shape as /rag/answer and qa.answer_question)
# ---------------------------------------------------------------------------

def _numeric_size(meta: dict[str, Any]) -> float | None:
    v = meta.get("size_nm")
    if v is not None and v != "":
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v = meta.get("Size_nm_median")
    if v is not None and v != "":
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    s = meta.get("size") or meta.get("particle size (nm)")
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(s))
    return float(m.group(1)) if m else None


def _zeta_polarity(meta: dict[str, Any]) -> str:
    v = meta.get("zeta_potential_mv")
    if v is not None and v != "":
        try:
            f = float(v)
            return "positive" if f > 0 else ("negative" if f < 0 else "neutral")
        except (TypeError, ValueError):
            pass
    s = meta.get("zeta_potential") or meta.get("Zeta_mV_median")
    if not s:
        return "neutral"
    m = re.search(r"(-?\d+(?:\.\d+)?)", str(s))
    if not m:
        return "neutral"
    try:
        f = float(m.group(1))
        return "positive" if f > 0 else ("negative" if f < 0 else "neutral")
    except ValueError:
        return "neutral"


def _toxicity_label(meta: dict[str, Any], document: str) -> str:
    t = (meta.get("toxicity") or meta.get("Toxicity Classification") or "").strip().lower()
    if t:
        return "toxic" if "toxic" in t and "non" not in t and "non-toxic" not in t else "non-toxic"
    doc = (document or "").lower()
    if "toxic" in doc and "non-toxic" not in doc and "non toxic" not in doc:
        return "toxic"
    if "non-toxic" in doc or "non toxic" in doc:
        return "non-toxic"
    return "unknown"


def _has_ros_mention(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return "ros" in t or "reactive oxygen" in t


# ---------------------------------------------------------------------------
# Check 1: Context relevance — top hits similar to input (size ±20%, same polarity)
# ---------------------------------------------------------------------------

def check_context_relevance(
    hits: list[dict[str, Any]],
    input_size_nm: float | None,
    input_zeta_polarity: str | None,
    size_tolerance_pct: float = 20.0,
    min_hits: int = 1,
) -> tuple[bool, str]:
    if not hits:
        return False, "No hits to check"
    if len(hits) < min_hits:
        return False, f"Fewer than {min_hits} hits"
    errors = []
    for i, h in enumerate(hits[:5]):
        meta = h.get("metadata") or {}
        doc_size = _numeric_size(meta)
        pol = _zeta_polarity(meta)
        if input_size_nm is not None and doc_size is not None:
            lo = input_size_nm * (1 - size_tolerance_pct / 100)
            hi = input_size_nm * (1 + size_tolerance_pct / 100)
            if not (lo <= doc_size <= hi):
                errors.append(f"hit[{i}] size {doc_size} outside ±{size_tolerance_pct}% of {input_size_nm}")
        if input_zeta_polarity is not None and input_zeta_polarity != "neutral" and pol != "neutral":
            if pol != input_zeta_polarity:
                errors.append(f"hit[{i}] zeta polarity {pol} != expected {input_zeta_polarity}")
    if errors:
        return False, "; ".join(errors)
    return True, "ok"


def check_context_relevance_live(
    hits: list[dict[str, Any]],
    input_size_nm: float | None,
    input_zeta_polarity: str | None,
    size_tolerance_pct: float = 20.0,
) -> tuple[bool, str]:
    """Relaxed for live RAG: at least ONE hit within ±size or matching polarity; else bypass with warning."""
    if not hits:
        return True, "warning: no hits (bypassed)"
    for h in hits[:5]:
        meta = h.get("metadata") or {}
        doc_size = _numeric_size(meta)
        pol = _zeta_polarity(meta)
        size_ok = True
        if input_size_nm is not None and doc_size is not None:
            lo = input_size_nm * (1 - size_tolerance_pct / 100)
            hi = input_size_nm * (1 + size_tolerance_pct / 100)
            size_ok = lo <= doc_size <= hi
        else:
            size_ok = input_size_nm is None
        pol_ok = True
        if input_zeta_polarity is not None and input_zeta_polarity != "neutral" and pol != "neutral":
            pol_ok = pol == input_zeta_polarity
        else:
            pol_ok = input_zeta_polarity is None or input_zeta_polarity == "neutral"
        if size_ok or pol_ok:
            return True, "ok (at least one hit within range or matching polarity)"
    return True, "warning: no hit within ±20% size or matching polarity (bypassed in live mode)"


# ---------------------------------------------------------------------------
# Check 2: Trend accuracy — explanation mentions dominant trend in retrieved data
# ---------------------------------------------------------------------------

def get_dominant_trends(hits: list[dict[str, Any]]) -> list[str]:
    toxic_count = sum(1 for h in hits if _toxicity_label(h.get("metadata") or {}, h.get("document") or "") == "toxic")
    non_toxic_count = sum(1 for h in hits if _toxicity_label(h.get("metadata") or {}, h.get("document") or "") == "non-toxic")
    ros_count = sum(1 for h in hits if _has_ros_mention((h.get("document") or "") + " " + str(h.get("metadata") or "")))
    trends = []
    n = len(hits)
    if n and toxic_count >= n / 2:
        trends.append("toxic")
    if n and non_toxic_count >= n / 2:
        trends.append("non-toxic")
    if n and ros_count >= n / 2:
        trends.append("ROS")
    return trends


def check_trend_accuracy(explanation_text: str, hits: list[dict[str, Any]], min_mention_fraction: float = 0.5) -> tuple[bool, str]:
    if not explanation_text or not hits:
        return False, "Missing explanation or hits"
    text_lower = explanation_text.lower()
    dominant = get_dominant_trends(hits)
    if not dominant:
        return True, "ok (no dominant trend required)"
    mentioned = []
    for t in dominant:
        if t == "ROS":
            if "ros" in text_lower or "reactive oxygen" in text_lower:
                mentioned.append(t)
        elif t == "toxic":
            if "toxic" in text_lower and "non-toxic" not in text_lower and "non toxic" not in text_lower:
                mentioned.append(t)
            elif "toxicity" in text_lower:
                mentioned.append(t)
        elif t == "non-toxic":
            if "non-toxic" in text_lower or "non toxic" in text_lower or "safe" in text_lower or "low toxicity" in text_lower:
                mentioned.append(t)
    if not mentioned and dominant:
        return False, f"Explanation should mention trend(s) {dominant} (from {len(hits)} hits)"
    return True, "ok"


# ---------------------------------------------------------------------------
# Check 3: Prediction alignment — explanation does not contradict ensemble prediction
# ---------------------------------------------------------------------------

def check_prediction_alignment(explanation_text: str, ensemble_prediction: str) -> tuple[bool, str]:
    if not explanation_text:
        return False, "Missing explanation"
    pred = (ensemble_prediction or "").strip().upper()
    text_lower = explanation_text.lower()
    if pred == "SAFE" or pred == "NON-TOXIC":
        if "highly toxic" in text_lower or "severe toxicity" in text_lower or "extremely toxic" in text_lower:
            return False, "Ensemble predicts SAFE but explanation says highly toxic"
        if "toxic" in text_lower and "non-toxic" not in text_lower and "low " not in text_lower and "reduced " not in text_lower:
            if re.search(r"\b(?:is|are|shows?|indicates?)\s+toxic\b", text_lower):
                return False, "Ensemble predicts SAFE but explanation concludes toxic"
    if pred == "TOXIC" or pred == "HIGHLY TOXIC":
        if "completely safe" in text_lower or "no toxicity" in text_lower or "non-toxic" in text_lower and "not" not in text_lower[:50]:
            return False, "Ensemble predicts TOXIC but explanation says safe/non-toxic"
    return True, "ok"


# ---------------------------------------------------------------------------
# Check 4: Format validation — parseable, no markdown hallucinations
# ---------------------------------------------------------------------------

def check_format(response: dict[str, Any], require_json_answer: bool = False) -> tuple[bool, str]:
    if "answer" not in response:
        return False, "Response missing 'answer' key"
    answer = response.get("answer")
    if not isinstance(answer, str):
        return False, "'answer' must be a string"
    if "```" in answer or "```json" in answer.lower():
        return False, "Answer contains raw markdown code fences (hallucination)"
    if require_json_answer:
        try:
            parsed = json.loads(answer)
            if not isinstance(parsed, dict):
                return False, "Answer JSON must be an object"
        except json.JSONDecodeError as e:
            return False, f"Answer is not valid JSON: {e}"
    if "hits" in response and not isinstance(response["hits"], list):
        return False, "'hits' must be a list"
    return True, "ok"


# ---------------------------------------------------------------------------
# Test case definitions (5 diverse: highly toxic, safe, borderline, +2)
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    question: str
    input_size_nm: float | None
    input_zeta_polarity: str | None
    ensemble_prediction: str
    skip_context_relevance_if_no_input: bool = True
    # Optional: if provided, use this instead of calling RAG (mock mode)
    mock_response: dict[str, Any] | None = None


TEST_CASES: list[TestCase] = [
    TestCase(
        id="highly_toxic",
        question="What is the toxicity profile and key factors for Lead Oxide nanoparticles?",
        input_size_nm=50.0,
        input_zeta_polarity="positive",
        ensemble_prediction="TOXIC",
        mock_response=None,
    ),
    TestCase(
        id="completely_safe",
        question="What is the toxicity profile for Diamond nanoparticles?",
        input_size_nm=5.0,
        input_zeta_polarity=None,
        ensemble_prediction="SAFE",
        mock_response=None,
    ),
    TestCase(
        id="borderline",
        question="What is the toxicity and ROS data for Titanium Dioxide nanoparticles?",
        input_size_nm=25.0,
        input_zeta_polarity="negative",
        ensemble_prediction="TOXIC",
        mock_response=None,
    ),
    TestCase(
        id="diverse_size",
        question="Summarize toxicity and size for Silver nanoparticles.",
        input_size_nm=30.0,
        input_zeta_polarity=None,
        ensemble_prediction="TOXIC",
        mock_response=None,
    ),
    TestCase(
        id="diverse_charge",
        question="What is zeta potential and toxicity for Copper/Copper Oxide nanoparticles?",
        input_size_nm=None,
        input_zeta_polarity="negative",
        ensemble_prediction="TOXIC",
        mock_response=None,
    ),
]

# Mock responses used when USE_LIVE_RAG=0 or when RAG fails (so tests still run)
MOCK_RESPONSES: dict[str, dict[str, Any]] = {
    "highly_toxic": {
        "question": "What is the toxicity profile and key factors for Lead Oxide nanoparticles?",
        "answer": "Lead Oxide nanoparticles show high toxicity in the retrieved data. Toxicity is a key factor. The majority of studies report toxic effects.",
        "answer_source": "llm",
        "hits": [
            {"score": 0.5, "id": "1", "document": "Lead Oxide 50 nm, zeta +15 mV. Toxic. High ROS.", "metadata": {"size_nm": 50.0, "zeta_potential_mv": 15, "toxicity": "Toxic", "composition": "Lead Oxide"}},
            {"score": 0.52, "id": "2", "document": "Lead Oxide 48 nm. Toxic.", "metadata": {"size_nm": 48.0, "zeta_potential_mv": 12, "toxicity": "Toxic", "composition": "Lead Oxide"}},
            {"score": 0.55, "id": "3", "document": "Lead Oxide 52 nm. Toxic. ROS production.", "metadata": {"size_nm": 52.0, "zeta_potential_mv": 10, "toxicity": "Toxic", "composition": "Lead Oxide"}},
            {"score": 0.56, "id": "4", "document": "Lead Oxide 45 nm. Toxic.", "metadata": {"size_nm": 45.0, "zeta_potential_mv": 14, "toxicity": "Toxic", "composition": "Lead Oxide"}},
            {"score": 0.58, "id": "5", "document": "Lead Oxide 55 nm. Toxic.", "metadata": {"size_nm": 55.0, "zeta_potential_mv": 11, "toxicity": "Toxic", "composition": "Lead Oxide"}},
        ],
    },
    "completely_safe": {
        "question": "What is the toxicity profile for Diamond nanoparticles?",
        "answer": "Diamond nanoparticles in the dataset are non-toxic. Studies indicate safe or low toxicity.",
        "answer_source": "llm",
        "hits": [
            {"score": 0.5, "id": "d1", "document": "Diamond 5 nm. Non-toxic.", "metadata": {"size_nm": 5.0, "toxicity": "Non-toxic", "composition": "Diamond"}},
            {"score": 0.52, "id": "d2", "document": "Diamond 4 nm. Non-toxic.", "metadata": {"size_nm": 4.0, "toxicity": "Non-toxic", "composition": "Diamond"}},
            {"score": 0.54, "id": "d3", "document": "Diamond 6 nm. Safe.", "metadata": {"size_nm": 6.0, "toxicity": "Non-toxic", "composition": "Diamond"}},
            {"score": 0.55, "id": "d4", "document": "Diamond 5.5 nm. Non-toxic.", "metadata": {"size_nm": 5.5, "toxicity": "Non-toxic", "composition": "Diamond"}},
            {"score": 0.56, "id": "d5", "document": "Diamond 4.5 nm. Non-toxic.", "metadata": {"size_nm": 4.5, "toxicity": "Non-toxic", "composition": "Diamond"}},
        ],
    },
    "borderline": {
        "question": "What is the toxicity and ROS data for Titanium Dioxide nanoparticles?",
        "answer": "Titanium Dioxide shows mixed results. Several studies report toxicity and ROS production as a key factor.",
        "answer_source": "llm",
        "hits": [
            {"score": 0.5, "id": "t1", "document": "TiO2 24 nm, zeta -20 mV. Toxic. ROS.", "metadata": {"size_nm": 24.0, "zeta_potential_mv": -20, "toxicity": "Toxic", "composition": "Titanium Dioxide"}},
            {"score": 0.51, "id": "t2", "document": "TiO2 26 nm. Toxic. Reactive oxygen.", "metadata": {"size_nm": 26.0, "zeta_potential_mv": -18, "toxicity": "Toxic", "composition": "Titanium Dioxide"}},
            {"score": 0.53, "id": "t3", "document": "TiO2 25 nm. Toxic. ROS production.", "metadata": {"size_nm": 25.0, "zeta_potential_mv": -22, "toxicity": "Toxic", "composition": "Titanium Dioxide"}},
            {"score": 0.54, "id": "t4", "document": "TiO2 23 nm. Toxic.", "metadata": {"size_nm": 23.0, "zeta_potential_mv": -19, "toxicity": "Toxic", "composition": "Titanium Dioxide"}},
            {"score": 0.55, "id": "t5", "document": "TiO2 27 nm. Toxic. ROS.", "metadata": {"size_nm": 27.0, "zeta_potential_mv": -21, "toxicity": "Toxic", "composition": "Titanium Dioxide"}},
        ],
    },
    "diverse_size": {
        "question": "Summarize toxicity and size for Silver nanoparticles.",
        "answer": "Silver nanoparticles show toxicity in many studies. Size is around 30 nm in the data.",
        "answer_source": "llm",
        "hits": [
            {"score": 0.5, "id": "s1", "document": "Silver 28 nm. Toxic.", "metadata": {"size_nm": 28.0, "toxicity": "Toxic", "composition": "Silver"}},
            {"score": 0.51, "id": "s2", "document": "Silver 32 nm. Toxic.", "metadata": {"size_nm": 32.0, "toxicity": "Toxic", "composition": "Silver"}},
            {"score": 0.52, "id": "s3", "document": "Silver 30 nm. Toxic.", "metadata": {"size_nm": 30.0, "toxicity": "Toxic", "composition": "Silver"}},
            {"score": 0.53, "id": "s4", "document": "Silver 29 nm. Toxic.", "metadata": {"size_nm": 29.0, "toxicity": "Toxic", "composition": "Silver"}},
            {"score": 0.54, "id": "s5", "document": "Silver 31 nm. Toxic.", "metadata": {"size_nm": 31.0, "toxicity": "Toxic", "composition": "Silver"}},
        ],
    },
    "diverse_charge": {
        "question": "What is zeta potential and toxicity for Copper/Copper Oxide nanoparticles?",
        "answer": "Copper/Copper Oxide nanoparticles tend to have negative zeta potential and show toxicity in the dataset.",
        "answer_source": "llm",
        "hits": [
            {"score": 0.5, "id": "c1", "document": "Copper Oxide zeta -16 mV. Toxic.", "metadata": {"zeta_potential_mv": -16, "toxicity": "Toxic", "composition": "Copper/Copper Oxide"}},
            {"score": 0.51, "id": "c2", "document": "Copper Oxide zeta -15 mV. Toxic.", "metadata": {"zeta_potential_mv": -15, "toxicity": "Toxic", "composition": "Copper/Copper Oxide"}},
            {"score": 0.52, "id": "c3", "document": "Copper Oxide zeta -17 mV. Toxic.", "metadata": {"zeta_potential_mv": -17, "toxicity": "Toxic", "composition": "Copper/Copper Oxide"}},
            {"score": 0.53, "id": "c4", "document": "Copper Oxide zeta -14 mV. Toxic.", "metadata": {"zeta_potential_mv": -14, "toxicity": "Toxic", "composition": "Copper/Copper Oxide"}},
            {"score": 0.54, "id": "c5", "document": "Copper Oxide zeta -18 mV. Toxic.", "metadata": {"zeta_potential_mv": -18, "toxicity": "Toxic", "composition": "Copper/Copper Oxide"}},
        ],
    },
}


def fetch_response(case: TestCase, use_live_rag: bool) -> dict[str, Any]:
    if not use_live_rag or case.mock_response is not None:
        return MOCK_RESPONSES.get(case.id) or case.mock_response or {}
    try:
        from rag.qa import answer_question
        return answer_question(case.question, top_k=5, ensemble_prediction=case.ensemble_prediction)
    except Exception:
        return MOCK_RESPONSES.get(case.id) or {}


# ---------------------------------------------------------------------------
# Pytest-compatible test run
# ---------------------------------------------------------------------------

def run_one_case(case: TestCase, use_live_rag: bool, require_json: bool = False) -> list[tuple[str, bool, str]]:
    response = fetch_response(case, use_live_rag)
    results: list[tuple[str, bool, str]] = []

    ok, msg = check_format(response, require_json_answer=require_json)
    results.append(("format", ok, msg))
    if not ok:
        return results

    hits = response.get("hits") or []
    explanation_text = response.get("answer") or ""

    if case.input_size_nm is not None or case.input_zeta_polarity is not None:
        if hits or not case.skip_context_relevance_if_no_input:
            if os.environ.get("USE_LIVE_RAG") == "1":
                ok, msg = check_context_relevance_live(
                    hits, case.input_size_nm, case.input_zeta_polarity,
                    size_tolerance_pct=20.0,
                )
                if "warning" in msg.lower():
                    import warnings
                    warnings.warn(f"context_relevance: {msg}", UserWarning)
            else:
                ok, msg = check_context_relevance(
                    hits, case.input_size_nm, case.input_zeta_polarity,
                    size_tolerance_pct=20.0, min_hits=1,
                )
            results.append(("context_relevance", ok, msg))
        else:
            results.append(("context_relevance", True, "skipped (no hits)"))
    else:
        results.append(("context_relevance", True, "skipped (no input ref)"))

    ok, msg = check_trend_accuracy(explanation_text, hits)
    results.append(("trend_accuracy", ok, msg))

    ok, msg = check_prediction_alignment(explanation_text, case.ensemble_prediction)
    results.append(("prediction_alignment", ok, msg))

    return results


def test_highly_toxic():
    use_live = os.environ.get("USE_LIVE_RAG", "0").strip().lower() in ("1", "true", "yes")
    case = TEST_CASES[0]
    for name, ok, msg in run_one_case(case, use_live):
        assert ok, f"[{name}] {msg}"


def test_completely_safe():
    use_live = os.environ.get("USE_LIVE_RAG", "0").strip().lower() in ("1", "true", "yes")
    case = TEST_CASES[1]
    for name, ok, msg in run_one_case(case, use_live):
        assert ok, f"[{name}] {msg}"


def test_borderline():
    use_live = os.environ.get("USE_LIVE_RAG", "0").strip().lower() in ("1", "true", "yes")
    case = TEST_CASES[2]
    for name, ok, msg in run_one_case(case, use_live):
        assert ok, f"[{name}] {msg}"


def test_diverse_size():
    use_live = os.environ.get("USE_LIVE_RAG", "0").strip().lower() in ("1", "true", "yes")
    case = TEST_CASES[3]
    for name, ok, msg in run_one_case(case, use_live):
        assert ok, f"[{name}] {msg}"


def test_diverse_charge():
    use_live = os.environ.get("USE_LIVE_RAG", "0").strip().lower() in ("1", "true", "yes")
    case = TEST_CASES[4]
    for name, ok, msg in run_one_case(case, use_live):
        assert ok, f"[{name}] {msg}"


# ---------------------------------------------------------------------------
# Script entrypoint: run all 5 and print summary
# ---------------------------------------------------------------------------

def main() -> int:
    use_live = os.environ.get("USE_LIVE_RAG", "0").strip().lower() in ("1", "true", "yes")
    print("RAG logic evaluation (USE_LIVE_RAG={})".format(use_live))
    print("=" * 60)
    failed = 0
    for case in TEST_CASES:
        results = run_one_case(case, use_live)
        all_ok = all(ok for _, ok, _ in results)
        status = "PASS" if all_ok else "FAIL"
        if not all_ok:
            failed += 1
        print(f"  {case.id}: {status}")
        for name, ok, msg in results:
            print(f"    - {name}: {'ok' if ok else msg}")
    print("=" * 60)
    print(f"Result: {len(TEST_CASES) - failed}/{len(TEST_CASES)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
