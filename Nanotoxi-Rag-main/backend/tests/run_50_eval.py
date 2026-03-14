"""
Real-world 50-question RAG evaluation: numeric threshold accuracy + prediction alignment.
Run from backend: python -m tests.run_50_eval
Uses live RAG only (answer_question). Requires built index and thresholds_raw.csv.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# backend on path when run as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from rag.config import DEFAULT_DATA_DIR
from rag.qa import answer_question

THRESHOLDS_CSV = DEFAULT_DATA_DIR / "thresholds_raw.csv"
NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

# Same field specs as threshold_benchmark (numeric ground truth)
FIELD_SPECS = [
    ("Toxic %", "Toxic %", "What is the Toxic % for category '{cat}'? Answer with only the number.", 0.6),
    ("Zeta_mV_median", "Zeta_mV_median", "What is the median zeta potential (Zeta_mV_median) for category '{cat}'? Answer with only the number in mV.", 0.6),
    ("Size_nm_median", "Size_nm_median", "What is the median particle size (Size_nm_median) for category '{cat}'? Answer with only the number in nm.", 0.6),
    ("Hydro_size_nm_median", "Hydro_size_nm_median", "What is the median hydrodynamic size (Hydro_size_nm_median) for category '{cat}'? Answer with only the number in nm.", 0.6),
    ("Exposure_hrs_median", "Exposure_hrs_median", "What is the median exposure time (Exposure_hrs_median) for category '{cat}'? Answer with only the number in hours.", 0.6),
    ("LC50_val_median", "LC50_val_median", "What is the median LC50 value (LC50_val_median) for category '{cat}'? Answer with only the number.", 1.0),
    ("IC50_val_median", "IC50_val_median", "What is the median IC50 value (IC50_val_median) for category '{cat}'? Answer with only the number.", 1.0),
]


def extract_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))
    m = NUM_RE.search(text)
    return float(m.group(1)) if m else None


def check_alignment(explanation_text: str, ensemble_prediction: str) -> bool:
    """True if explanation does not contradict ensemble (TOXIC/SAFE)."""
    if not explanation_text:
        return False
    pred = (ensemble_prediction or "").strip().upper()
    text_lower = explanation_text.lower()
    if pred in ("SAFE", "NON-TOXIC"):
        if "highly toxic" in text_lower or "severe toxicity" in text_lower or "extremely toxic" in text_lower:
            return False
        if re.search(r"\b(?:is|are|shows?|indicates?)\s+toxic\b", text_lower) and "non-toxic" not in text_lower and "low " not in text_lower:
            return False
    if pred in ("TOXIC", "HIGHLY TOXIC"):
        if "completely safe" in text_lower or "no toxicity" in text_lower:
            return False
        if "non-toxic" in text_lower and "not " not in text_lower[:80]:
            return False
    return True


def build_50_questions(df: pd.DataFrame, rng_seed: int = 42) -> list[dict]:
    """Build 50 questions: 35 numeric (5 per field × 7 fields) + 15 profile (alignment)."""
    import random
    random.seed(rng_seed)
    out = []

    # 35 numeric: 5 categories per field, 7 fields
    n_per_field = 5
    for col_name, col, template, tol in FIELD_SPECS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
        eligible = df[df[col].notna()].dropna(subset=["Category"])
        if eligible.empty:
            continue
        n = min(n_per_field, len(eligible))
        sample = eligible.sample(n=n, random_state=rng_seed + hash(col_name) % 10000)
        for _, row in sample.iterrows():
            cat = str(row["Category"]).strip()
            expected = float(row[col])
            q = template.format(cat=cat)
            out.append({
                "type": "numeric",
                "question": q,
                "expected": expected,
                "tolerance": tol,
                "category": cat,
                "field": col,
            })

    # 15 profile: "toxicity profile for X" with ensemble = TOXIC if Toxic % >= 50 else SAFE
    df["Toxic %"] = pd.to_numeric(df.get("Toxic %"), errors="coerce")
    eligible = df[df["Toxic %"].notna()].dropna(subset=["Category"])
    if not eligible.empty:
        n_profile = min(15, len(eligible))
        sample = eligible.sample(n=n_profile, random_state=rng_seed + 999)
        for _, row in sample.iterrows():
            cat = str(row["Category"]).strip()
            toxic_pct = float(row["Toxic %"])
            pred = "TOXIC" if toxic_pct >= 50 else "SAFE"
            q = f"What is the toxicity profile for {cat} nanoparticles? Summarize briefly."
            out.append({
                "type": "profile",
                "question": q,
                "ensemble_prediction": pred,
                "category": cat,
                "toxic_pct": toxic_pct,
            })

    # Pad to 50 with more profile if needed
    profile_eligible = df[df["Toxic %"].notna()].dropna(subset=["Category"]) if "Toxic %" in df.columns else pd.DataFrame()
    while len(out) < 50 and not profile_eligible.empty:
        need = 50 - len(out)
        extra = profile_eligible.sample(n=min(need, len(profile_eligible)), random_state=rng_seed + len(out))
        for _, row in extra.iterrows():
            if len(out) >= 50:
                break
            cat = str(row["Category"]).strip()
            toxic_pct = float(row.get("Toxic %", 0))
            pred = "TOXIC" if toxic_pct >= 50 else "SAFE"
            out.append({
                "type": "profile",
                "question": f"What is the toxicity profile for {cat} nanoparticles?",
                "ensemble_prediction": pred,
                "category": cat,
                "toxic_pct": toxic_pct,
            })
        break

    return out[:50]


def run_eval() -> dict:
    if not THRESHOLDS_CSV.exists():
        return {"error": f"Missing {THRESHOLDS_CSV}", "total": 0, "correct": 0, "accuracy_pct": 0.0}

    df = pd.read_csv(THRESHOLDS_CSV)
    df = df[df["Category"].notna()].copy()
    questions = build_50_questions(df)
    # Pad to 50 with more numeric if needed
    while len(questions) < 50:
        added = 0
        for _col_name, col, template, tol in FIELD_SPECS:
            if len(questions) >= 50:
                break
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
            eligible = df[df[col].notna()].dropna(subset=["Category"])
            for _, row in eligible.iterrows():
                if len(questions) >= 50:
                    break
                cat = str(row["Category"]).strip()
                if any(q.get("category") == cat and q.get("field") == col for q in questions):
                    continue
                questions.append({
                    "type": "numeric",
                    "question": template.format(cat=cat),
                    "expected": float(row[col]),
                    "tolerance": tol,
                    "category": cat,
                    "field": col,
                })
                added += 1
        if added == 0:
            break
    questions = questions[:50]

    numeric_correct = 0
    numeric_total = 0
    profile_correct = 0
    profile_total = 0
    results = []

    for i, q in enumerate(questions):
        if q["type"] == "numeric":
            resp = answer_question(q["question"], top_k=8)
            ans = resp.get("answer", "")
            got = extract_number(ans)
            expected = q["expected"]
            tol = q["tolerance"]
            ok = got is not None and abs(got - expected) <= tol
            numeric_total += 1
            if ok:
                numeric_correct += 1
            results.append({
                "idx": i + 1,
                "type": "numeric",
                "category": q.get("category"),
                "field": q.get("field"),
                "expected": expected,
                "got": got,
                "answer_raw": ans[:80] + "..." if len(ans) > 80 else ans,
                "correct": ok,
            })
        else:
            pred = q.get("ensemble_prediction", "SAFE")
            resp = answer_question(q["question"], top_k=5, ensemble_prediction=pred)
            ans = resp.get("answer", "")
            ok = check_alignment(ans, pred)
            profile_total += 1
            if ok:
                profile_correct += 1
            results.append({
                "idx": i + 1,
                "type": "profile",
                "category": q.get("category"),
                "ensemble_prediction": pred,
                "correct": ok,
                "answer_preview": ans[:80] + "..." if len(ans) > 80 else ans,
            })

    total = numeric_total + profile_total
    correct = numeric_correct + profile_correct
    accuracy_pct = (correct / total * 100.0) if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": round(accuracy_pct, 2),
        "numeric": {"total": numeric_total, "correct": numeric_correct, "accuracy_pct": round(numeric_correct / numeric_total * 100, 2) if numeric_total else 0},
        "profile": {"total": profile_total, "correct": profile_correct, "accuracy_pct": round(profile_correct / profile_total * 100, 2) if profile_total else 0},
        "results": results,
    }


def main() -> int:
    print("Running 50-question real-world RAG evaluation (live)...")
    print()
    data = run_eval()
    if data.get("error"):
        print("ERROR:", data["error"])
        return 1
    total = data["total"]
    correct = data["correct"]
    acc = data["accuracy_pct"]
    print("=" * 60)
    print("50-QUESTION REAL-WORLD ACCURACY REPORT")
    print("=" * 60)
    print(f"  Total questions:     {total}")
    print(f"  Correct:             {correct}")
    print(f"  Overall accuracy:    {acc}%")
    print()
    if data.get("numeric", {}).get("total"):
        n = data["numeric"]
        print(f"  Numeric (threshold): {n['correct']}/{n['total']}  ({n['accuracy_pct']}%)")
    if data.get("profile", {}).get("total"):
        p = data["profile"]
        print(f"  Profile (alignment): {p['correct']}/{p['total']}  ({p['accuracy_pct']}%)")
    print("=" * 60)
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
